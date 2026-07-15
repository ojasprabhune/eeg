"""
Script to train the language model in the language model dataset, given many
sequences of sentences, trying to translate between EEG sequence probabilities
and sentence tokens.
"""

import torch
import yaml
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from eeg.language_model import LanguageDataset, LanguageModel

# --- configuration -----------------------------------------------------------

with open("config/language_model.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

    vocab_size = config["vocab_size"]
    num_layers = config["num_layers"]
    decoder_num_layers = config["decoder_num_layers"]
    num_heads = config["num_heads"]
    num_classes = config["num_classes"]
    embedding_dim = config["embedding_dim"]
    decoder_embedding_dim = config["decoder_embedding_dim"]
    ffn_hidden_dim = config["ffn_hidden_dim"]
    qk_length = config["qk_length"]
    value_length = config["value_length"]
    max_length = config["max_length"]

    encoder_dropout = config["encoder_dropout"]
    decoder_dropout = config["decoder_dropout"]

    recon_lambda = config["recon_lambda"]

    device = config["device"]
    batch_size = config["batch_size"]
    warmup_steps = config["warmup_steps"]
    base_lr = float(config["base_lr"])
    epochs = config["epochs"]

    min_value = config["min_value"]
    k = config["k"]

    run_name = config["run_name"]
    use_ckpt_path = config["use_ckpt_path"]
    save_ckpt_path = config["save_ckpt_path"]
    save_every = config["save_every"]

# --- data --------------------------------------------------------------------

train_language_dataset = LanguageDataset(
    num_classes=num_classes,
    mode="train",
    print_shapes=False,
)

val_language_dataset = LanguageDataset(
    num_classes=num_classes,
    mode="val",
    print_shapes=False,
)

train_language_dataloader = DataLoader(
    train_language_dataset,
    batch_size=32,
    shuffle=True,
)
val_language_dataloader = DataLoader(
    val_language_dataset,
    batch_size=32,
    shuffle=False,
)

# --- model -------------------------------------------------------------------

model = LanguageModel(
    vocab_size=vocab_size,
    num_layers=num_layers,
    decoder_num_layers=decoder_num_layers,
    num_heads=num_heads,
    num_inputs_classes=num_classes,
    decoder_embedding_dim=decoder_embedding_dim,
    ffn_hidden_dim=ffn_hidden_dim,
    encoder_dropout=encoder_dropout,
    decoder_dropout=decoder_dropout,
    k=k,
    min_value=min_value,
).to(device)

# --- training ----------------------------------------------------------------

optimizer = AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9)
ce_loss_fn = CrossEntropyLoss(ignore_index=0)  # ignore <PAD> token id
mse_loss_fn = MSELoss()

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of model parameters: {param_count:,}")

# --- checkpoint --------------------------------------------------------------

if use_ckpt_path is not None:
    state_dict = torch.load(use_ckpt_path, map_location=device)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    start = state_dict["epochs"]
    print(f"Loaded checkpoint from {use_ckpt_path}")
else:
    start = 0

# --- validation --------------------------------------------------------------


def validate() -> tuple[float, float]:
    """
    Run one pass over the validation set, return (loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    step = 0

    with torch.no_grad():
        for feature, feature_mask, label, label_mask, _ in val_language_dataloader:
            feature = feature.to(device)
            feature_mask = feature_mask.to(device).bool()
            label = label.to(device).to(torch.int64)
            label_mask = label_mask.to(device).bool()

            valid = label_mask[:, 1:]  # align with gt_label

            in_feature = feature[:, :-1, :]
            in_feature_mask = feature_mask[:, :-1]
            in_label = label[:, :-1]
            in_label_mask = label_mask[:, :-1]

            gt_feature = feature[:, 1:, :]
            gt_label = label[:, 1:]

            label_logits, recon = model(
                src=in_feature,
                tgt=in_label,
                src_pad_mask=~in_feature_mask,  # flip because 1 should mean padding
                tgt_pad_mask=~in_label_mask,
                step=step,
                return_epsilon=False,
            )  # out: (B, seq_len, vocab_size)

            label_logits = label_logits.transpose(1, 2)  # (B, vocab_size, seq_len)

            loss_letters = ce_loss_fn(label_logits, gt_label)
            loss_recon = mse_loss_fn(recon, gt_feature)

            loss = loss_letters + recon_lambda * loss_recon

            total_loss += loss.item() * batch_size
            preds = label_logits.argmax(dim=1)

            correct += ((preds == gt_label) & valid).sum().item()
            total += valid.sum().item()

            all_preds.append(preds.cpu())
            all_labels.append(gt_label.cpu())

            step += 1

    model.train()
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# --- training ----------------------------------------------------------------


def train():
    run = wandb.init(
        name=run_name,
        entity="prabhuneojas-evergreen-valley-high-school",
        project="eeg",
        config={
            "learning_rate": base_lr,
            "architecture": "Transformer",
            "dataset": "language_dataset",
            "epochs": epochs,
        },
    )

    wandb.log({"param_count": param_count})
    model.train()

    step = 0

    epoch_tqdm = tqdm(range(start, epochs), dynamic_ncols=True)
    for i in epoch_tqdm:
        epoch_tqdm.set_description(f"Epoch {i + 1}")

        iter_tqdm = tqdm(train_language_dataloader, dynamic_ncols=True)
        for feature, feature_mask, label, label_mask, _ in iter_tqdm:
            feature = feature.to(device)
            feature_mask = feature_mask.to(device).bool()
            label = label.to(device).to(torch.int64)
            label_mask = label_mask.to(device).bool()

            in_feature = feature[:, :-1, :]
            in_feature_mask = feature_mask[:, :-1]
            in_label = label[:, :-1]
            in_label_mask = label_mask[:, :-1]

            gt_feature = feature[:, 1:, :]
            gt_label = label[:, 1:]

            label_logits, recon, epsilon = model(
                src=in_feature,
                tgt=in_label,
                src_pad_mask=~in_feature_mask,  # flip because 1 should mean padding
                tgt_pad_mask=~in_label_mask,
                step=step,
                return_epsilon=True,
            )  # out: (B, seq_len, vocab_size)

            label_logits = label_logits.transpose(1, 2)  # (B, vocab_size, seq_len)

            loss_letters = ce_loss_fn(label_logits, gt_label)
            loss_recon = mse_loss_fn(recon, gt_feature)

            loss = loss_letters + recon_lambda * loss_recon

            iter_tqdm.set_postfix({"loss": loss.item()})
            run.log(
                {
                    "loss": loss.item(),
                    "letter loss": loss_letters.item(),
                    "recon loss": loss_recon.item(),
                    "epsilon": epsilon,
                }
            )

            optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
            loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly

            step += 1

        val_loss, val_acc = validate()
        epoch_tqdm.set_postfix(
            {
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.3f}",
            }
        )
        run.log(
            {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": i + 1,
            }
        )

        if (i + 1) % save_every == 0 and save_ckpt_path is not None:
            latest_ckpt = {
                "epochs": i,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(latest_ckpt, f"{save_ckpt_path}_epoch_{i + 1}.pth")

    run.finish()

    latest_ckpt = {
        "epochs": epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if save_ckpt_path is not None:
        torch.save(latest_ckpt, save_ckpt_path)


train()
