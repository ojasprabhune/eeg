"""
Script to train the language model in the language model dataset, given many
sequences of sentences, trying to translate between EEG sequence probabilities
and sentence tokens.
"""

import torch
import yaml
import yaml
import torch
import wandb
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from eeg.language_model import LanguageDataset
from eeg.language_model import LanguageModel

with open("config/language_model.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

    vocab_size = config["vocab_size"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    num_classes = config["num_classes"]
    embedding_dim = config["embedding_dim"]
    ffn_hidden_dim = config["ffn_hidden_dim"]
    qk_length = config["qk_length"]
    value_length = config["value_length"]
    max_length = config["max_length"]
    dropout = config["dropout"]

    device = config["device"]
    batch_size = config["batch_size"]
    warmup_steps = config["warmup_steps"]
    base_lr = float(config["base_lr"])
    epochs = config["epochs"]

    run_name = config["run_name"]
    use_ckpt_path = config["use_ckpt_path"]
    save_ckpt_path = config["save_ckpt_path"]
    save_every = config["save_every"]

language_dataset = LanguageDataset(
    features_sequences=torch.tensor(0),
    print_shapes=True,
)

language_dataloader = DataLoader(language_dataset, batch_size=32, shuffle=True)

model = LanguageModel(
    vocab_size=vocab_size,
    num_layers=num_layers,
    num_heads=num_heads,
    embedding_dim=embedding_dim,
    ffn_hidden_dim=ffn_hidden_dim,
    qk_length=qk_length,
    value_length=value_length,
    max_length=max_length,

    dropout=dropout
).to(device)

optimizer = AdamW(model.parameters(), lr=base_lr, betas=[0.9, 0.98], eps=1e-9)
loss_fn = CrossEntropyLoss(reduction="none")

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of model parameters: {param_count:,}")

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
    model.to(device)
    model.train()

    epoch_tqdm = tqdm(range(epochs), dynamic_ncols=True)
    for i in epoch_tqdm:
        epoch_tqdm.set_description(f"Epoch {i + 1}")

        iter_tqdm = tqdm(language_dataloader, dynamic_ncols=True)
        for feature, mask, label in iter_tqdm:
            # chunk: (B, T, C)
            print(feature.shape)
            print(mask.shape)
            print(label.shape)

            feature = feature.to(device)
            mask = mask.to(device)
            label = label.to(device)

            label_logits = model(src=feature, tgt=label, mask=mask)  # out: (B, seq_len, vocab_size)

            print(label_logits.shape)

            quit()
            loss = loss_fn(label_logits, labels)

            iter_tqdm.set_postfix({"loss": loss.item()})
            run.log({"loss": loss.item()})

            optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
            loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly
            scheduler.step()  # steps lr

        if (i + 1) % save_every == 0:
            latest_ckpt = {
                "epochs": i,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(latest_ckpt, f"{save_ckpt_path}_epoch_{i + 1}.pth")

    run.finish()
#
#
train()
#
# latest_ckpt = {
#     "epochs": epochs,
#     "model": model.state_dict(),
#     "optimizer": optimizer.state_dict(),
# }
#
# torch.save(latest_ckpt, save_ckpt_path)
