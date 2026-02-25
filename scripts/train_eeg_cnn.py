import yaml
import torch
import wandb
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from eeg.eeg_data import HandDatasetCNN, EEGCNN

with open("config/eeg_basic_cnn.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

    vocab_size = config["vocab_size"]
    ffn_embedding_dim = config["ffn_embedding_dim"]
    num_features = config["num_features"]
    kernel_size_temporal = config["kernel_size_temporal"]
    kernel_size_spatial = config["kernel_size_spatial"]
    kernel_size_avg_pool = config["kernel_size_avg_pool"]

    device = config["device"]
    batch_size = config["batch_size"]
    warmup_steps = config["warmup_steps"]
    base_lr = float(config["base_lr"])
    epochs = config["epochs"]

    run_name = config["run_name"]
    use_ckpt_path = config["use_ckpt_path"]
    save_ckpt_path = config["save_ckpt_path"]
    save_every = config["save_every"]


hand_dataset_cnn: HandDatasetCNN = HandDatasetCNN(num_folders=32)

hand_dataloader = DataLoader(
    hand_dataset_cnn, batch_size=batch_size, shuffle=True)

model = EEGCNN(seq_len=hand_dataset_cnn.train_eeg_chunks.shape[-1],
               num_features=num_features,
               kernel_size_temporal=kernel_size_temporal,
               kernel_size_spatial=kernel_size_spatial,
               kernel_size_avg_pool=kernel_size_avg_pool,
               ffn_embedding_dim=ffn_embedding_dim,
               vocab_size=vocab_size
               )

optimizer = AdamW(model.parameters(), lr=base_lr, betas=[0.9, 0.98], eps=1e-9)
loss_fn = CrossEntropyLoss(reduction="none")

model.to(device)


def train():
    run = wandb.init(
        name=run_name,
        entity="prabhuneojas-evergreen-valley-high-school",
        project="eeg",
        config={
            "learning_rate": base_lr,
            "architecture": "linear",
            "dataset": "physionet_dataset",
            "epochs": epochs,
        },
    )

    model.to(device)

    epoch_tqdm = tqdm(range(epochs), dynamic_ncols=True)
    for i in epoch_tqdm:
        epoch_tqdm.set_description(f"Epoch {i + 1}")

        iter_tqdm = tqdm(hand_dataloader, dynamic_ncols=True)
        for eeg_chunk, label_chunk, mask in iter_tqdm:
            # eeg_chunk: (B, C, T)
            # label_chunk: (B,)

            eeg_chunk = eeg_chunk.to(device).float()
            label_chunk = label_chunk.to(device)
            mask = mask.to(device)

            train_hand_pos_logits = model(eeg_chunk)  # out: (B, vocab_size)

            train_loss = loss_fn(train_hand_pos_logits, label_chunk)

            # mask
            train_loss = (train_loss * mask).sum() / (mask.sum() + 1e-8)

            iter_tqdm.set_postfix({"loss": train_loss.item()})
            run.log({"train loss": train_loss.item()})

            optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
            train_loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly

        if (i + 1) % save_every == 0:
            latest_ckpt = {
                "epochs": i,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(latest_ckpt, f"{save_ckpt_path}_epoch_{i + 1}.pth")

        model.eval()
        val_eeg_chunk, val_label_chunk, _ = hand_dataset_cnn.get_validation_data(
            i)
        val_eeg_chunk = torch.tensor(val_eeg_chunk).to(
            device).float().unsqueeze(0)  # (1, C, T)
        val_label_chunk = val_label_chunk.to(device).unsqueeze(0)  # (1,)

        val_hand_pos_logits = model(val_eeg_chunk)  # out: (1, vocab_size)

        val_loss = loss_fn(val_hand_pos_logits, val_label_chunk)

        run.log({"val loss": val_loss.item()})
        model.train()

    run.finish()


train()

latest_ckpt = {
    "epochs": epochs,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}

torch.save(latest_ckpt, save_ckpt_path)
