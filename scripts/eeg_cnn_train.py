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

    device = config["device"]
    batch_size = config["batch_size"]
    warmup_steps = config["warmup_steps"]
    base_lr = float(config["base_lr"])
    epochs = config["epochs"]

    run_name = config["run_name"]
    use_ckpt_path = config["use_ckpt_path"]
    save_ckpt_path = config["save_ckpt_path"]


hand_dataset: HandDatasetCNN = HandDatasetCNN(num_folders=5)

hand_dataloader = DataLoader(hand_dataset, batch_size=32, shuffle=True)

model = EEGCNN(seq_len=hand_dataset.eeg_chunks.shape[-1])

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
        for chunk, label_chunk, mask in iter_tqdm:
            # chunk: (B, num_channels, T)

            chunk = chunk.to(device).float()
            label_chunk = label_chunk.to(device)
            mask = mask.to(device)

            in_labels = label_chunk[:, :-1]
            gt_labels = label_chunk[:, 1:]

            hand_pos_logits = model(chunk.float(), in_labels) # out: (B, T, vocab_size)
            hand_pos_logits = hand_pos_logits.transpose(1, 2) # (B, T, vocab_size) -> (B, vocab_size, T)

            loss = loss_fn(hand_pos_logits, gt_labels)
            loss = (loss * mask[:, 1:]).sum() / (mask[:, 1:].sum() + 1e-8)

            iter_tqdm.set_postfix({"loss": loss.item()})
            run.log({"loss": loss.item()})

            optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
            loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly

        if (i + 1) % 10 == 0:
            latest_ckpt = {
                "epochs": i,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(latest_ckpt, f"{save_ckpt_path}_epoch_{i + 1}.pth")

    run.finish()


train()

latest_ckpt = {
    "epochs": epochs,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}

torch.save(latest_ckpt, save_ckpt_path)
