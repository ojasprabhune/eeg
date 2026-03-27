import wandb
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from eeg.big_hand.position_llm.vqvae import VQVAE
from eeg.big_hand.position_llm import AppendageDataset

lr = 3e-4
device = "cuda"
epochs = 2000

parser = argparse.ArgumentParser()
parser.add_argument(
    "checkpoint_name", type=str, help="The name of the saved checkpoint"
)
args = parser.parse_args()

checkpoint_name = args.checkpoint_name

vqvae = VQVAE(input_dim=12, codebook_size=512, embedding_dim=1024)
vqvae.to(device)

optimizer = optim.AdamW(vqvae.parameters(), lr=lr)
loss_fn = nn.MSELoss()
commitment_beta = 0.25

appendage_dataset: AppendageDataset = AppendageDataset(
    data_path="/var/log/thavamount/eeg_dataset"
)
appendage_dataloader = DataLoader(appendage_dataset, batch_size=32, shuffle=True)


def save_checkpoint(epoch: int, latest: bool) -> None:
    if latest:
        checkpoint = {
            "epoch": epoch,
            "model": vqvae.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        torch.save(
            checkpoint, f"/var/log/thavamount/eeg_ckpts/eeg_vqvae/{checkpoint_name}.pth"
        )
    else:
        checkpoint = {
            "epoch": epoch,
            "model": vqvae.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        torch.save(
            checkpoint,
            f"/var/log/thavamount/eeg_ckpts/eeg_vqvae/{checkpoint_name}_{epoch}.pth",
        )


def train():
    run = wandb.init(
        name="vqvae_final",
        entity="prabhuneojas-evergreen-valley-high-school",
        project="eeg",
        config={
            "learning_rate": lr,
            "architecture": "transformer",
            "dataset": "region_dataset",
            "epochs": epochs,
        },
    )

    for epoch in range(epochs):
        for region_batch, appendage_batch in tqdm(
            appendage_dataloader,
            desc=f"Epoch {epoch + 1}",
            dynamic_ncols=True,
        ):
            x_reconstructed, z_e, z_q = vqvae(
                appendage_batch.to(torch.float32).to(device)
            )

            recon_loss = loss_fn(
                x_reconstructed, appendage_batch.to(torch.float32).to(device)
            )
            commitment_loss = loss_fn(z_e, z_q.detach())

            total_loss = recon_loss + commitment_beta * commitment_loss

            run.log({"loss": total_loss.item()})

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if epoch % 250 == 0:
            save_checkpoint(epoch=epoch, latest=False)

    run.finish()


train()
save_checkpoint(epochs, latest=True)
