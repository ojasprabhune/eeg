import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from eeg.big_hand.position_llm.vqvae import VQVAE
from eeg.big_hand.position_llm import AppendageDataset

lr = 3e-4
device = "cuda"
epochs = 10000

vqvae = VQVAE(input_dim=12,
              codebook_size=512,
              embedding_dim=16)
vqvae.to(device)

optimizer = optim.AdamW(vqvae.parameters(), lr=lr)
loss_fn = nn.MSELoss()
commitment_beta = 0.25

appendage_dataset: AppendageDataset = AppendageDataset()
appendage_dataloader = DataLoader(appendage_dataset, batch_size=32, shuffle=True)

def train():
    # run = wandb.init(
    #     name="vq_vae_10_000",
    #     entity="prabhuneojas-evergreen-valley-high-school",
    #     project="eeg",
    #     config={
    #         "learning_rate": lr,
    #         "architecture": "transformer",
    #         "dataset": "region_dataset",
    #         "epochs": epochs,
    #     },
    # )

    iter_tqdm = tqdm(range(epochs))
    for i in tqdm(range(epochs)):
        iter_tqdm.set_description(f"Epoch {i + 1}")
        for region_batch, appendage_batch in appendage_dataloader:
            x_reconstructed, z_e, z_q = vqvae(appendage_batch.to(torch.float32).to(device))

            recon_loss = loss_fn(x_reconstructed, appendage_batch.to(torch.float32).to(device))
            commitment_loss = loss_fn(z_e, z_q.detach())

            total_loss = recon_loss + commitment_beta * commitment_loss
            print(recon_loss)
            print(commitment_loss)

            iter_tqdm.set_postfix({"loss": total_loss.item()})
            # run.log({"loss": total_loss.item()})

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if i == 5:
            break

        if i % 2000 == 0:

            checkpoint = {
                "epoch": i,
                "model": vqvae.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(checkpoint, f"/var/log/thavamount/eeg_ckpts/vqvae.pth")

    # run.finish()

train()
