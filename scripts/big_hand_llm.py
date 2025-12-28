"""
Train a Position LLM to generalize to Ojas's hand movements. Implement all
previous techniques like VQVAE tokenization on appendage vector components.
"""

import torch
import wandb
import argparse
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from eeg.big_hand.position_llm import PositionLLM, AppendageDataset
from eeg.big_hand.position_llm.vqvae import VQVAE

# TODO add arg prase for hyperparameters and experiment name
parser = argparse.ArgumentParser()
parser.add_argument("experiment_name", type=str)
parser.add_argument("embedding_dim", type=int)
parser.add_argument("epochs", type=int)
args = parser.parse_args()

experiment_name = args.experiment_name
embedding_dim = args.embedding_dim
epochs = args.epoch

appendage_dataset: AppendageDataset = AppendageDataset(duration_prediction=True)

appendage_dataloader = DataLoader(appendage_dataset, 
                                  batch_size=32, 
                                  shuffle=True, 
                                  collate_fn=appendage_dataset.collate_fn)

print(
    f"Raw positions shape: {appendage_dataset.train_data.shape}, expected: (2, T, 63)"
)
print(
    f"Appendage data shape: {appendage_dataset.app_data.shape}, expected: (T, 12)")

warmup_steps = 4000
base_lr = 5e-5


def lr_lambda(step):
    if step == 0:
        step = 1  # avoid div by zero
    if step < warmup_steps:
        return step / warmup_steps
    else:
        return (warmup_steps**0.5) / (step**0.5)


device = "cuda"
epochs = 10000

model = PositionLLM(
    vocab_size=512,
    num_layers=4,
    num_heads=4,
    embedding_dim=64,
    ffn_hidden_dim=64,
    qk_length=64,
    value_length=64,
    max_length=2048,
    dropout=0.1,
    duration_prediction=True,
)

optimizer = AdamW(model.parameters(), lr=base_lr, betas=[0.9, 0.98], eps=1e-9)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

class_loss_fn = CrossEntropyLoss(reduction="none")
duration_loss_fn = L1Loss(reduction="none")
lambda_appendage_loss = 1

# state_dict = torch.load("/var/log/thavamount/eeg_ckpts/checkpoint0.pth", map_location="cuda")
# model.load_state_dict(state_dict["model"])
# optimizer.load_state_dict(state_dict["optimizer"])
# scheduler.load_state_dict(state_dict["scheduler"])

model.to(device)


def train():
    # start a new wandb run to track this script.
    run = wandb.init(
        name="VQVAE_big_hand",
        # set the wandb entity where your project will be logged (generally your team name).
        entity="prabhuneojas-evergreen-valley-high-school",
        # set the wandb project where this run will be logged.
        project="eeg",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": base_lr,
            "architecture": "transformer",
            "dataset": "region_dataset",
            "epochs": epochs,
        },
    )

    epoch_tqdm = tqdm(range(epochs), dynamic_ncols=True)
    for i in epoch_tqdm:
        epoch_tqdm.set_description(f"Epoch {i + 1}")

        iter_tqdm = tqdm(appendage_dataloader, dynamic_ncols=True)
        for batch in iter_tqdm:

            tokens = batch["tokens"]  # (B, T)
            durations = batch["durations"]  # (B, T)
            masks = batch["masks"]  # (B, T)

            in_tokens = tokens[:, :-1].to(torch.int64).to(device)
            gt_tokens = tokens[:, 1:].to(torch.int64).to(device)

            token_logits, duration_preds = model(in_tokens)

            # cross entropy loss expects (B, C, *additional_dims)
            token_logits = token_logits.transpose(1, 2)

            token_loss = class_loss_fn(token_logits, gt_tokens)
            token_loss = (token_loss * masks[:, 1:]).mean()

            duration_loss = duration_loss_fn(duration_preds, durations[:, 1:])
            duration_loss = (duration_loss * masks[:, 1:]).mean()

            total_loss = token_loss + duration_loss

            iter_tqdm.set_postfix({"loss": total_loss.item()})
            run.log({"loss": total_loss.item()})

            optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
            total_loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly
            scheduler.step()

        if i % 200 == 0:
            # plot_out(model)

            latest_ckpt = {
                "epoch": i,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }

            torch.save(
                latest_ckpt, f"/var/log/thavamount/eeg_ckpts/hand_lm/posllm_duration_prediction.pth")

        elif i % 5000 == 0:
            # plot_out(model)

            checkpoint = {
                "epoch": i,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }

            torch.save(
                checkpoint, f"/var/log/thavamount/eeg_ckpts/hand_lm/checkpoint_{i}.pth")

    run.finish()

train()
