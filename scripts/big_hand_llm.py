"""
Train a Position LLM to generalize to Ojas's hand movements. Implement all
previous techniques like VQVAE tokenization on appendage vector components.
"""

import yaml
import torch
import wandb
from torch.nn import CrossEntropyLoss, L1Loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from eeg.big_hand.position_llm import PositionLLM, AppendageDataset

with open("config/log_position_llm.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

    batch_size = config["hyperparameters"]["batch_size"]
    warmup_steps = config["hyperparameters"]["warmup_steps"]
    base_lr = float(config["hyperparameters"]["base_lr"])
    epochs = config["hyperparameters"]["epochs"]
    vocab_size = config["hyperparameters"]["vocab_size"]
    num_layers = config["hyperparameters"]["num_layers"]
    num_heads = config["hyperparameters"]["num_heads"]
    embedding_dim = config["hyperparameters"]["embedding_dim"]
    ffn_hidden_dim = config["hyperparameters"]["ffn_hidden_dim"]
    qk_length = config["hyperparameters"]["qk_length"]
    value_length = config["hyperparameters"]["value_length"]
    max_length = config["hyperparameters"]["max_length"]
    dropout = config["hyperparameters"]["dropout"]
    duration_prediction = config["hyperparameters"]["duration_prediction"]
    lambda_duration = config["hyperparameters"]["lambda_duration"]
    run_name = config["hyperparameters"]["run_name"]
    use_ckpt_path = config["hyperparameters"]["use_ckpt_path"]
    save_ckpt_path = config["hyperparameters"]["save_ckpt_path"]

appendage_dataset: AppendageDataset = AppendageDataset(
    duration_prediction=True)

appendage_dataloader = DataLoader(appendage_dataset,
                                  batch_size=32,
                                  shuffle=True,
                                  collate_fn=AppendageDataset.collate_fn)

print(
    f"Raw positions shape: {appendage_dataset.train_data.shape}, expected: (2, T, 63)"
)
print(
    f"Appendage data shape: {appendage_dataset.app_data.shape}, expected: (T, 12)")


def lr_lambda(step):
    if step == 0:
        step = 1  # avoid div by zero
    if step < warmup_steps:
        return step / warmup_steps
    else:
        return (warmup_steps**0.5) / (step**0.5)


device = "cuda"

model = PositionLLM(
    vocab_size=vocab_size,
    num_layers=num_layers,
    num_heads=num_heads,
    embedding_dim=embedding_dim,
    ffn_hidden_dim=ffn_hidden_dim,
    qk_length=qk_length,
    value_length=value_length,
    max_length=max_length,
    dropout=dropout,
    duration_prediction=duration_prediction,
)

optimizer = AdamW(model.parameters(), lr=base_lr, betas=[0.9, 0.98], eps=1e-9)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

class_loss_fn = CrossEntropyLoss(reduction="none")
duration_loss_fn = L1Loss(reduction="none")

model.to(device)

if use_ckpt_path is not None:
    print(f"Loading checkpoint from {use_ckpt_path}")
    state_dict = torch.load(use_ckpt_path, map_location=device)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    # scheduler.load_state_dict(state_dict["scheduler"])
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    print("Freezing everything except duration prediction head.")
    model.embedding.weight.requires_grad = False
    for i, module in enumerate(model.positionllm_layers):
        if i < len(model.positionllm_layers) - 1:
            for param in module.parameters():
                param.requires_grad = False
    # model.linear.weight.requires_grad = False
    # model.linear.bias.requires_grad = False


def train():
    # start a new wandb run to track this script.
    run = wandb.init(
        name=run_name,
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
            durations = durations.to(device)
            masks = masks.to(device)

            # shape (B, T, vocab_size of vqvae tokens), (B, T,)
            token_logits, duration_preds = model(in_tokens)

            # cross entropy loss expects (B, C, *additional_dims)
            token_logits = token_logits.transpose(1, 2)

            token_loss = class_loss_fn(token_logits, gt_tokens)
            token_loss = (
                token_loss * masks[:, 1:]).sum() / (masks[:, 1:].sum() + 1e-8)

            duration_loss = duration_loss_fn(duration_preds, durations[:, 1:])
            duration_loss = (
                duration_loss * masks[:, 1:]).sum() / (masks[:, 1:].sum() + 1e-8)

            total_loss = token_loss + lambda_duration * duration_loss

            loss_report = {
                "loss": total_loss.item(),
                "token_loss": token_loss.item(),
                "duration_loss": duration_loss.item()
            }

            iter_tqdm.set_postfix(loss_report)
            run.log(loss_report)

            optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
            total_loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly
            scheduler.step()

        if i % 200 == 0:
            latest_ckpt = {
                "epoch": i,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }

            torch.save(
                latest_ckpt, save_ckpt_path)

        elif i % 5000 == 0:
            checkpoint = {
                "epoch": i,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }

            torch.save(
                checkpoint, f"/var/log/thavamount/eeg_ckpts/hand_lm/{run_name}_epoch_{i}.pth")

    run.finish()


train()
