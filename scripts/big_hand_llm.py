"""
Train a Position LLM to generalize to Ojas's hand movements. Implement all
previous techniques like region tokenization on appendage vector components.
"""

import torch
import wandb
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from eeg.big_hand.position_llm import E2EPositionLLM, AppendageDataset

appendage_dataset: AppendageDataset = AppendageDataset()

appendage_dataloader = DataLoader(
    appendage_dataset, batch_size=32, shuffle=True)

print(
    f"Raw positions shape: {appendage_dataset.train_data.shape}, expected: (2, T, 63)"
)
print(
    f"Appendage data shape: {appendage_dataset.app_data.shape}, expected: (T, 12)")
print(
    f"Region tokens shape: {appendage_dataset.region_tokens.shape}, expected: (T)")

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

model = E2EPositionLLM()  # end to end position llm
model.to(device)
optimizer = AdamW(model.parameters(), lr=base_lr, betas=[0.9, 0.98], eps=1e-9)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
class_loss_fn = CrossEntropyLoss()
appendage_loss_fn = MSELoss()
lambda_appendage_loss = 1

# state_dict = torch.load("/var/log/thavamount/eeg_ckpts/checkpoint0.pth", map_location="cuda")
# model.load_state_dict(state_dict["model"])
# optimizer.load_state_dict(state_dict["optimizer"])
# scheduler.load_state_dict(state_dict["scheduler"])


def train(train_appendage: bool = True):
    # start a new wandb run to track this script.
    run = wandb.init(
        name="big_hand_10_000",
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

    model.to(device)
    iter_tqdm = tqdm(range(epochs))
    for i in tqdm(range(epochs)):
        iter_tqdm.set_description(f"Epoch {i + 1}")
        # iter_tqdm = appendage_dataloader
        for region_batch, appendage_batch in appendage_dataloader:
            in_region_tokens = (
                region_batch[:, :-1].to(torch.int64).to(device)
            )  # (1, T-1,)
            gt_region_tokens = (
                region_batch[:, 1:].to(torch.int64).to(device)
            )  # (1, T-1,)

            gt_appendange = (
                appendage_batch[:, 1:].to(torch.float32).to(device)
            )  # (1, T-1, 63)

            region_logits, pred_appendage = model(in_region_tokens)

            # cross entropy loss expects (B, C, *additional_dims)
            region_logits = region_logits.transpose(1, 2)

            region_loss = class_loss_fn(region_logits, gt_region_tokens)

            # combine losses to optimize for all at once
            if train_appendage:
                appendage_loss = appendage_loss_fn(pred_appendage, gt_appendange)
                total_loss = region_loss + lambda_appendage_loss * appendage_loss
            else:
                total_loss = region_loss

            iter_tqdm.set_postfix({"loss": total_loss.item()})
            run.log({"loss": total_loss.item()})

            optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
            total_loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly
            scheduler.step()
        if i % 500 == 0:
            # plot_out(model)

            latest_ckpt = {
                "epoch": i,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }

            torch.save(latest_ckpt, f"/var/log/thavamount/eeg_ckpts/e2e_posllm_latest.pth")

        elif i % 5000 == 0:
            # plot_out(model)

            checkpoint = {
                "epoch": i,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }

            torch.save(checkpoint, f"/var/log/thavamount/eeg_ckpts/checkpoint_{i}.pth")


    run.finish()
    model.to("cpu")


region_tokens, appendage_values = appendage_dataset[0]
print(appendage_values)


def inference(model: torch.nn.Module) -> torch.Tensor:
    all_region_tokens = region_tokens.unsqueeze(0).to(torch.int64)

    first_region_token = all_region_tokens[:, 0]

    region_tokens_so_far = [first_region_token.item()]
    appendage_values_so_far = [torch.tensor(appendage_values[0])]

    for _ in range(100):
        region_token_logits, pred_appendage = model(
            torch.tensor(region_tokens_so_far).unsqueeze(0)
        )

        best_region_token = torch.argmax(region_token_logits.squeeze(0)[-1])
        region_tokens_so_far.append(best_region_token.item())

        last_pred_appendage = pred_appendage.squeeze(0)[-1]  # (12,)
        appendage_values_so_far.append(last_pred_appendage)

    appendage_values_so_far = torch.stack(appendage_values_so_far).detach()
    return region_tokens_so_far, appendage_values_so_far  # (T, 12)


def plot_out(model):
    out, pred_appendage = inference(model)

    fig, ax = plt.subplots(1, 5, figsize=(12, 4))

    ax[0].plot(out[:200], label="pred")
    ax[0].plot(region_tokens[:200], label="region true")
    ax[0].legend()

    ax[1].plot(pred_appendage[:200, 0], label="pred")
    ax[1].plot(appendage_values[:200, 0], label="app 0")
    ax[1].legend()

    ax[2].plot(pred_appendage[:200, 4], label="pred")
    ax[2].plot(appendage_values[:200, 4], label="app 4")
    ax[2].legend()

    ax[3].plot(pred_appendage[:200, 7], label="pred")
    ax[3].plot(appendage_values[:200, 7], label="app 7")
    ax[3].legend()

    ax[4].plot(pred_appendage[:200, 11], label="pred")
    ax[4].plot(appendage_values[:200, 11], label="app 13")
    ax[4].legend()

    plt.show()


# plot_out(model)
train(train_appendage=True)
# plot_out(model)
train(train_appendage=False)
