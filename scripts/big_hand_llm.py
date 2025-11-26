"""
Train a Position LLM to generalize to Ojas's hand movements. Implement all
previous techniques like region tokenization on appendage vector components.

TODO:
    - hyperparameter tuning
"""

import torch
import wandb
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from eeg.big_hand.position_llm import E2EPositionLLM, RegionDataset, AppendageDataset

# import region data set to iterate over batches
# region_dataset: RegionDataset = RegionDataset(
#     "data/relative_old/", seq_len=100)
appendage_dataset: AppendageDataset = AppendageDataset("data")

# region_dataloader = DataLoader(region_dataset, batch_size=16, shuffle=True)
appendage_dataloader = DataLoader(appendage_dataset, batch_size=16, shuffle=True)

# verify resulted and expected shapes
print(
    f"Raw positions shape: {appendage_dataset.train_data.shape}, expected: (2, T, 63)"
)
print(f"Appendage data shape: {appendage_dataset.app_data.shape}, expected: (T, 12)")
print(f"Region tokens shape: {appendage_dataset.region_tokens.shape}, expected: (T)")

warmup_steps = 4000
base_lr = 5e-5


def lr_lambda(step):
    if step == 0:
        step = 1  # avoid div by zero
    if step < warmup_steps:
        return step / warmup_steps
    else:
        return (warmup_steps**0.5) / (step**0.5)


device = 0
epochs = 500
model = E2EPositionLLM()  # end to end position llm
optimizer = AdamW(model.parameters(), lr=base_lr, betas=[0.9, 0.98], eps=1e-9)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
class_loss_fn = CrossEntropyLoss()
appendage_loss_fn = MSELoss()
lambda_appendage_loss = 1

# start a new wandb run to track this script.
run = wandb.init(
    name="big_hand_100",
    # set the wandb entity where your project will be logged (generally your team name).
    entity="tejasprabhune-uc-berkeley-electrical-engineering-compute",
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


def train():
    model.to(device)
    for i in range(epochs):
        iter_tqdm = tqdm(appendage_dataloader)
        for region_batch, appendage_batch in iter_tqdm:
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

            appendage_loss = appendage_loss_fn(pred_appendage, gt_appendange)

            # combine losses to optimize for all at once
            total_loss = region_loss + lambda_appendage_loss * appendage_loss

            iter_tqdm.set_postfix({"loss": total_loss.item()})
            run.log({"loss": total_loss.item()})

            optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
            total_loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly
            scheduler.step()

    run.finish()
    model.to("cpu")


region_tokens, appendage_values = appendage_dataset[0]
print(appendage_values)
quit()


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


random_out, pred_appendage = inference(model)

fig, ax = plt.subplots(1, 5, figsize=(20, 10))

ax[0].plot(random_out[:100], label="pred")
ax[0].plot(region_tokens[:100], label="region true")
ax[0].legend()

ax[1].plot(pred_appendage[:100, 0], label="pred")
ax[1].plot(appendage_values[:100, 0], label="app 0")
ax[1].legend()

ax[2].plot(pred_appendage[:100, 4], label="pred")
ax[2].plot(appendage_values[:100, 4], label="app 4")
ax[2].legend()

ax[3].plot(pred_appendage[:, 7], label="pred")
ax[3].plot(appendage_values[:, 7], label="app 7")
ax[3].legend()

ax[4].plot(pred_appendage[:, 11], label="pred")
ax[4].plot(appendage_values[:, 11], label="app 13")
ax[4].legend()


plt.show()

train()

trained_out, pred_appendage = inference(model)

fig, ax = plt.subplots(1, 5, figsize=(20, 10))

ax[0].plot(random_out[:100], label="pred")
ax[0].plot(region_tokens[:100], label="region true")
ax[0].legend()

ax[1].plot(pred_appendage[:100, 0], label="pred")
ax[1].plot(appendage_values[:100, 0], label="app 0")
ax[1].legend()

ax[2].plot(pred_appendage[:100, 4], label="pred")
ax[2].plot(appendage_values[:100, 4], label="app 4")
ax[2].legend()

ax[3].plot(pred_appendage[:100, 7], label="pred")
ax[3].plot(appendage_values[:100, 7], label="app 7")
ax[3].legend()

ax[4].plot(pred_appendage[:100, 11], label="pred")
ax[4].plot(appendage_values[:100, 11], label="app 13")
ax[4].legend()

plt.show()
