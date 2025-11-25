"""
Train a Position LLM to generalize to Ojas's hand movements. Implement all
previous techn

TODO:
    - hyperparameter tuning
"""

import torch
import wandb
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from eeg.big_hand.position_llm import E2EPositionLLM, RegionDataset

# import region data set to iterate over batches
region_dataset: RegionDataset = RegionDataset("data/relative_old/", seq_len=100)

region_dataloader = DataLoader(region_dataset, batch_size=16, shuffle=True)

# verify resulted and expected shapes
print(f"Raw positions shape: {region_dataset.original_data.shape}, expected: (T, 63)")
print(f"Delta tokens shape: {region_dataset.delta_tokens.shape}, expected: (T, 63)")
print(f"Region tokens shape: {region_dataset.region_tokens.shape}, expected: (T,)")

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
epochs = 30
model = E2EPositionLLM()  # end to end position llm
optimizer = AdamW(model.parameters(), lr=base_lr, betas=[0.9, 0.98], eps=1e-9)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
loss_fn = CrossEntropyLoss()  # cross entropy loss function

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
        iter_tqdm = tqdm(region_dataloader)
        for region_batch, delta_batch in iter_tqdm:
            in_region_tokens = (
                region_batch[:, :-1].to(torch.int64).to(device)
            )  # (1, T-1,)
            gt_region_tokens = (
                region_batch[:, 1:].to(torch.int64).to(device)
            )  # (1, T-1,)

            gt_delta_tokens = delta_batch[:, 1:].to(device)  # (1, T-1, 63)

            region_logits, delta_logits = model(in_region_tokens)

            # cross entropy loss expects (B, C, *additional_dims)
            region_logits = region_logits.transpose(1, 2)

            region_loss = loss_fn(region_logits, gt_region_tokens)

            total_delta_loss = 0

            for channel in range(delta_logits.shape[2]):
                # get gt tokens for this channel
                gt_delta_tokens_channel = gt_delta_tokens[:, :, channel]

                # index into each channel (channel = joint)
                delta_logits_channel = delta_logits[:, :, channel, :]

                # switch for cross-entropy (see above)
                delta_logits_channel = delta_logits_channel.transpose(1, 2)

                # find loss for each channel
                delta_loss = loss_fn(delta_logits_channel, gt_delta_tokens_channel)

                # add to total
                total_delta_loss += delta_loss

            # combine losses to optimize for all at once
            total_loss = region_loss + (1 / 63) * total_delta_loss

            iter_tqdm.set_postfix({"loss": total_loss.item()})
            run.log({"loss": total_loss.item()})

            optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
            total_loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly
            scheduler.step()

    run.finish()
    model.to("cpu")


region_tokens, delta_tokens = region_dataset[0]


def inference(model: torch.nn.Module) -> torch.Tensor:
    all_region_tokens = region_tokens.unsqueeze(0).to(torch.int64)

    first_region_token = all_region_tokens[:, 0]
    first_delta_token = delta_tokens[0].unsqueeze(0)

    region_tokens_so_far = [first_region_token.item()]
    delta_tokens_so_far = [first_delta_token]

    for _ in range(100):
        region_token_logits, delta_token_logits = model(
            torch.tensor(region_tokens_so_far).unsqueeze(0)
        )

        best_region_token = torch.argmax(region_token_logits.squeeze(0)[-1])

        delta_token_logits = delta_token_logits.squeeze(0)[-1]
        best_delta_token = torch.argmax(delta_token_logits, dim=1)

        region_tokens_so_far.append(best_region_token.item())
        delta_tokens_so_far.append(best_delta_token.unsqueeze(0))

    return region_tokens_so_far, torch.cat(delta_tokens_so_far, dim=0)


random_out, out_delta_tokens = inference(model)
print(out_delta_tokens.shape)

fig, ax = plt.subplots(1, 5, figsize=(20, 10))

ax[0].plot(random_out, label="pred")
ax[0].plot(region_tokens, label="region true")
ax[0].legend()

ax[1].plot(out_delta_tokens[:, 21], label="pred")
ax[1].plot(delta_tokens[:, 21], label="delta 21")
ax[1].legend()

ax[2].plot(out_delta_tokens[:, 13], label="pred")
ax[2].plot(delta_tokens[:, 13], label="delta 13")
ax[2].legend()

ax[3].plot(out_delta_tokens[:, 18], label="pred")
ax[3].plot(delta_tokens[:, 18], label="delta 18")
ax[3].legend()

ax[4].plot(out_delta_tokens[:, 5], label="pred")
ax[4].plot(delta_tokens[:, 5], label="delta 5")
ax[4].legend()


plt.show()

train()

trained_out, out_delta_tokens = inference(model)

fig, ax = plt.subplots(1, 5, figsize=(20, 10))

ax[0].plot(trained_out, label="pred")
ax[0].plot(region_tokens, label="region true")
ax[0].legend()

ax[1].plot(out_delta_tokens[:, 21], label="pred")
ax[1].plot(delta_tokens[:, 21], label="delta 21")
ax[1].legend()

ax[2].plot(out_delta_tokens[:, 13], label="pred")
ax[2].plot(delta_tokens[:, 13], label="delta 13")
ax[2].legend()

ax[3].plot(out_delta_tokens[:, 18], label="pred")
ax[3].plot(delta_tokens[:, 18], label="delta 18")
ax[3].legend()

ax[4].plot(out_delta_tokens[:, 5], label="pred")
ax[4].plot(delta_tokens[:, 5], label="delta 5")
ax[4].legend()

plt.show()
