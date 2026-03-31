import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

import matplotlib.pyplot as plt

from eeg.region_token.position_llm import (
    RegionTokenizer,
    DeltaTokenizer,
    E2EPositionLLM,
)
from eeg.region_token.data_collection.utils import normalize

region_tokenizer = RegionTokenizer("models/delta_tokens")
delta_tokenizer = DeltaTokenizer()  # delta tokenizer
scaler = region_tokenizer.scaler

data = np.load("data/open_fist_front.npy")  # load file
data = np.array(scaler.transform(data))  # standardize before processing
deltas = np.diff(data, axis=0)  # difference between time steps
deltas = normalize(deltas, deltas.max(), deltas.min(), 10, -10)  # crunch
deltas = np.round(deltas, decimals=1)  # round to tenths

delta_tokens = delta_tokenizer.encode(deltas)[:100]  # deltas -> tokens
region_tokens = region_tokenizer.encode(delta_tokens)  # (100,)

# raw pos: expected (T, 63)
# DTokens: expected (T, 63)
# RTokens: expected (T,)

print(f"Raw pos data shape: {data.shape}, Expected: (T, 63)")
print(f"Delta tokens shape: {delta_tokens.shape}, Expected: (T, 63)")
print(f"Region tokens shape: {region_tokens.shape}, Expected: (T,)")

model = E2EPositionLLM()

optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
loss_fn = CrossEntropyLoss()


def train():
    iter_tqdm = tqdm(range(2000))
    for _ in iter_tqdm:
        # mac torch expects Long, which is the same as int64, so we convert
        # positional encoding needs a batch size
        all_delta_tokens = delta_tokens.unsqueeze(0).to(torch.int64)
        all_region_tokens = region_tokens.unsqueeze(0).to(torch.int64)

        in_region_tokens = all_region_tokens[:, :-1]  # (1, T-1,)
        gt_region_tokens = all_region_tokens[:, 1:]  # (1, T-1,)

        gt_delta_tokens = all_delta_tokens[:, 1:]  # (1, T-1, 63)

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
        total_loss = 63 * region_loss + total_delta_loss

        iter_tqdm.set_postfix({"loss": total_loss.item()})

        optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
        total_loss.backward()  # calculates and adds gradients to params so optim sees
        optimizer.step()  # optim looks at gradients and steps accordingly


def inference(model: torch.nn.Module) -> tuple[list, torch.Tensor]:
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

ax[0].plot(random_out)
ax[0].plot(region_tokens)

ax[1].plot(out_delta_tokens[:, 21])
ax[1].plot(delta_tokens[:, 21])

ax[2].plot(out_delta_tokens[:, 13])
ax[2].plot(delta_tokens[:, 13])

ax[3].plot(out_delta_tokens[:, 18])
ax[3].plot(delta_tokens[:, 18])

ax[4].plot(out_delta_tokens[:, 5])
ax[4].plot(delta_tokens[:, 5])

plt.show()

train()

trained_out, out_delta_tokens = inference(model)

fig, ax = plt.subplots(1, 5, figsize=(20, 10))

ax[0].plot(trained_out)
ax[0].plot(region_tokens)

ax[1].plot(out_delta_tokens[:, 21])
ax[1].plot(delta_tokens[:, 21])

ax[2].plot(out_delta_tokens[:, 13])
ax[2].plot(delta_tokens[:, 13])

ax[3].plot(out_delta_tokens[:, 18])
ax[3].plot(delta_tokens[:, 18])

ax[4].plot(out_delta_tokens[:, 5])
ax[4].plot(delta_tokens[:, 5])
plt.show()
