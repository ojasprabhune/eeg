import matplotlib
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from eeg.region_token.position_llm import PositionLLM, RegionTokenizer, DeltaTokenizer
from eeg.region_token.data_collection.utils import normalize

matplotlib.use("module://matplotlib-backend-wezterm")  # needed for WSL matplib display
import matplotlib.pyplot as plt

# use trained KMeans model on delta tokens
region_tokenizer = RegionTokenizer("models/delta_tokens")
delta_tokenizer = DeltaTokenizer()  # delta tokenizer
scaler = region_tokenizer.scaler

data = np.load("data/open_fist_front.npy")  # load file
data = scaler.transform(data)  # standardize before processing
deltas = np.diff(data, axis=0)  # difference between time steps
deltas = normalize(deltas, deltas.max(), deltas.min(), 20, -20)  # crunch
deltas = np.round(deltas, decimals=1)  # round to tenths

delta_tokens = delta_tokenizer.encode(deltas)  # deltas -> tokens
print("deltatokens", delta_tokens[0])
region_tokens = region_tokenizer.encode(delta_tokens)[:64]  # (1472,)
print("regionstokens", region_tokens)
print("region_tokens:", region_tokens.shape)

model = PositionLLM(
    # vocab size is amount of regions
    vocab_size=len(region_tokenizer.region_centers),
    num_layers=1,
    num_heads=1,
    embedding_dim=64,
    ffn_hidden_dim=64,
    qk_length=64,
    value_length=64,
    max_length=2048,
    dropout=0.1,
)

optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
loss_fn = CrossEntropyLoss()

# adds an empty dimension at the start so the shape looks like (1, 1472)
in_tokens = region_tokens.unsqueeze(dim=0).to(
    dtype=torch.int64
)  # why ????????????????????????????????
print("region_tokens unsqueezed:", in_tokens.shape)


def train():
    epoch_tqdm = tqdm(range(5000))
    for epoch in epoch_tqdm:
        # up to second-last so we have a loss for our second-last token
        in_ = in_tokens[:, :-1]  # all tokens except last one
        out = model(in_)  # gives prob distributions for second token to the end

        # the second token onwards
        gt = in_tokens.squeeze()[1:]  # all tokens except first one

        # cross entropy loss makes one-hot encoding for gt automatically
        loss = loss_fn(out.squeeze(), gt)
        epoch_tqdm.set_postfix({"loss": loss.item()})  # prints loss

        optimizer.zero_grad()

        # calculates gradients for all parameters
        loss.backward()  # out comes from the model, so loss has access to all params

        # step all parameters using the gradients and the learning rate
        optimizer.step()


def inference(model: torch.nn.Module) -> torch.Tensor:
    first_token = in_tokens[:, 0]
    tokens_so_far = [first_token.item()]

    for i in range(64):
        input_tokens = torch.tensor(tokens_so_far)
        input_tokens = input_tokens.unsqueeze(dim=0)

        next_token_distribution = model(input_tokens)
        next_token = torch.argmax(next_token_distribution, dim=-1)

        # goes to first row and gets last item
        tokens_so_far.append(next_token[0][-1].item())

    return torch.tensor(tokens_so_far)


random_out = inference(model)

plt.plot(random_out)
plt.plot(in_tokens[0])
plt.show()

train()

trained_out = inference(model)

plt.plot(trained_out)
plt.plot(in_tokens[0])
plt.show()
