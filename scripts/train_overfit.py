from eeg.position_llm import DeltaTokenizer, DeltaDataset, PositionLLM

from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import matplotlib.pyplot as plt

tokenizer = DeltaTokenizer()
dataset = DeltaDataset(data_file="data/open_fist_front.npy")
model = PositionLLM(
    vocab_size=len(tokenizer.mapping),
    num_layers=1,
    num_heads=1,
    embedding_dim=64,
    ffn_hidden_dim=64,
    qk_length=64,
    value_length=64,
    max_length=1024,
    dropout=0.1,
)
optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
loss_fn = CrossEntropyLoss()

# dataset[0] gives shape (64,)
in_tokens = tokenizer.encode(dataset[0])
# adds an empty dimension at the start so the shape looks like (1, 64)
in_tokens = in_tokens.unsqueeze(dim=0)


def train():
    epoch_tqdm = tqdm(range(5000))
    for epoch in epoch_tqdm:
        # up to second-last so we have a loss for our second-last token
        out = model(in_tokens[:, :-1])
        # print(out.shape)

        # the second token onwards
        loss = loss_fn(out.squeeze(), in_tokens.squeeze()[1:])
        epoch_tqdm.set_postfix({"loss": loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def inference(model: torch.nn.Module) -> torch.Tensor:
    first_token = in_tokens[:, 0]
    # print(first_token)

    tokens_so_far = [first_token.item()]

    for i in range(64):
        input_tokens = torch.tensor(tokens_so_far)
        input_tokens = input_tokens.unsqueeze(0)
        # print(f"input: {input_tokens.shape}")

        next_token_distribution = model(input_tokens)

        next_token = torch.argmax(next_token_distribution, dim=-1)

        # print(f"next_token: {next_token.shape}")

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
