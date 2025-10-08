from eeg.position_llm import DeltaTokenizer, DeltaDataset, PositionLLM

import torch
import matplotlib.pyplot as plt

tokenizer = DeltaTokenizer()

deltas = [-8, -3, 4, 8, 2, 0, 5]

tokens = tokenizer.encode(torch.tensor(deltas))
new_deltas = tokenizer.decode(tokens)

print(deltas)
print(tokens)
print(new_deltas)


def plot_values():
    plt.plot(dataset.original_data)
    plt.show()

    plt.plot(dataset.data)
    plt.show()

    plt.plot(tokenizer.encode(dataset[0]))
    plt.show()


dataset = DeltaDataset(data_file="data/open_fist_front.npy")
# dataset = DeltaDataset(data_file="data/test.npy")

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

# dataset[0] gives shape (64,)
in_tokens = tokenizer.encode(dataset[0])
# adds an empty dimension at the start so the shape looks like (1, 64)
in_tokens = in_tokens.unsqueeze(dim=0)

print(in_tokens.shape)

out = model(in_tokens)
print(out.shape)
