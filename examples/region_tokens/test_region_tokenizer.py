from eeg.region_token.position_llm.tokenizer import RegionTokenizer

import torch
import numpy as np

tokenizer = RegionTokenizer("models/region_token")
print(f"Number of regions: {len(tokenizer.region_centers)}")

data = np.load("data/open_fist_front.npy")

region_tokens = tokenizer.encode(torch.tensor(data))
print(region_tokens.shape)

decoded = tokenizer.decode(region_tokens)
print(decoded.shape)
