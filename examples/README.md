# Examples

This folder contains various examples on overfitting Position LLMs, region tokenization, and handling data.

### `train_e2e_overfit.py`

A Position LLM trained (overfit) end to end (E2E) using region tokens obtained
from delta tokens. The delta tokens are retrieved from the region tokens through
a Linear layer.

#### Overview:

1. **Raw positions** (T, 63) &larr; Norm + tokenize to deltas
1. **Delta tokens** (T, 63) &larr; Vector quantization (tokenize to regions)
1. **Region tokens** (T,) &larr; Position LLM
1. **Next region probabilities** (T, 50) &larr; Linear layer to delta tokens
    1. 50 is vocab size (num_clusters) of KMeans clustering (probabilities)
1. **Next delta probabilities** (T, 63, 210)
    1. 63 is number of channels
    1. 210 is vocab size of deltas (probabilities)

#### Loss

A new loss was used to calculate gradients for the Position LLM more effectively:

$L = \lambda L_{region}(\hat r_{tokens},\ r_{gt}) + \lambda L_{delta}(\hat d_{tokens},\ d_{gt})$,

where $\lambda$ is the relative importance of a particular loss versus the other losses.

