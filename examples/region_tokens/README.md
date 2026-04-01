# Region tokenization examples

Region tokens are obtained through vector quantization of the 63 channels in a data file. These tokens allow the Position LLM to capture the high-dimensional movement of all 21 hand joints simultaneously.

### `test_region_tokenizer.py`
Loads a trained `RegionTokenizer` (K-Means), encodes raw positional data into region tokens, and verifies the process by decoding them back into the region centers.

### `train_overfit_regions.py`
Trains a `PositionLLM` to overfit on region tokens derived directly from raw positional data.

### `train_overfit_regions_deltas.py`
Trains a `PositionLLM` to overfit on region tokens derived from normalized delta tokens. This approach focuses on modeling movement dynamics rather than absolute positions.
