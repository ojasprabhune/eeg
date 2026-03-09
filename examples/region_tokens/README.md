# Training for region tokens
<<<<<<< HEAD
=======

Region tokens are obtained through vector quantization of the 63 channels in a data file. Each channel are X, Y, or Z positions for 21 joints on a hand. Region tokens allow the Position LLM to train on the movement (or deltas) of all joints on all axes (i.e., not just the X movement of the index finger tip).

### `test_region_tokenizer.py`

Loads region tokenizer (KMeans trained with 50 regions), encodes raw positional data into region tokens, and then decodes region tokens back into raw positional data.

### `train_overfit_regions.py`

Trains and overfits a Position LLM on all 63 channels (joints + XYZ) of raw positional data using regions for 64 time steps.

### `train_overfit_regions_deltas.py`

Trains and overfits a Position LLM on all 63 channels (joints + XYZ) of the *delta __tokens__* of raw positional data for 64 time steps. Uses a KMeans clustering model trained on deltas tokens.
>>>>>>> eeg_data
