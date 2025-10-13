# Data

### `collect_data.py`

Collects hand pose estimation data for XYZ position data for 21 joints on a hand, stored in an .npy file. The .npy file will contains T (number of frames) rows and 63 columns (channels) for each joint.

#### Arguments

Run `collect_data.py` with two required arguments: the filename and length of time to run the collection for (seconds). Optional arguments:

1. `boolean --save_video` - save a video of the hand pose estimation.
1. `boolean --plot` - plot a specific joint's data.
1. `joint --joint` - specific joint to plot.

### `train_kmeans.py`

Kmeans is a unsupervised machine learning clustering algorithm for vector quantization. It will partition $n$ observations into k clusters.

We can use this file to train on any type of data. For example, we can train it on raw position values and turn then into region tokens. We can also train KMeans on delta tokens instead.

#### Arguments
1. Save location
1. Data file location
