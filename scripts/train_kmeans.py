"""
Kmeans is a unsupervised machine learning clustering
algorithm for vector quantization. It will partition
n observations into k clusters.

This file has 3 arguments:
    1. Save location
    2. Data file location
    3. Difference? (find deltas)

We can use this file to train on any type of data.

For example, we can train it on raw position values
and turn then into region tokens. We can also train
KMeans on delta tokens instead.
"""

import argparse

from eeg.region_token.cluster.kmeans import kmeans

parser = argparse.ArgumentParser()  # arguments in command line
parser.add_argument("save_location", type=str)  # save location
parser.add_argument("data_file", type=str)  # data file location
parser.add_argument("difference", type=bool)  # data file location
args = parser.parse_args()  # add argument variables to parser

# train kmeans on data and save to location for future use
kmeans(save_location=args.save_location, data_file=args.data_file, diff=args.difference)
