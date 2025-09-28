from eeg.region_token.cluster.kmeans import kmeans

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("save_location", type=str)

args = parser.parse_args()

kmeans(save_location=args.save_location)
