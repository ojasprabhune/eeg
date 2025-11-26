import argparse

from eeg.big_hand.cluster.kmeans import kmeans

parser = argparse.ArgumentParser()  # arguments in command line
parser.add_argument("save_location", type=str)  # save location
args = parser.parse_args()  # add argument variables to parser

# train kmeans on data and save to location for future use
kmeans(save_location=args.save_location)
