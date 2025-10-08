# Collecting data

`collect_data.py` will collect hand pose estimation data for XYZ position data for 21 joints on a hand, stored in an .npy file. The .npy file will contain T (number of frames) rows and 63 columns (channels) for each joint.

## Arguments

Run `collect_data.py` with two required arguments: the filename and length of time to run the collection for (seconds). Optional arguments:

1. `boolean --save_video` - save a video of the hand pose estimation.
1. `boolean --plot` - plot a specific joint's data.
1. `joint --joint` - specific joint to plot.

# Training
