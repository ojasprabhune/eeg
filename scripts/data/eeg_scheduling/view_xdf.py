import matplotlib.pyplot as plt
import pyxdf

streams, header = pyxdf.load_xdf(
    "/Users/ojasprabhune/Documents/CurrentStudy/sub-P001/"
    "ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
)

# find marker stream
marker_stream = None
for stream in streams:
    if stream["info"]["type"][0] == "Markers":
        marker_stream = stream
        break

if marker_stream is None:
    raise ValueError("No marker stream found")

# extract data
markers = [x[0] for x in marker_stream["time_series"]]
timestamps = marker_stream["time_stamps"]

# convert to time relative to start
relative_time = timestamps - timestamps[0]

plt.figure(figsize=(14, 5))

# use eventplot for clean event markers
plt.eventplot(relative_time, lineoffsets=0, linelengths=0.4, colors="tab:blue")

# alternate label Y-positions to avoid horizontal overlap
y_positions = [0.3, 0.5, 0.7, 0.9]  # cycle through different heights

for i, (t, label) in enumerate(zip(relative_time, markers)):
    y_pos = y_positions[i % len(y_positions)]

    # draw a thin stem line linking marker to text
    plt.plot([t, t], [0.2, y_pos - 0.05], color="gray", linestyle=":", alpha=0.6)

    # annotate text
    plt.text(
        t,
        y_pos,
        label,
        rotation=45,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"
        ),
    )

# formatting
plt.ylim(-0.3, 1.4)  # leave headroom for labels and title
plt.yticks([])
plt.xlabel("Time (seconds)")
plt.title("LSL Marker Timeline", pad=20)
plt.grid(True, axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.show()
