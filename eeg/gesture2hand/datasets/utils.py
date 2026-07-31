import numpy as np
import os

from eeg.data_collection import JointData, Joint, DataType

EMOTIV_CHANNELS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def appendages(joint_data: JointData) -> np.ndarray:
    """
    Uses joint data to calculate appendage vectors and
    unit vectors to perform change of basis and return
    an numpy array of size (T, 12).
    """
    origin = joint_data.get_positions(DataType.NORM, Joint.W)
    mid_mcp = joint_data.get_positions(DataType.NORM, Joint.MM)
    pinky_mcp = joint_data.get_positions(DataType.NORM, Joint.PM)

    # unit vectors shapes: (T, 3)

    unit_z = mid_mcp - origin  # vector from wrist to mid_mcp
    # divide by magnitude (becomes unit vector)
    unit_z /= np.linalg.norm(unit_z, axis=1, keepdims=True)

    vec_y = pinky_mcp - origin
    unit_x = np.cross(vec_y, unit_z)  # perpendicular
    unit_x /= np.linalg.norm(unit_x, axis=1, keepdims=True)

    unit_y = np.cross(unit_z, unit_x)

    # rotational matrix for linear transformation with time steps
    # (3, T, 3)
    R: np.ndarray = np.array([-unit_x, -unit_y, unit_z])

    R = R.transpose(1, 0, 2)  # (3, T, 3) -> (T, 3, 3)

    def change_of_basis(tip_idx: Joint, mcp_idx: Joint) -> np.ndarray:
        v = joint_data.get_positions(
            DataType.WORLD, tip_idx
        ) - joint_data.get_positions(DataType.WORLD, mcp_idx)  # (T, 3)

        return np.matmul(R, v[:, :, None]).squeeze(-1)

    # appendage vectors should be (T, 3)

    index = change_of_basis(Joint.IT, Joint.IM)
    middle = change_of_basis(Joint.MT, Joint.MM)
    ring = change_of_basis(Joint.RT, Joint.RM)
    thumb = change_of_basis(Joint.TT, Joint.TM)

    # concats 4 fingers' 3 vector components horizontally, retaining time
    result = np.concatenate([index, middle, ring, thumb], axis=1)  # (T, 12)

    return result


def process_deltas(data: np.ndarray) -> np.ndarray:
    deltas = np.diff(data, axis=0)  # deltas
    norm_deltas = normalize(deltas, deltas.max(), deltas.min(), 10, -10)
    round_data = norm_deltas.round(decimals=1)
    return round_data


def normalize(
    value,
    old_max: float,
    old_min: float,
    new_max: float,
    new_min: float,
):
    """
    Converts a number range to another range while maintaining ratio.
    """
    old_range = old_max - old_min
    new_range = new_max - new_min
    new_value = (((value - old_min) * new_range) / old_range) + new_min

    return new_value


def compute_bandpower_features(
    eeg_128hz: np.ndarray,
    sfreq: float = 128.0,
    window_sec: float = 1.0,
    step_samples_128: int = 4,
) -> np.ndarray:
    """
    Compute bandpower features from 128 Hz EEG via FFT.

    Parameters
    ----------
    eeg_128hz : np.ndarray, shape (T_128, 14)
        Filtered EEG at native sample rate.
    sfreq : float
        Sampling frequency.
    window_sec : float
        FFT window length in seconds.
    step_samples_128 : int
        Step size in samples. 4 @ 128 Hz ≈ 32 Hz output.

    Returns
    -------
    features : np.ndarray, shape (T_out, 84)
        14 channels × 6 features (theta, mu, beta, low_gamma, mu/beta, total).

    """
    T, C = eeg_128hz.shape
    nperseg = int(window_sec * sfreq)  # number of samples in a window
    half_win = nperseg // 2  # number of samples in half a window

    # frequency bands of interest
    bands = {
        "theta": (4, 8),
        "mu": (8, 13),
        "beta": (13, 30),
        "low_gamma": (30, 50),
    }

    # pre-compute frequency masks for FFT bins of shape (nperseg//2 + 1,)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sfreq)

    # dictionary of boolean masks that says True for those frequencies that fall
    # into a band. the mask for each band has shape (nperseg//2 + 1,) and can be
    # applied to the FFT output
    band_masks = {
        name: np.logical_and(freqs >= flo, freqs <= fhi)
        for name, (flo, fhi) in bands.items()
    }

    # fades edges to zero to reduce spectral leakage. it contains the values of
    # a Hanning window of length nperseg, which is a smooth curve that starts
    # and ends at zero and peaks at 1 in the middle. by multiplying each
    # windowed segment of EEG data by this Hanning window, we ensure that the
    # edges of the segment are weighted less in the FFT, which helps to minimize
    # artifacts in the frequency domain caused by abrupt changes at the segment
    # boundaries
    hann = np.hanning(nperseg)[:, None]  # (nperseg, 1)

    # np.arange goes from number of samples in half a window to T minus that
    # number, stepping by a step size. the physical meaning of this is that we
    # are centering a window around each point in time where we have enough
    # samples on either side to fill the window, and we are moving this center
    # point by a certain step size to get the next window. the output will be a
    # sequence of bandpower features that are aligned with the original EEG time
    # series, but at a lower temporal resolution
    centers = np.arange(half_win, T - half_win, step_samples_128)

    # number of output time points after windowing
    n_out = len(centers)

    # np.zeros to create an array to hold the bandpower features, with shape
    # (n_out, C * 6) where C is the number of channels and 6 is the number of
    # features per channel
    features = np.zeros((n_out, C * 6), dtype=np.float32)  # (T_out, 84)

    # index and actual time of the center of each window
    for i, t in enumerate(centers):
        # extract a segment of EEG data centered around time t with length equal
        # to the window size. this segment will be used to compute the FFT and
        # bandpower features for that time point. by multiplying the segment by
        # the Hanning window, we are applying a smooth taper to the data, which
        # helps to reduce spectral leakage in the FFT. the resulting segment has
        # shape (nperseg, C) where nperseg is the number of samples in the
        # window and C is the number of channels
        segment = eeg_128hz[t - half_win : t + half_win, :] * hann  # (nperseg, C)

        # compute the FFT of the windowed segment along the time axis (axis=0).
        fft_vals = np.fft.rfft(segment, axis=0)  # (nperseg//2 + 1, C)

        # compute the power spectral density (PSD) from the FFT values. the PSD
        # is a measure of the power of the signal at different frequencies, and
        # it is computed by taking the squared magnitude of the FFT values and
        # normalizing by the number of samples in the window. the resulting PSD
        # has shape (nperseg//2 + 1, C) and contains the power of the signal at
        # each frequency bin for each channel
        psd = (np.abs(fft_vals) ** 2) / nperseg

        for ch in range(C):
            # start position for this channel's features in the output array
            base = ch * 6

            bp = {}

            # j is the index, and (name, mask) is the tuple of band name and its
            # corresponding frequency mask.
            for j, (name, mask) in enumerate(band_masks.items()):
                # psd has shape (nperseg//2 + 1, C) or (num_freq_bins, C). mask
                # selects only frequences inside a band (e.g., 8-13 Hz for mu).
                # psd[mask, ch] -> power values for that band for this channel.
                # .sum() -> total power in that frequency band. this is stored
                # in bp[name] (e.g., bp["mu"] = bandpower). bandpower is type
                # float and is just a single number representing the total power
                # in that frequency band
                bp[name] = psd[mask, ch].sum()

                # i is time window index, and base + j is which band (0=theta,
                # 1=mu, etc.) this stores the computed bandpower into the
                # output feature vector, effectively building: [theta, mu, beta,
                # low_gamma, ...] per channel]
                features[i, base + j] = bp[name]

            # compute mu-to-beta ratio, which is a common EEG feature for motor
            # activity and engagement. 1e-10 prevents division by zero, and
            # this is stored as the 5th feature for this channel
            features[i, base + 4] = bp["mu"] / (bp["beta"] + 1e-10)

            # sum all bandpowers -> total signal power across all bands. it
            # acts as a normalization reference or overall energy measure
            features[i, base + 5] = sum(bp.values()) + 1e-10

    return features  # (T, 84)


def min_max_npy(directory_path):
    """
    Finds the overall minimum and maximum values across all .npy files in a given directory.

    Args:
        directory_path (str): The path to the directory containing the .npy files.

    Returns:
        tuple: A tuple containing (overall_min, overall_max).
    """
    overall_min = 0
    overall_max = 0
    found_npy_files = False

    for filename in os.listdir(directory_path):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory_path, filename)
            try:
                data = np.load(filepath)

                # Initialize overall_min and overall_max with the first file's min/max
                if not found_npy_files:
                    overall_min = np.min(data)
                    overall_max = np.max(data)
                    found_npy_files = True
                else:
                    overall_min = min(overall_min, np.min(data))
                    overall_max = max(overall_max, np.max(data))

            except Exception as e:
                print(f"Error loading or processing {filename}: {e}")

    return overall_min, overall_max


if __name__ == "__main__":
    # example usage:
    # Replace with the actual path to your directory
    directory = "/home/prabhune/projects/research/2026/eeg/data/"
    min_val, max_val = min_max_npy(directory)

    print(f"Minimum value: {min_val}")
    print(f"Maximum value: {max_val}")

    min_value = normalize(min_val, max_val, min_val, 1, -1)
    max_value = normalize(max_val, max_val, min_val, 1, -1)
    middle = normalize(-48.5, max_val, min_val, 1, -1)

    print(f"Normalized Minimum value: {min_value}")
    print(f"Normalized middle value: {middle}")
    print(f"Normalized Maximum value: {max_value}")
