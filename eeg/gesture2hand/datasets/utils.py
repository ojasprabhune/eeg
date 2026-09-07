import numpy as np
import pywt

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
    # this looks like [0, 1, 2, 3, 4, 5, ... 64]
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

            bp = {}  # name: power

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
                bp[name] = psd[mask, ch].sum()  # 1 number

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


def compute_dwt_features(
    epoch: np.ndarray, wavelet: str = "coif1", level: int = 5
) -> np.ndarray:
    """
    Compute wavelet-energy features from one epoch via the discrete wavelet
    transform (DWT). Used by GestureDataset as an alternative to
    compute_bandpower_features above - same idea (summarize how much signal
    power is in different frequency bands) but computed with wavelets
    instead of an FFT, which also keeps some information about *when* in the
    epoch each frequency band was active (an FFT window only tells you how
    much of a frequency was present, not when). "coif1" (Coiflet, 1
    vanishing moment) is the same wavelet AlQattan & Sepulveda (2017) used
    for EEG-based ASL sign classification.

    Parameters
    ----------
    epoch : np.ndarray, shape (T, 14)
        One raw filtered-channel epoch (e.g. one 3s motor-execution trial).
    wavelet : str
        Wavelet family to decompose with.
    level : int
        Number of decomposition levels.

    Returns
    -------
    features : np.ndarray, shape (14 * (level + 1),)
        Per channel, in order: [energy(cA_level), energy(cD_level), ...,
        energy(cD1)] - one approximation band plus `level` detail bands.
    """
    T, C = epoch.shape

    # one approximation-band energy + one energy per detail level, per channel
    n_bands = level + 1
    features = np.zeros(C * n_bands, dtype=np.float32)

    for ch in range(C):
        # pywt.wavedec repeatedly splits the signal into a smoothed "low
        # frequency" half (the approximation) and a "high frequency" half
        # (the detail), then re-splits the approximation again on the next
        # level - each split roughly halves the frequency range it covers.
        # coeffs comes back ordered coarsest-to-finest:
        # [cA_level, cD_level, cD_level-1, ..., cD1]
        coeffs = pywt.wavedec(epoch[:, ch], wavelet, level=level)

        for band_idx, band_coeffs in enumerate(coeffs):
            # energy = sum of squared coefficients: one number summarizing
            # how much signal power lives in that band, the same idea as the
            # bandpower sum in compute_bandpower_features above but derived
            # from wavelet coefficients instead of an FFT power spectrum
            energy = np.sum(band_coeffs**2)
            features[ch * n_bands + band_idx] = energy

    return features  # (14 * (level + 1),)
