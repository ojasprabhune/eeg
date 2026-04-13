from pathlib import Path

import mne
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from eeg.big_hand.position_llm import RegionTokenizer
from eeg.big_hand.position_llm.vqvae import VQVAE
from eeg.data_collection import JointData

from .utils import Colors, appendages


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
    print(f"Number of frequencies: {freqs.shape}")

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


class TemporalDataset(Dataset):
    def __init__(
        self,
        eeg_data_path: str = "/var/log/thavamount/eeg_dataset/home_eeg",
        hand_data_path: str = "/var/log/thavamount/eeg_dataset/hand_data",
        vqvae_path: str = "/var/log/thavamount/eeg_ckpts/eeg_vqvae/vqvae_final_1250.pth",
        region_tokenizer_path: str = "models/appendages",
        seq_len: int = 60,
        stride: int = 15,
        device: str = "cpu",
        mode: str = "train",
        val_ratio: float = 0.2,
        verbose: bool = False,
    ) -> None:
        """
        Dataset for regressing EEG data to hand movements. Loads and
        preprocesses the data.
        """

        print(f"{Colors.HEADER}{Colors.BOLD}Initializing dataset...{Colors.ENDC}")
        self.verbose = verbose
        self.seq_len = seq_len
        self.stride = stride
        self.device = device

        super().__init__()

        # --- vqvae -----------------------------------------------------------

        print(f"{Colors.OKBLUE}Getting VQVAE model...{Colors.ENDC}")
        self.vqvae = VQVAE(input_dim=12, codebook_size=512, embedding_dim=1024)
        vqvae_state_dict = torch.load(vqvae_path, map_location=device)
        self.vqvae.load_state_dict(vqvae_state_dict["model"])
        self.vqvae.to(device)
        self.vqvae.eval()

        self.region_tokenizer = RegionTokenizer(region_tokenizer_path)

        # --- EEG -------------------------------------------------------------

        print(f"{Colors.OKBLUE}Getting EEG data...{Colors.ENDC}")
        self.raws = []
        for path in sorted(Path(f"{eeg_data_path}").rglob("*_cut_raw.fif")):
            raw = mne.io.read_raw_fif(path, verbose=False)
            # some edf+ files have 4 fewer channels
            if raw.info["nchan"] == 71:
                extra_channels = [
                    "OrTimestampS",
                    "OrTimestampMs",
                    "MOT.OrTimestampS",
                    "MOT.OrTimestampM",
                ]
                raw.drop_channels(extra_channels)
            self.raws.append(raw)
            # break  # TODO remove this to load all data

        # - filtering -
        self.raw = mne.concatenate_raws(self.raws, preload=True, verbose=False)
        if isinstance(self.raw, tuple):
            self.raw = self.raw[0]
        self.filtered = self.raw.copy().filter(l_freq=0.1, h_freq=50, verbose=False)
        self.filtered = self.filtered.notch_filter(freqs=60, verbose=False)
        self.filtered.set_eeg_reference(
            "average", projection=False, verbose=False
        )  # common average reference

        # - pick channels before branching -
        self.eeg_channels = [
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
        self.filtered.pick(self.eeg_channels, verbose=False)

        # --- 128 Hz branch: bandpower features -------------------------------
        # keep a copy at native rate BEFORE resampling

        self.sfreq_native = self.filtered.info["sfreq"]  # should be 128
        eeg_128hz = self.filtered.get_data().T  # (T_128, 14)

        print(
            f"{Colors.OKBLUE}Computing bandpower features at {self.sfreq_native} Hz...{Colors.ENDC}"
        )
        self.bandpower_features_raw = compute_bandpower_features(
            eeg_128hz,
            sfreq=self.sfreq_native,
            window_sec=1.0,
            step_samples_128=4,  # ≈ 32 Hz output
        )
        # Z-score bandpower features
        bp_mean = self.bandpower_features_raw.mean(axis=0, keepdims=True)
        bp_std = self.bandpower_features_raw.std(axis=0, keepdims=True) + 1e-8
        self.bandpower_features = (
            (self.bandpower_features_raw - bp_mean) / bp_std
        ).astype(np.float32)

        print(
            f"{Colors.OKGREEN}Bandpower features shape: {self.bandpower_features.shape} or (T, 84){Colors.ENDC}"
        )

        # --- 30 Hz branch: raw EEG filtering ---------------------------------

        filtered_30 = self.filtered.copy()
        filtered_30.filter(
            8, 30, method="iir", iir_params=dict(order=4, ftype="butter"), verbose=False
        )
        filtered_30.resample(sfreq=29.973234, verbose=False)

        self.eeg_data: np.ndarray = filtered_30.get_data().T  # (T_30, 14)

        print(f"{Colors.OKGREEN}Filtered & processed EEG data.{Colors.ENDC}")

        # --- labels ----------------------------------------------------------

        print(f"{Colors.OKBLUE}Getting labels...{Colors.ENDC}")
        self.label_files = []
        for path in sorted(Path(f"{hand_data_path}").rglob("*labels_cut.npy")):
            self.label_files.append(np.load(path))

        self.labels = np.concatenate(self.label_files, axis=0) - 1

        # --- appendages + regions --------------------------------------------

        print(f"{Colors.OKBLUE}Getting appendage data...{Colors.ENDC}")
        self.hands = []
        for path in sorted(Path(f"{hand_data_path}").rglob("*hands_cut.npy")):
            self.hands.append(np.load(path))
            # break  # TODO remove this to load all data

        self.raw_app_data = np.concatenate(self.hands, axis=1)  # along time dim
        self.data_joints = JointData(self.raw_app_data)
        self.app_data = appendages(self.data_joints)  # (T, 12)
        self.app_data = np.array(self.region_tokenizer.scaler.transform(self.app_data))

        print(f"{Colors.OKGREEN}Retrieved appendage data.{Colors.ENDC}")
        print(f"{Colors.OKGREEN}Successful retrieved all data.{Colors.ENDC}")

        # --- align bandpower features with hand data -------------------------

        min_len = min(
            len(self.bandpower_features), len(self.app_data), len(self.eeg_data)
        )
        self.bandpower_features = self.bandpower_features[:min_len]
        self.app_data = self.app_data[:min_len]
        self.eeg_data = self.eeg_data[:min_len]
        self.labels = self.labels[:min_len]

        # - vq-vae pre-computing -
        print(f"{Colors.OKBLUE}Pre-computing VQ-VAE tokens...{Colors.ENDC}")
        self.vqvae_tokens_all = []
        chunk_size = 2048
        with torch.no_grad():
            for i in tqdm(range(0, len(self.app_data), chunk_size)):
                chunk = self.app_data[i : i + chunk_size, :]
                chunk_tensor = (
                    torch.tensor(chunk, dtype=torch.float32).to(device).unsqueeze(0)
                )
                tokens = self.vqvae.encode(chunk_tensor)
                self.vqvae_tokens_all.append(tokens.cpu().numpy().flatten())
        self.vqvae_tokens_all = np.concatenate(self.vqvae_tokens_all)

        if verbose:
            print(Colors.HEADER)
            print("EEG shape (30Hz):   ", self.eeg_data.shape)
            print("Bandpower shape:    ", self.bandpower_features.shape)
            print("Appendages shape:   ", self.app_data.shape)
            print("VQ-VAE tokens shape:", self.vqvae_tokens_all.shape)
            print("Labels shape:       ", self.labels.shape)
            print(Colors.ENDC)

        # --- sequences -------------------------------------------------------

        self.eeg_chunks = []
        self.app_chunks = []
        self.token_chunks = []
        self.bp_chunks = []
        self.label_chunks = []

        # we iterate over the data in steps of stride, creating chunks of length
        # seq_len. for each chunk, we extract the corresponding segments of
        # eeg_data, bandpower_features, app_data, vqvae_tokens_all, and labels,
        # and store them in lists. this way we create overlapping sequences of
        # data that can be used for training a temporal model. the resulting
        # chunks will have shape (num_chunks, seq_len, feature_dim) for eeg,
        # bandpower, and appendage data, and (num_chunks, seq_len) for tokens
        # and labels
        for i in range(0, min_len - seq_len + 1, stride):
            eeg_chunk = self.eeg_data[i : i + seq_len, :]  # (seq_len, 14)
            bp_chunk = self.bandpower_features[i : i + seq_len, :]  # (seq_len, 84)
            app_chunk = self.app_data[i : i + seq_len, :]  # (seq_len, 12)
            token_chunk = self.vqvae_tokens_all[i : i + seq_len]  # (seq_len,)
            label_chunk = self.labels[i : i + seq_len]

            self.eeg_chunks.append(eeg_chunk)
            self.bp_chunks.append(bp_chunk)
            self.app_chunks.append(app_chunk)
            self.token_chunks.append(token_chunk)
            self.label_chunks.append(label_chunk)

        self.eeg_chunks = np.array(self.eeg_chunks, dtype=np.float32)
        self.bp_chunks = np.array(self.bp_chunks, dtype=np.float32)
        self.app_chunks = np.array(self.app_chunks, dtype=np.float32)
        self.token_chunks = np.array(self.token_chunks, dtype=np.int64)
        self.label_chunks = np.array(self.label_chunks, dtype=np.int64)

        # --- train-val split -------------------------------------------------

        # total number of chunks we have created from the data
        n_chunks = len(self.bp_chunks)

        # determine how many smaller chunks fit into one sequence block.
        # seq_len // stride ≈ how many steps per sequence, and max(1, ...)
        # ensure at least 1 chunk per block
        chunks_per_block = max(1, self.seq_len // self.stride)

        # total number of bandpower chunks available. each chunk = one
        # timestep (or small window)
        n_blocks = n_chunks // chunks_per_block

        if verbose:
            print(Colors.HEADER)
            print("EEG chunks (30Hz):   ", self.eeg_data.shape)
            print("Bandpower chunks:    ", self.bandpower_features.shape)
            print("Appendages chunks:   ", self.app_data.shape)
            print("VQ-VAE tokens chunks:", self.vqvae_tokens_all.shape)
            print("Labels chunks:       ", self.labels.shape)
            print(Colors.ENDC)

        # labels in each block
        block_labels = np.array(
            # select all labels inside block b and compute the most common label
            # value in that block. this is a more general approach that works
            # for multi-class labels, where we take the mode (most frequent
            # label) instead of the mean. the argmax of the bincount gives us
            # the most common label in that block, and we convert it to an
            # integer. this way we can assign a single label to each block based
            # on the majority class
            [
                int(
                    np.bincount(
                        self.label_chunks[
                            b * chunks_per_block : (b + 1) * chunks_per_block
                        ].flatten()
                    ).argmax()
                )
                for b in range(n_blocks)
            ]
        )  # (n_blocks,)

        print("hello", block_labels.shape)

        rng = np.random.RandomState(42)

        # block_labels is shape (n_blocks,). block_labels == 1 -> boolean array
        # of same shape. np.where gives us the indices of the blocks that belong
        # to each class. [0] is to get the indices from the tuple output of
        # np.where
        fist_block_ids = np.where(block_labels == 0)[0]
        left_block_ids = np.where(block_labels == 1)[0]
        finger_block_ids = np.where(block_labels == 2)[0]
        open_block_ids = np.where(block_labels == 3)[0]

        print("fist ids", fist_block_ids.shape)
        print("left ids", left_block_ids.shape)
        print("finger ids", finger_block_ids.shape)
        print("open ids", open_block_ids.shape)

        # randomly shuffle each class separately. preserve class balance before
        # splitting
        rng.shuffle(fist_block_ids)
        rng.shuffle(left_block_ids)
        rng.shuffle(finger_block_ids)
        rng.shuffle(open_block_ids)

        # determine how many blocks to allocate to the validation set for each
        n_val_fist = max(1, int(len(open_block_ids) * val_ratio))
        n_val_left = max(1, int(len(left_block_ids) * val_ratio))
        n_val_finger = max(1, int(len(finger_block_ids) * val_ratio))
        n_val_open = max(1, int(len(open_block_ids) * val_ratio))

        # concatenate the validation blocks from each class to form the complete
        # validation set, and the remaining blocks form the training set. this
        # way we ensure that the validation set has a representative sample of
        # each class, and we can evaluate our model's performance on a balanced
        # subset of the data
        val_block_ids = np.concatenate(
            [
                fist_block_ids[:n_val_fist],
                left_block_ids[:n_val_left],
                finger_block_ids[:n_val_finger],
                open_block_ids[:n_val_open],
            ]
        )
        train_block_ids = np.concatenate(
            [
                fist_block_ids[n_val_fist:],
                left_block_ids[n_val_left:],
                finger_block_ids[n_val_finger:],
                open_block_ids[n_val_open:],
            ]
        )

        # filter data based on mode
        if mode == "train":
            target_block_ids = train_block_ids
        else:
            target_block_ids = val_block_ids

        # function to convert block indices to chunk indices. for each block
        # index, we calculate the start and end chunk indices that
        # correspond to that block, and we extend the list of chunk indices with
        # the range of chunk indices for that block. this way we can easily
        # select the chunks that belong to the training and validation sets
        # based on the block indices we determined earlier
        def blocks_to_chunks(block_ids: np.ndarray) -> np.ndarray:
            chunk_ids = []
            for b in block_ids:
                start = b * chunks_per_block  # first chunk index for block b

                # last chunk index (exclusive). min prevents overflow at
                # dataset end
                end = min(start + chunks_per_block, n_chunks)

                # add all chunk indicies in this block
                chunk_ids.extend(range(start, end))
            return np.array(chunk_ids)  # shape (num_chunks_in_split,)

        print(f"{Colors.OKBLUE}Splitting data into train and val sets...{Colors.ENDC}")

        split_idx = blocks_to_chunks(target_block_ids)

        # TODO test split index shape and check that four types of classes work
        print("split_idx.shape:", split_idx.shape)

        # these chunks are of shape (num_chunks, seq_len, feature_dim) for eeg,
        # bandpower, and appendage data, and (num_chunks, seq_len) for tokens
        # and labels

        self.eeg_chunks = np.array(self.eeg_chunks, dtype=np.float32)
        self.bp_chunks = np.array(self.bp_chunks, dtype=np.float32)
        self.app_chunks = np.array(self.app_chunks, dtype=np.float32)
        self.token_chunks = np.array(self.token_chunks, dtype=np.int64)
        self.label_chunks = np.array(self.label_chunks, dtype=np.int64)

        self.eeg_chunks = self.eeg_chunks[split_idx]
        self.bp_chunks = self.bp_chunks[split_idx]
        self.app_chunks = self.app_chunks[split_idx]
        self.token_chunks = self.token_chunks[split_idx]
        self.label_chunks = self.label_chunks[split_idx]

        if verbose:
            print(
                f"{Colors.WARNING}Total number of chunks: {self.__len__()}{Colors.ENDC}"
            )
            # train_labels.mean() specifically gives open proportions
            print(
                f"Train: {len(split_idx)} chunks ({self.label_chunks.mean():.1%} open)"
            )
            print(
                f"Val:   {len(split_idx)} chunks ({self.label_chunks.mean():.1%} open)"
            )

    def __len__(self) -> int:
        return len(self.bp_chunks)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns EEG, bandpower, appendage data, VQ-VAE tokens, labels, durations, and
        masks for the given index from the training set.
        """

        eeg = self.eeg_chunks[index]
        bp = self.bp_chunks[index]
        apps = self.app_chunks[index]
        tokens = self.token_chunks[index]
        labels = self.label_chunks[index]

        # vqvae_tokens: (T,)
        reversed_tokens = tokens[::-1]
        durations = [1]
        for i in range(1, len(reversed_tokens)):
            current_tok = reversed_tokens[i]
            prev_tok = reversed_tokens[i - 1]

            if prev_tok == current_tok:
                durations.append(durations[-1] + 1)
            else:
                durations.append(1)

        durations = durations[::-1]

        masks: list[int | float] = [1]
        for i in range(len(tokens) - 1):
            current_tok = tokens[i]
            next_tok = tokens[i + 1]

            if next_tok == current_tok:
                masks.append(1e-4)
            else:
                masks.append(1)

        return (
            torch.tensor(eeg),
            torch.tensor(bp),
            torch.tensor(apps),
            torch.tensor(tokens),
            torch.tensor(labels),
            torch.tensor(durations),
            torch.tensor(masks),
        )

    def get_sampler_weights(self) -> tuple[list[int], torch.Tensor]:
        # self.label_chunks is (N, seq_len). we want the label that defines the
        # chunk. find the majority label of the sequence (the mode).
        chunk_labels = np.array([np.bincount(c).argmax() for c in self.label_chunks])

        # count instances of each class (1, 2, 3, 4)
        class_counts = np.bincount(chunk_labels)

        # calculate weight per class: total / (num classes * class count). this
        # makes rare classes have very high weights.
        total_samples = len(chunk_labels)
        weights_per_class = total_samples / (4 * (class_counts))
        print(weights_per_class.shape)
        quit()

        # assign the specific weight to every individual sample
        sample_weights = [weights_per_class[int(label)] for label in chunk_labels]

        # for the Loss function, we need the weight per class as a tensor
        class_weights_tensor = torch.tensor(weights_per_class, dtype=torch.float32)

        return sample_weights, class_weights_tensor
