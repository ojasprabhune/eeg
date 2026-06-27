import re

import numpy as np
import torch
from torch.utils.data import Dataset

from eeg.gesture2hand import Colors
from eeg.viterbi_decoding import make_placeholder_feature_sequences

from .tokenizer import LanguageTokenizer


class LanguageDataset(Dataset):
    """
    Dataset for loading the outputs of the EEG classifier as inputs for
    the language model.
    """

    def __init__(
        self,
        features_sequences: torch.Tensor,
        label_sentence_path: str = "eeg/language_model/data/",
        device: str = "cuda",
        print_shapes: bool = False,
    ) -> None:
        """
        Input is a list of feature sequences, each of shape (T, 4), and the
        labels are a list. Each sequence in the features is the same max
        sequence length as each tokenized label sequence.
        """

        print(f"{Colors.HEADER}{Colors.BOLD}Initializing dataset...{Colors.ENDC}")
        self.print_shapes = print_shapes
        self.device = device

        super().__init__()

        # --- sentences ---
        language_tokenizer = LanguageTokenizer()

        sentences_file = open(f"{label_sentence_path}150sentences.txt", "r")
        sentences = sentences_file.readlines()
        features, feature_masks = make_placeholder_feature_sequences(sentences)

        sentences = [re.sub(r"[^a-z]", "", sentence.lower()) for sentence in sentences]
        labels = language_tokenizer.encode(sentences)

        max_seq_length = labels.shape[-1]

        label_mask_sequences = [] # 150 rows
        for sequence_mask in feature_masks:
            label_sequence_masks = [] # sequence length
            for i, mask_value in enumerate(sequence_mask):

                if i == 0:
                    label_sequence_masks.append([1] * 4) # add <SOS> token if first label

                elif sequence_mask[i] == 0 and sequence_mask[i - 1] == 1:
                    label_sequence_masks.append([1] * 4) # <EOS> token if padding started

                elif i == max_seq_length - 1:
                    label_sequence_masks.append([1] * 4) # <EOS> token if end of sequence and still valid

                else:
                    # add 4D vector for each value in the sequence
                    label_sequence_masks.append([mask_value] * 4) 

            label_mask_sequences.append(label_sequence_masks) # add sequence to full list

        label_masks = torch.tensor(label_mask_sequences)

        print(label_masks[0])
        print(labels[0])
        quit()

        if print_shapes:
            print(f"{Colors.OKGREEN}Features shape: {features.shape}{Colors.ENDC}")
            print(f"{Colors.OKGREEN}Feature masks shape: {feature_masks.shape}{Colors.ENDC}")
            print(f"{Colors.OKGREEN}Labels shape: {labels.shape}{Colors.ENDC}")
            print(f"{Colors.OKGREEN}Label masks shape: {label_masks.shape}{Colors.ENDC}")

        # --- train-val split ---

        # index at 80% on time dim
        self.split_idx = int(len(features) * 0.8)

        self.features = np.array(features, dtype=np.float32)
        self.feature_masks = np.array(feature_masks, dtype=np.int32)
        self.labels = np.array(labels, dtype=np.int32)
        self.label_masks = np.array(label_masks, dtype=np.int32)

        self.train_features = self.features[: self.split_idx, :, :]
        self.train_feature_masks = self.feature_masks[: self.split_idx, :]
        self.train_labels = self.labels[: self.split_idx, :]
        self.train_label_masks = self.label_masks[: self.split_idx, :]

        self.val_features = self.features[self.split_idx :, :, :]
        self.val_feature_masks = self.feature_masks[self.split_idx :, :]
        self.val_labels = self.labels[self.split_idx :, :]
        self.val_label_masks = self.label_masks[self.split_idx :, :]

        if print_shapes:
            print(f"{Colors.WARNING}Total # of chunks: {self.__len__()}{Colors.ENDC}")

    def __len__(self) -> int:
        return len(self.train_features)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the feature sequences, masks, and label sequences in the chunk
        at the given index from the training set.
        """

        features = self.train_features[index]
        feature_masks = self.train_feature_masks[index]
        labels = self.train_labels[index]
        label_masks = self.train_label_masks[index]

        return torch.tensor(features), torch.tensor(masks), torch.tensor(labels), torch.tensor(label_masks)

    def get_val_data(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the feature sequences, masks, and label sequences in the chunk
        at the given index from the validation set.
        """

        features = self.val_features[index]
        feature_masks = self.val_feature_masks[index]
        labels = self.val_labels[index]
        label_masks = self.val_label_masks[index]

        return torch.tensor(features), torch.tensor(masks), torch.tensor(labels), torch.tensor(label_masks)
