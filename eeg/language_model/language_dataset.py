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
        mode: str = "train",
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
        self.mode = mode

        super().__init__()

        # --- sentences ---
        language_tokenizer = LanguageTokenizer()

        sentences_file = open(f"{label_sentence_path}150sentences.txt", "r")
        sentences = sentences_file.readlines()
        features, feature_masks = make_placeholder_feature_sequences(
            sentences, num_classes=5, correct_mean=0.6
        )

        # --- word boundaries ---
        # labels are just a flat run of letters with no space token, so we
        # can't recover word boundaries from them later. so grab the word
        # lengths now, while spaces are still in the raw sentences, and pad
        # them into a (N, max_num_words) matrix the same way features/labels
        # are padded.
        words_per_sentence = [
            re.sub(r"[^a-z ]", "", sentence.lower()).split() for sentence in sentences
        ]

        max_num_words = max(len(words) for words in words_per_sentence)
        word_lengths = np.zeros((len(sentences), max_num_words), dtype=np.int32)

        for i, words in enumerate(words_per_sentence):
            for j, word in enumerate(words):
                word_lengths[i, j] = len(word)

        sentences = [re.sub(r"[^a-z]", "", sentence.lower()) for sentence in sentences]
        labels = language_tokenizer.encode(sentences)
        label_masks = (labels != 0).type(torch.int)

        if print_shapes:
            print(f"{Colors.OKGREEN}Features shape: {features.shape}{Colors.ENDC}")
            print(
                f"{Colors.OKGREEN}Feature masks shape: {feature_masks.shape}{Colors.ENDC}"
            )
            print(f"{Colors.OKGREEN}Labels shape: {labels.shape}{Colors.ENDC}")
            print(
                f"{Colors.OKGREEN}Label masks shape: {label_masks.shape}{Colors.ENDC}"
            )

        # --- train-val split ---

        # index at 80% on time dim
        self.split_idx = int(len(features) * 0.8)

        self.features = np.array(features, dtype=np.float32)
        self.feature_masks = np.array(feature_masks, dtype=np.int32)
        self.labels = np.array(labels, dtype=np.int32)
        self.label_masks = np.array(label_masks, dtype=np.int32)
        self.word_lengths = np.array(word_lengths, dtype=np.int32)

        self.train_features = self.features[: self.split_idx, :, :]
        self.train_feature_masks = self.feature_masks[: self.split_idx, :]
        self.train_labels = self.labels[: self.split_idx, :]
        self.train_label_masks = self.label_masks[: self.split_idx, :]
        self.train_word_lengths = self.word_lengths[: self.split_idx, :]

        self.val_features = self.features[self.split_idx :, :, :]
        self.val_feature_masks = self.feature_masks[self.split_idx :, :]
        self.val_labels = self.labels[self.split_idx :, :]
        self.val_label_masks = self.label_masks[self.split_idx :, :]
        self.val_word_lengths = self.word_lengths[self.split_idx :, :]

        if print_shapes:
            print(f"{Colors.WARNING}Number of sequences: {self.__len__()}{Colors.ENDC}")

    def __len__(self) -> int:
        return (
            len(self.train_features) if self.mode == "train" else len(self.val_features)
        )

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the feature sequences, masks, label sequences, and word
        length boundaries in the chunk at the given index from either the
        training or validation set.
        """
        if self.mode == "train":
            features = self.train_features[index]
            feature_mask = self.train_feature_masks[index]
            labels = self.train_labels[index]
            label_mask = self.train_label_masks[index]
            word_lengths = self.train_word_lengths[index]
        else:
            features = self.val_features[index]
            feature_mask = self.val_feature_masks[index]
            labels = self.val_labels[index]
            label_mask = self.val_label_masks[index]
            word_lengths = self.val_word_lengths[index]
        return (
            torch.tensor(features),
            torch.tensor(feature_mask),
            torch.tensor(labels),
            torch.tensor(label_mask),
            torch.tensor(word_lengths),
        )
