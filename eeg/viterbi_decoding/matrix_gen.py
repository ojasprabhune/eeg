"""
A file for generating matrixes used in Viterbi or language model decoding.
Generates sequences like observation matrices or emission matrices.
"""

import re

import numpy as np
from numpy.typing import NDArray

from eeg.gesture2hand import Colors, gesture_classes, get_gesture_class

alphabet = "abcdefghijklmnopqrstuvwxyz"


def get_viterbi_matrices(
    num_words: int, num_sentences: int, p: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Returns a tuple of:
        - an initial equally probable state matrix of 26 letters,
        - a transition matrix based on a bigram language model,
        - and an emission matrix to produce the probability of a letter given a
          class
    """

    if num_words not in (150, 50, 25, 10):
        raise ValueError(
            f"{Colors.FAIL}Warning: num_words {num_words} not valid. Choose between 150, 50, 25, or 10.{Colors.ENDC}"
        )

    # --- initial matrix ---
    initial_state = np.array([1 / 26] * 26)

    # --- transition matrix ---
    transition_matrix = np.array([])

    words_file = open(f"eeg/viterbi_decoding/data/{num_words}words.txt", "r")
    words = words_file.readline().split(", ")
    words[-1] = words[-1].rstrip("\n")
    words = [word.lower() for word in words]

    sentences_file = open(
        f"eeg/viterbi_decoding/data/{num_sentences}sentences.txt", "r"
    )
    sentences = sentences_file.readlines()

    sentences_of_words = []
    for sentence in sentences:
        list_of_words = sentence.split()
        for i, word in enumerate(list_of_words):
            list_of_words[i] = word.lower().rstrip(".?,!")
        sentences_of_words.append(list_of_words)

    sentence_words = [
        subword for subsentence in sentences_of_words for subword in subsentence
    ]

    counts = np.ones((len(alphabet), len(alphabet)))  # add-1 (laplace) smoothing

    # increase count per bigram
    for sentence in sentences_of_words:
        # recreate the continuous rentence string Viterbi will actually see
        continuous_sentence = "".join(sentence)
        for i in range(len(continuous_sentence) - 1):
            letter1 = alphabet.index(continuous_sentence[i])
            letter2 = alphabet.index(continuous_sentence[i + 1])
            counts[letter1, letter2] += 1

    # normalize rows
    transition_matrix = counts / counts.sum(axis=1, keepdims=True)

    # --- emission matrix ---
    confusion_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i == j:
                confusion_matrix[i, j] = p
            else:
                confusion_matrix[i, j] = (1 - p) / 3
    emission_matrix = np.zeros((26, 4))

    for i in range(len(emission_matrix)):
        gesture_class = get_gesture_class(i)
        emission_matrix[i] = confusion_matrix[gesture_class]

    return initial_state, transition_matrix, emission_matrix


def make_observation_matrix(
    sequence: list[int], num_classes: int, correct_mean: float
) -> NDArray[np.float64]:
    """
    Returns an observation matrix of shape (T, num_classes) where T is the
    length of the input sequence. Each row corresponds to a timestep and each
    column corresponds to a class. The values in the matrix represent the
    probability of observing a particular class at a given timestep, based on
    the input sequence and the specified correct_mean. The correct class for
    each timestep is assigned a probability of correct_mean, while the other
    classes are assigned a probability of (1 - correct_mean) / (num_classes - 1).

    Does not use zero-based indexing for the input sequence, so the input
    sequence should contain class labels in the range [1, num_classes]. The
    function will internally convert these to zero based indexing for processing.
    """

    sequence = [x - 1 for x in sequence]

    seq_length = len(sequence)
    obs_mat = np.zeros((seq_length, num_classes))

    for i in range(seq_length):
        for j in range(num_classes):
            if j == sequence[i]:
                obs_mat[i, j] = correct_mean
            else:
                obs_mat[i, j] = (1 - correct_mean) / (num_classes - 1)

    return obs_mat


def make_placeholder_feature_sequences(
    sentences: list[str],
    num_classes: int = 4,
    correct_mean: float = 0.8,
) -> tuple[NDArray, NDArray]:
    """
    Returns:
        features: (N, max_T, 4)
        masks:    (N, max_T)

    where:
        masks[i, t] = 1 if timestep is real
                      0 if timestep is padding
    """

    cleaned_sentences = [
        re.sub(r"[^a-z]", "", sentence.lower()) for sentence in sentences
    ]

    max_len = max(len(sentence) for sentence in cleaned_sentences)

    features = []
    masks = []

    for sentence in cleaned_sentences:
        T = len(sentence)

        gesture_sequence = [
            get_gesture_class(letter, zero_based_idx=False) for letter in sentence
        ]

        obs_mat = make_observation_matrix(
            gesture_sequence,
            num_classes=num_classes,
            correct_mean=correct_mean,
        )

        padded = np.zeros((max_len, num_classes), dtype=np.float32)
        padded[:T] = obs_mat

        mask = np.zeros(max_len, dtype=np.int64)
        mask[:T] = 1

        features.append(padded)
        masks.append(mask)

    return np.stack(features), np.stack(masks)


def sequence_to_letters(sequence: NDArray[np.int64]) -> str:
    return "".join(alphabet[i] for i in sequence)


def get_sentence_data(idx: int) -> tuple[list[int], str, list[str]]:
    """
    Takes in an index to return a sentence from the corpus. It returns the
    integer version of the sentence and the sentence string formatted.
    """
    # shift 1-4 to 0-3
    gesture_classes_shifted = {k: v - 1 for k, v in gesture_classes.items()}

    sentences_file = open("eeg/viterbi_decoding/data/150sentences.txt", "r")
    sentences = [
        sentence.strip().lower().rstrip(".?,!")
        for sentence in sentences_file.readlines()
    ]

    sentence_words = sentences[idx].split(" ")
    sentence = sentences[idx].replace(" ", "")
    sequence = [
        gesture_classes_shifted[c] for c in sentence if c in gesture_classes_shifted
    ]

    return sequence, sentence, sentence_words


def reconstruct_words(pred_str: str, gt_words: list[str]) -> list[str]:
    pred_words = []
    idx = 0

    for word in gt_words:
        pred_words.append(pred_str[idx : idx + len(word)])
        idx += len(word)

    return pred_words
