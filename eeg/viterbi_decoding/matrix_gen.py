import numpy as np
from numpy.typing import NDArray

from eeg.gesture2hand import Colors, get_gesture_class

alphabet = "abcdefghijklmnopqrstuvwxyz"


def get_viterbi_matrices(
    word_count: int, p: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Returns a tuple of:
        - an initial equally probable state matrix of 26 letters,
        - a transition matrix based on a bigram language model,
        - and an emission matrix to produce the probability of a letter given a
          class
    """

    if word_count not in (50, 30, 25):
        print(
            f"{Colors.FAIL}Warning: word_count {word_count} not valid. Defaulting to 50.{Colors.ENDC}"
        )
        word_count = 50

    # --- initial matrix ---
    initial_state = np.array([1 / 26] * 26)

    # --- transition matrix ---
    transition_matrix = np.array([])

    words_file = open(f"eeg/viterbi_decoding/data/{word_count}words.txt", "r")
    words = words_file.readline().split(", ")
    words[-1] = words[-1].rstrip("\n")
    words = [word.lower() for word in words]

    sentences_file = open("eeg/viterbi_decoding/data/sentences.txt", "r")
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
    for word in sentence_words:
        for i in range(len(word) - 1):
            letter1 = alphabet.index(word[i])
            letter2 = alphabet.index(word[i + 1])
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
    sequence = [x - 1 for x in sequence]

    seq_length = len(sequence)
    obs_mat = np.zeros((seq_length, num_classes))

    for i in range(len(obs_mat)):
        for j in range(len(obs_mat[i])):
            if j == sequence[i]:
                obs_mat[i, j] = correct_mean
            else:
                obs_mat[i, j] = (1 - correct_mean) / 3

    return obs_mat


def sequence_to_letters(sequence: list[int]) -> str:
    return str([alphabet[i] for i in sequence])
