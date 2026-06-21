import numpy as np

from eeg.viterbi_decoding import (
    Viterbi,
    get_viterbi_matrices,
    make_observation_matrix,
    sequence_to_letters,
)

initial_state, transition_matrix, emission_matrix = get_viterbi_matrices(
    word_count=50, p=1
)

viterbi = Viterbi(
    initial_state=initial_state,
    transition_matrix=transition_matrix,
    emission_matrix=emission_matrix,
)

sentence = "We need help."  # [3, 1, 1, 1, 1, 3, 2, 1, 3, 2]
sentence = "thecat"  # [1, 2, 1, 4, 1, 1]
observations = make_observation_matrix(
    sequence=[1, 2, 1, 4, 1, 1], num_classes=4, correct_mean=1
)

best_sequence = viterbi.calculate(observations=np.array([1, 2, 1, 4, 1, 1]))

print(best_sequence.shape)
print(sequence_to_letters(list(np.array(best_sequence, dtype=np.int64))))
