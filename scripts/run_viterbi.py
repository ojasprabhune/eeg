import numpy as np

from eeg.viterbi_decoding import (
    Viterbi,
    compute_cer,
    compute_wer,
    get_sentence_data,
    get_viterbi_matrices,
    reconstruct_words,
    sequence_to_letters,
)

num_vocab = 150

initial_state, transition_matrix, emission_matrix = get_viterbi_matrices(
    num_words=num_vocab, num_sentences=num_vocab, p=1
)

viterbi = Viterbi(
    initial_state=initial_state,
    transition_matrix=transition_matrix,
    emission_matrix=emission_matrix,
)

wers = []
accuracies = []

for i in range(num_vocab):
    gt_sequence, gt_sentence, gt_sentence_words = get_sentence_data(idx=i)

    pred_sequence = viterbi.calculate(observations=np.array(gt_sequence))
    pred_sentence = sequence_to_letters(pred_sequence)
    pred_sentence_words = reconstruct_words(pred_sentence, gt_sentence_words)

    wer = compute_wer(pred_sentence_words, gt_sentence_words)
    accuracy = compute_cer(pred_sentence, gt_sentence)
    wers.append(wer)
    accuracies.append(accuracy)

avg_wer = round(sum(wers) / num_vocab, 2)
avg_accuracy = round(sum(accuracies) / num_vocab, 2)
avg_cer = round(1 - avg_accuracy, 2)
print("avg accuracy:", avg_accuracy)
print("avg wer:", avg_wer)
print("avg cer:", avg_cer)
