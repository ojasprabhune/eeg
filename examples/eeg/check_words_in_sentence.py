"""
Check that all words in sentences are in the word list.
"""

words_file = open("eeg/viterbi_decoding/data/50words.txt", "r")
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

for sentence in sentences_of_words:
    for word in sentence:
        if word not in words:
            print(
                f"{word} is not in the word list in sentence #{sentences_of_words.index(sentence)}"
            )
