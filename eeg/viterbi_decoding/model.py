import numpy as np
from numpy.typing import NDArray

from eeg.gesture2hand import Colors


class Viterbi:
    def __init__(
        self,
        initial_state: NDArray[np.float64],
        transition_matrix: NDArray[np.float64],
        emission_matrix: NDArray[np.float64],
    ) -> None:
        """
        Viterbi decoding for generating the "best path": the most likely
        sequence of letters that explains the observed Temporal model
        classifier predictions while also being consistent with the language
        model.

        The letters are the hidden states, and the outputs from the Temporal
        model are the observations.

        T: length of observation sequence
        N: number of states
        M: number of observation symbols

        initial_state: (N,)
        transition_matrix: (N, N)
        emission_matrix: (N, M)
        """

        self.N = 26
        self.M = 4

        self.initial_state = initial_state
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix

        print(
            f"{Colors.OKGREEN}Initialized Viterbi decoder with {self.N} states and {self.M} observation symbols.{Colors.ENDC}"
        )

    def calculate(self, observations: NDArray[np.float64]) -> NDArray[np.int64]:
        """
        Calculates the most probable path of hidden states to match the
        sequence of observations. In other words, this function predicts the
        most likely sentence(s) based on the gestures per letter using
        statistics and a language model.

        EEG --> Temporal class ouputs --> Viterbi sentences

        observations: (T, M)
            - contains class probability distributions over symbols ∈ [M]

        Returns a "best" sequence of letters in index form.
        """

        self.T = len(observations)
        self.observations = observations

        print(self.initial_state.shape)
        print(self.emission_matrix.shape)
        print(self.observations.shape)

        # --- initial scores ---

        scores = np.array(
            [
                self.initial_state[i] * self.emission_matrix[i, observations[0]]
                for i in range(self.N)
            ]
        )  # (N,)

        # --- moving forward in time ---
        sequence_scores = [list(scores)]  # will be (T, N)
        path_backpointers = [list(range(self.N))]  # will be (T, N)

        for t in range(1, self.T):  # loop through time steps
            new_scores = []
            backpointers = []

            for i in range(self.N):  # loop through each current state
                path_scores = []

                for j in range(self.N):  # loop through each past state
                    score = scores[j]
                    transition = self.transition_matrix[j, i]
                    emission = self.emission_matrix[i][observations[t]]

                    path_score = score * transition * emission
                    path_scores.append(path_score)

                max_score = max(path_scores)  # best path for current state
                best_backpointer = np.argmax(path_scores)
                new_scores.append(max_score)
                backpointers.append(best_backpointer)

            scores = new_scores  # (N,)
            sequence_scores.append(scores)
            path_backpointers.append(backpointers)

        sequence_scores = np.array(sequence_scores)
        path_backpointers = np.array(path_backpointers)

        # --- choosing best final path ---

        print(sequence_scores.shape)
        print(path_backpointers.shape)

        best_path_idx = np.argmax(sequence_scores[-1, :])  # index of best final state
        print(path_backpointers[:, best_path_idx])
        best_path: NDArray[np.int64] = np.array(
            path_backpointers[:, best_path_idx], dtype=np.int64
        )  # (T,)

        print(best_path_idx)
        print(best_path.shape)

        return best_path
