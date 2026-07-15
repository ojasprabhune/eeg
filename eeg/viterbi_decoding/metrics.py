def _edit_distance(a: list, b: list) -> int:
    """
    Computes the Levenshtein edit distance between two sequences. Used by
    both compute_cer (character sequences) and compute_wer (word sequences).
    """
    m, n = len(a), len(b)

    # dp[i][j] = edit distance between a[:i] and b[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i  # deletions
    for j in range(n + 1):
        dp[0][j] = j  # insertions

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # match, no cost
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # deletion
                    dp[i][j - 1],  # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[m][n]


def compute_cer(pred: str, true: str, verbose=False) -> float:
    """
    Calculates the Character Error Rate between two strings using
    Levenshtein edit distance. Handles insertions, deletions, and
    substitutions, so misaligned strings are handled correctly.
    """
    if len(true) == 0:
        return 0.0
    if verbose:
        print(f"True sentence: {true}")
        print(f"Predicted sentence: {pred}")

    distance = _edit_distance(list(pred), list(true))
    return round(distance / len(true), 2)


def compute_wer(pred_words: list[str], true_words: list[str]) -> float:
    """
    Calculates the Word Error Rate between two word sequences using
    Levenshtein edit distance. Handles insertions, deletions, and
    substitutions at the word level.
    """
    if len(true_words) == 0:
        return 0.0

    distance = _edit_distance(pred_words, true_words)
    return distance / len(true_words)
