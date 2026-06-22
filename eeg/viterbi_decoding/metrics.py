def compute_cer(pred: str, true: str, verbose=False) -> float:
    """
    Calculates the Character Error Rate between two equally long strings.
    """
    n = min(len(pred), len(true))
    if n == 0:
        return 0.0

    if verbose:
        print(f"True sentence: {true}")
        print(f"Predicted sentence: {pred}")

    correct = sum(pred[i] == true[i] for i in range(n))
    return round(correct / len(true), 2)


def compute_wer(pred_words: list[str], true_words: list[str]) -> float:
    """
    Calculates the Word Error Rate between two equally long strings.
    """
    substitutions = sum(1 for true, pred in zip(true_words, pred_words) if true != pred)
    total_words = len(true_words)
    wer = substitutions / total_words
    return wer
