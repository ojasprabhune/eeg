gesture_experiments = {
    "standard": {
        "a": 1,
        "b": 4,
        "c": 4,
        "d": 3,
        "e": 1,
        "f": 3,
        "g": 2,
        "h": 2,
        "i": 3,
        "j": 1,
        "k": 3,
        "l": 3,
        "m": 1,
        "n": 1,
        "o": 4,
        "p": 2,
        "q": 2,
        "r": 3,
        "s": 1,
        "t": 1,
        "u": 3,
        "v": 3,
        "w": 3,
        "x": 3,
        "y": 3,
        "z": 3,
    },
    "asl_8_letters": {
        "e": 1,
        "t": 1,
        "h": 2,
        "f": 2,
        "i": 3,
        "r": 3,
        "o": 4,
        "c": 4,
    },
    "common_8_letters": {
        "n": 1,
        "t": 2,
        "e": 3,
        "s": 4,
        "a": 4,
        "o": 3,
        "r": 2,
        "i": 1,
    },
    "6_letters": {
        "e": 1,
        "t": 2,
        "a": 3,
        "o": 4,
        "i": 5,
        "n": 6,
    },
}


def get_gesture_class(
    i: str | int, experiment: str, zero_based_idx: bool = True
) -> int:
    """
    Uses dictionary lookup to return the gesture class of the inputted
    letter or index i.

    Either use zero-based indexing or return true class number.
    """
    if isinstance(i, int):
        letter = list(gesture_experiments[experiment].keys())[i]
        return (
            gesture_experiments[experiment][letter] - 1
            if zero_based_idx
            else gesture_experiments[experiment][letter]
        )
    else:
        return (
            gesture_experiments[experiment][i] - 1
            if zero_based_idx
            else gesture_experiments[experiment][i]
        )
