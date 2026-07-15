# gesture_classes = {
#     "a": 1,
#     "b": 4,
#     "c": 4,
#     "d": 3,
#     "e": 1,
#     "f": 3,
#     "g": 2,
#     "h": 2,
#     "i": 3,
#     "j": 1,
#     "k": 3,
#     "l": 3,
#     "m": 1,
#     "n": 1,
#     "o": 4,
#     "p": 2,
#     "q": 2,
#     "r": 3,
#     "s": 1,
#     "t": 1,
#     "u": 3,
#     "v": 3,
#     "w": 3,
#     "x": 3,
#     "y": 3,
#     "z": 3,
# }

gesture_classes = {
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "e": 5,
    "f": 1,
    "g": 2,
    "h": 3,
    "i": 4,
    "j": 5,
    "k": 1,
    "l": 2,
    "m": 3,
    "n": 4,
    "o": 5,
    "p": 1,
    "q": 2,
    "r": 3,
    "s": 4,
    "t": 5,
    "u": 1,
    "v": 2,
    "w": 3,
    "x": 4,
    "y": 5,
    "z": 1,
}


def get_gesture_class(i: str | int, zero_based_idx: bool = True) -> int:
    """
    Uses dictionary lookup to return the gesture class of the inputted
    letter or index i.

    Either use zero-based indexing or return true class number.
    """
    if isinstance(i, int):
        letter = list(gesture_classes.keys())[i]
        return (
            gesture_classes[letter] - 1 if zero_based_idx else gesture_classes[letter]
        )
    else:
        return gesture_classes[i] - 1 if zero_based_idx else gesture_classes[i]
