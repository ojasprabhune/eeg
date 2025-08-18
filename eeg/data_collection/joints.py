from enum import Enum
from typing import Self


# joints enumerator with all mappings of joints
class Joint(Enum):
    W = 0  # WRIST
    TC = 1  # THUMB_CMC
    TM = 2  # THUMB_MCP
    TI = 3  # THUMB_IP
    TT = 4  # THUMB_TIP
    IM = 5  # INDEX_FINGER_MCP
    IP = 6  # INDEX_FINGER_PIP
    ID = 7  # INDEX_FINGER_DIP
    IT = 8  # INDEX_FINGER_TIP
    MM = 9  # MIDDLE_FINGER_MCP
    MP = 10  # MIDDLE_FINGER_PIP
    MD = 11  # MIDDLE_FINGER_DIP
    MT = 12  # MIDDLE_FINGER_TIP
    RM = 13  # RING_FINGER_MCP
    RP = 14  # RING_FINGER_PIP
    RD = 15  # RING_FINGER_DIP
    RT = 16  # RING_FINGER_TIP
    PM = 17  # PINKY_MCP
    PP = 18  # PINKY_PIP
    PD = 19  # PINKY_DIP
    PT = 20  # PINKY_TIP

    def from_str(name: str) -> Self:
        name: str = name.upper()
        return getattr(Joint, name)
