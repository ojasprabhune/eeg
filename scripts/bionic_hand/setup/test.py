import time

import numpy as np
from rustypot import Scs0009PyController

# Side
Side = 1  # 1=> Right Hand // 2=> Left Hand

# Speed
MaxSpeed = 7
CloseSpeed = 3

# Fingers middle poses
MiddlePos = [5, 0, -5, -7, -4, 0, -6, 8]  # replace values by your calibration results

c = Scs0009PyController(
    serial_port="/dev/tty.usbmodem5A7C1217101",
    baudrate=1000000,
    timeout=0.5,
)


def main():
    c.write_torque_enable(1, 1)  # 1 = On / 2 = Off / 3 = Free

    while True:
        # --- 1. WAKING UP ---
        CloseHand()
        time.sleep(1.0)
        OpenHand_Progressive()
        time.sleep(1.0)

        # --- 2. COUNTING 1, 2, 3, 4 ---
        CloseHand()
        time.sleep(0.5)
        Count_One()
        time.sleep(1.0)
        Count_Two()
        time.sleep(1.0)
        Count_Three()
        time.sleep(1.0)
        Count_Four()
        time.sleep(1.5)

        # --- 3. FUN & EXPRESSIVE ---
        RockOn()
        time.sleep(1.5)

        ThumbsUp()
        time.sleep(1.5)

        Perfect()
        time.sleep(1.5)

        Italian_Hand()  # The "Chef's Kiss"
        time.sleep(1.5)

        # --- 4. ATTITUDE ---
        Nonono()
        time.sleep(0.5)  # Nonono has built-in sleep

        Swag()
        time.sleep(1.5)

        # --- 5. THE CREEPY SPIDER CRAWL ---
        OpenHand()
        time.sleep(0.5)
        Spider()
        time.sleep(0.5)

        # --- 6. FLEXING ---
        SpreadHand()
        time.sleep(0.8)
        ClenchHand()
        time.sleep(0.8)
        SpreadHand()
        time.sleep(0.8)

        # --- 7. PLAYFUL MOVES ---
        Scissors()
        time.sleep(0.5)

        Pinched()
        time.sleep(1.0)

        # --- 8. THE GRAND FINALE (Fast Waving/Clapping) ---
        for _ in range(3):
            CloseHand()
            time.sleep(0.3)
            OpenHand()
            time.sleep(0.3)

        # Long pause before the whole performance repeats!
        time.sleep(2.5)


# ==========================================
# NEW AWESOME GESTURES
# ==========================================


def Count_One():
    Move_Index(-35, 35, MaxSpeed)
    Move_Middle(90, -90, MaxSpeed)
    Move_Ring(90, -90, MaxSpeed)
    Move_Thumb(90, -90, MaxSpeed)


def Count_Two():
    Move_Index(-35, 35, MaxSpeed)
    Move_Middle(-35, 35, MaxSpeed)
    Move_Ring(90, -90, MaxSpeed)
    Move_Thumb(90, -90, MaxSpeed)


def Count_Three():
    Move_Index(-35, 35, MaxSpeed)
    Move_Middle(-35, 35, MaxSpeed)
    Move_Ring(-35, 35, MaxSpeed)
    Move_Thumb(90, -90, MaxSpeed)


def Count_Four():
    OpenHand()


def ThumbsUp():
    Move_Index(90, -90, MaxSpeed)
    Move_Middle(90, -90, MaxSpeed)
    Move_Ring(90, -90, MaxSpeed)
    Move_Thumb(-40, 40, MaxSpeed)


def RockOn():
    Move_Index(-35, 35, MaxSpeed)
    Move_Middle(90, -90, MaxSpeed)
    Move_Ring(-35, 35, MaxSpeed)
    Move_Thumb(90, -90, MaxSpeed)


def Italian_Hand():
    # Brings all fingertips together to touch the thumb
    Move_Index(50, -50, MaxSpeed)
    Move_Middle(50, -50, MaxSpeed)
    Move_Ring(50, -50, MaxSpeed)
    Move_Thumb(40, -40, MaxSpeed)


def Spider():
    # Rapidly wiggles half-bent fingers like a walking spider
    for _ in range(4):
        Move_Index(20, -20, MaxSpeed)
        Move_Middle(60, -60, MaxSpeed)
        Move_Ring(20, -20, MaxSpeed)
        Move_Thumb(60, -60, MaxSpeed)
        time.sleep(0.25)

        Move_Index(60, -60, MaxSpeed)
        Move_Middle(20, -20, MaxSpeed)
        Move_Ring(60, -60, MaxSpeed)
        Move_Thumb(20, -20, MaxSpeed)
        time.sleep(0.25)


# ==========================================
# ORIGINAL GESTURES (Preserved)
# ==========================================


def Swag():
    Move_Index(-35, 35, MaxSpeed)
    Move_Middle(90, -90, MaxSpeed)
    Move_Ring(-35, 35, MaxSpeed)
    Move_Thumb(90, -90, MaxSpeed)


def OpenHand():
    Move_Index(-35, 35, MaxSpeed)
    Move_Middle(-35, 35, MaxSpeed)
    Move_Ring(-35, 35, MaxSpeed)
    Move_Thumb(-35, 35, MaxSpeed)


def CloseHand():
    Move_Index(90, -90, CloseSpeed)
    Move_Middle(90, -90, CloseSpeed)
    Move_Ring(90, -90, CloseSpeed)
    Move_Thumb(90, -90, CloseSpeed + 1)


def OpenHand_Progressive():
    Move_Index(-35, 35, MaxSpeed - 2)
    time.sleep(0.2)
    Move_Middle(-35, 35, MaxSpeed - 2)
    time.sleep(0.2)
    Move_Ring(-35, 35, MaxSpeed - 2)
    time.sleep(0.2)
    Move_Thumb(-35, 35, MaxSpeed - 2)


def SpreadHand():
    if Side == 1:  # Right Hand
        Move_Index(4, 90, MaxSpeed)
        Move_Middle(-32, 32, MaxSpeed)
        Move_Ring(-90, -4, MaxSpeed)
        Move_Thumb(-90, -4, MaxSpeed)
    if Side == 2:  # Left Hand
        Move_Index(-60, 0, MaxSpeed)
        Move_Middle(-35, 35, MaxSpeed)
        Move_Ring(-4, 90, MaxSpeed)
        Move_Thumb(-4, 90, MaxSpeed)


def ClenchHand():
    if Side == 1:  # Right Hand
        Move_Index(-60, 0, MaxSpeed)
        Move_Middle(-35, 35, MaxSpeed)
        Move_Ring(0, 70, MaxSpeed)
        Move_Thumb(-4, 90, MaxSpeed)
    if Side == 2:  # Left Hand
        Move_Index(0, 60, MaxSpeed)
        Move_Middle(-35, 35, MaxSpeed)
        Move_Ring(-70, 0, MaxSpeed)
        Move_Thumb(-90, -4, MaxSpeed)


def Index_Pointing():
    Move_Index(-40, 40, MaxSpeed)
    Move_Middle(90, -90, MaxSpeed)
    Move_Ring(90, -90, MaxSpeed)
    Move_Thumb(90, -90, MaxSpeed)


def Nonono():
    Index_Pointing()
    for i in range(3):
        time.sleep(0.2)
        Move_Index(-10, 80, MaxSpeed)
        time.sleep(0.2)
        Move_Index(-80, 10, MaxSpeed)

    Move_Index(-35, 35, MaxSpeed)
    time.sleep(0.4)


def Perfect():
    if Side == 1:  # Right Hand
        Move_Index(50, -50, MaxSpeed)
        Move_Middle(0, -0, MaxSpeed)
        Move_Ring(-20, 20, MaxSpeed)
        Move_Thumb(65, 12, MaxSpeed)
    if Side == 2:  # Left Hand
        Move_Index(50, -50, MaxSpeed)
        Move_Middle(0, -0, MaxSpeed)
        Move_Ring(-20, 20, MaxSpeed)
        Move_Thumb(-12, -65, MaxSpeed)


def Victory():
    if Side == 1:  # Right Hand
        Move_Index(-15, 65, MaxSpeed)
        Move_Middle(-65, 15, MaxSpeed)
        Move_Ring(90, -90, MaxSpeed)
        Move_Thumb(90, -90, MaxSpeed)
    if Side == 2:  # Left Hand
        Move_Index(-65, 15, MaxSpeed)
        Move_Middle(-15, 65, MaxSpeed)
        Move_Ring(90, -90, MaxSpeed)
        Move_Thumb(90, -90, MaxSpeed)


def Pinched():
    if Side == 1:  # Right Hand
        Move_Index(90, -90, MaxSpeed)
        Move_Middle(90, -90, MaxSpeed)
        Move_Ring(90, -90, MaxSpeed)
        Move_Thumb(0, -75, MaxSpeed)
    if Side == 2:  # Left Hand
        Move_Index(90, -90, MaxSpeed)
        Move_Middle(90, -90, MaxSpeed)
        Move_Ring(90, -90, MaxSpeed)
        Move_Thumb(75, 5, MaxSpeed)


def Scissors():
    Victory()
    if Side == 1:  # Right Hand
        for i in range(3):
            time.sleep(0.2)
            Move_Index(-50, 20, MaxSpeed)
            Move_Middle(-20, 50, MaxSpeed)
            time.sleep(0.2)
            Move_Index(-15, 65, MaxSpeed)
            Move_Middle(-65, 15, MaxSpeed)
    if Side == 2:  # Left Hand
        for i in range(3):
            time.sleep(0.2)
            Move_Index(-20, 50, MaxSpeed)
            Move_Middle(-50, 20, MaxSpeed)
            time.sleep(0.2)
            Move_Index(-65, 15, MaxSpeed)
            Move_Middle(-15, 65, MaxSpeed)


# ==========================================
# MOTOR CONTROLS
# ==========================================


def Move_Index(Angle_1, Angle_2, Speed):
    c.write_goal_speed(1, Speed)
    time.sleep(0.0002)
    c.write_goal_speed(2, Speed)
    time.sleep(0.0002)
    Pos_1 = np.deg2rad(MiddlePos[0] + Angle_1)
    Pos_2 = np.deg2rad(MiddlePos[1] + Angle_2)
    c.write_goal_position(1, Pos_1)
    c.write_goal_position(2, Pos_2)
    time.sleep(0.005)


def Move_Middle(Angle_1, Angle_2, Speed):
    c.write_goal_speed(3, Speed)
    time.sleep(0.0002)
    c.write_goal_speed(4, Speed)
    time.sleep(0.0002)
    Pos_1 = np.deg2rad(MiddlePos[2] + Angle_1)
    Pos_2 = np.deg2rad(MiddlePos[3] + Angle_2)
    c.write_goal_position(3, Pos_1)
    c.write_goal_position(4, Pos_2)
    time.sleep(0.005)


def Move_Ring(Angle_1, Angle_2, Speed):
    c.write_goal_speed(5, Speed)
    time.sleep(0.0002)
    c.write_goal_speed(6, Speed)
    time.sleep(0.0002)
    Pos_1 = np.deg2rad(MiddlePos[4] + Angle_1)
    Pos_2 = np.deg2rad(MiddlePos[5] + Angle_2)
    c.write_goal_position(5, Pos_1)
    c.write_goal_position(6, Pos_2)
    time.sleep(0.005)


def Move_Thumb(Angle_1, Angle_2, Speed):
    c.write_goal_speed(7, Speed)
    time.sleep(0.0002)
    c.write_goal_speed(8, Speed)
    time.sleep(0.0002)
    Pos_1 = np.deg2rad(MiddlePos[6] + Angle_1)
    Pos_2 = np.deg2rad(MiddlePos[7] + Angle_2)
    c.write_goal_position(7, Pos_1)
    c.write_goal_position(8, Pos_2)
    time.sleep(0.005)


if __name__ == "__main__":
    main()
