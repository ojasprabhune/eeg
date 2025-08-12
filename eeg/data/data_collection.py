import mediapipe as mp
import cv2
import csv

# each joint is a landmark

# helps draw landmarks on screen
mp_drawing = mp.solutions.drawing_utils
# hands model
mp_hands = mp.solutions.hands

# initialize list of frames with landmarks
data = []

def hand_detection(hands, frame):
    frame_data = []
    # BGR to RGB
    # model requires 3 channel RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # set flag to false
    frame.flags.writeable = False

    # run model on image for hand pose detection
    results = hands.process(frame)

    # set flag back to true
    # allows us to draw on image and render
    frame.flags.writeable = True

    # recolor back to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # if solution has landmarks then render results on image
    if results.multi_hand_landmarks:
        # iterate through each landmark
        # num is for multiple hands
        # hand_landmarks represents the landmarks for that hand
        for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # pass in three variables:
            # 1. image
            # 2. hand (set of landmarks)
            # 3. HAND_CONNECTIONS represents the set of coordinates of relations between joints
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # iterate through hand landmarks list and figure out the joint number and position
            for landmark_num, landmark in enumerate(hand_landmarks.landmark):
                # create another list to include these values to append
                landmark_data = [landmark.x, landmark.y, landmark.z]
                frame_data.append(landmark_data)

    # add to data
    data.append(frame_data)
    return frame 

# getting camera / webcam (0, 1, 2 for connected webcams)
cap = cv2.VideoCapture(0)

# desired FPS
set_fps = 30
# set FPS to camera
cap.set(cv2.CAP_PROP_FPS, set_fps)
# verify FPS
fps = cap.get(cv2.CAP_PROP_FPS)
# log FPS for verification
print(f"Set FPS: {set_fps}\nFPS: {fps}")

# resource management: open hands as the term below and close it automatically
# two metrics:
# 1. detection: threshold for initial detection to be successful
# 2. tracking: threshold for tracking after initial detection
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    # reading frames while the capture is opened
    while cap.isOpened():
        # read each frame from webcam
        # ret and frame variables unpacking cap.read() function
        # a return value and image from webcame
        ret, frame = cap.read()

        # detection function: run all processing before displaying
        frame = hand_detection(hands, frame)

        # render image to screen using OpenCV with "Hand tracking." window title
        cv2.imshow("Hand tracking.", frame)

        # gracefully close the window
        # hit "q" and close window
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

# release webcam
cap.release()
# close windows
cv2.destroyAllWindows()

# temporarily open a csv writer that writes (hence the "w") to data.csv with a single newline between rows
with open("data.csv", "w", newline="\n") as writer:
    # create csv writer object
    writer = csv.writer(writer)
    # for each embedded list of data for a frame in data, write it as a row
    writer.writerows(data)