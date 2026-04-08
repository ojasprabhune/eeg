import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
from pathlib import Path


# Model definition (from scripts/data/train_hand_classifier.py)
class HandClassifier(nn.Module):
    def __init__(self, vocab_size: int = 4, embedding_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(63, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)
        self.fc4 = nn.Linear(embedding_dim, embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.head(x)
        return x


# Initialize model and load checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HandClassifier()
ckpt_path = Path(
    "models/hand_classifier/hand_classifier_epoch_100.pth"
).resolve()
if ckpt_path.exists():
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded checkpoint from {ckpt_path}")
else:
    print(f"Warning: Checkpoint not found at {ckpt_path}")

model.to(device)
model.eval()

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        label_text = "No hand detected"

        if results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Extract 63 joints (21 landmarks * 3 coordinates)
            # Use multi_hand_landmarks for normalized coordinates (as in train_hand_classifier.py dataset logic)
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks_list = []
            for landmark in hand_landmarks.landmark:
                landmarks_list.extend([landmark.x, landmark.y, landmark.z])

            # Run inference
            input_tensor = (
                torch.tensor(landmarks_list, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            with torch.no_grad():
                logits = model(input_tensor)
                prediction = torch.argmax(logits, dim=-1).item()

            # Convert class (0-3) back to original labels (1-4) if needed,
            # but user just asked for "current label"
            label_text = f"Class: {prediction + 1}"

        # Display label on screen
        cv2.putText(
            frame,
            label_text,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Hand Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
