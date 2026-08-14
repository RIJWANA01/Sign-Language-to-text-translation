import cv2
import mediapipe as mp
import csv
import os

# -------------------------------
# Enter the label (A, B, C...)
# -------------------------------
label = input("Enter Alphabet (A-Z): ").strip().upper()

# -------------------------------
# Create Data folder
# -------------------------------
os.makedirs("Data", exist_ok=True)

dataset_file = "Data/dataset.csv"

# -------------------------------
# MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------------------------------
# Open Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

sample_count = 0

print("\n==============================")
print("Press S -> Save Sample")
print("Press Q -> Quit")
print("==============================\n")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera.")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    landmarks = []

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        for lm in hand.landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)
            landmarks.append(lm.z)

    cv2.putText(
        frame,
        f"Alphabet : {label}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Samples : {sample_count}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Dataset Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):

        if len(landmarks) == 63:

            with open(dataset_file, "a", newline="") as file:

                writer = csv.writer(file)

                writer.writerow(landmarks + [label])

            sample_count += 1

            print(f"Sample {sample_count} saved")

        else:
            print("Hand not detected!")

    elif key == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()