import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

# --------------------------------
# SETTINGS
# --------------------------------
SEQUENCE_LENGTH = 60
DATA_PATH = "Data/dynamic_data"

os.makedirs(DATA_PATH, exist_ok=True)

# --------------------------------
# SELECT SIGN
# --------------------------------
label = input("Enter dynamic sign (J/Z): ").strip().upper()

if label not in ["J", "Z"]:
    print("❌ Please enter only J or Z")
    exit()

# --------------------------------
# MEDIAPIPE
# --------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --------------------------------
# WEBCAM
# --------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("\n======================================")
print(f"Dynamic Sign: {label}")
print("Press S to START recording")
print("Perform the complete sign movement")
print("60 frames will be captured automatically")
print("Press Q to quit")
print("======================================\n")

recording = False
sequence = []
sample_count = len(
    [f for f in os.listdir(DATA_PATH) if f.startswith(label + "_")]
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # --------------------------------
    # GET LANDMARKS
    # --------------------------------
    landmarks = None

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        landmarks = []

        for lm in hand.landmark:
            landmarks.extend([
                lm.x,
                lm.y,
                lm.z
            ])

    # --------------------------------
    # RECORDING
    # --------------------------------
    if recording:

        if landmarks is not None:

            sequence.append(landmarks)

        else:

            # If hand temporarily disappears,
            # add zeros for this frame
            sequence.append([0] * 63)

        current_frames = len(sequence)

        cv2.putText(
            frame,
            f"RECORDING: {current_frames}/{SEQUENCE_LENGTH}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # --------------------------------
        # AUTOMATICALLY SAVE
        # --------------------------------
        if len(sequence) >= SEQUENCE_LENGTH:

            filename = os.path.join(
                DATA_PATH,
                f"{label}_{sample_count}.pkl"
            )

            with open(filename, "wb") as f:
                pickle.dump(
                    np.array(sequence),
                    f
                )

            print(f"✅ Saved: {filename}")

            sample_count += 1
            sequence = []
            recording = False

    else:

        cv2.putText(
            frame,
            "Press S to START",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Sign: {label}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {sample_count}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Dynamic ASL Data Collection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------
    # START RECORDING
    # --------------------------------
    if key == ord("s") and not recording:

        sequence = []
        recording = True

        print("\n🔴 Recording started!")
        print("Perform the complete sign now...")

    # --------------------------------
    # QUIT
    # --------------------------------
    elif key == ord("q"):

        break


cap.release()
hands.close()
cv2.destroyAllWindows()

print("\n✅ Data collection stopped.")