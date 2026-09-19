import cv2
import mediapipe as mp
import pandas as pd
import os
import time

# ==========================================
# GET COLLECTOR AND SIGN
# ==========================================

collector = input("Enter collector name: ").strip().upper()
label = input("Enter alphabet/sign: ").strip().upper()

# ==========================================
# SETTINGS
# ==========================================

TOTAL_SAMPLES = 50
SAVE_INTERVAL = 0.5

# ==========================================
# CREATE INDIVIDUAL FOLDER AND CSV
# ==========================================

collector_folder = os.path.join("Data", collector.capitalize())

os.makedirs(collector_folder, exist_ok=True)

dataset_path = os.path.join(
    collector_folder,
    f"dataset_{collector.lower()}.csv"
)

print("\n===================================")
print("DATA COLLECTION STARTED")
print("Collector:", collector)
print("Sign:", label)
print("Saving to:", dataset_path)
print("===================================\n")

# ==========================================
# MEDIAPIPE SETUP
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# OPEN CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

sample_count = 0
last_save_time = time.time()

# ==========================================
# DATA COLLECTION LOOP
# ==========================================

while sample_count < TOTAL_SAMPLES:

    success, frame = cap.read()

    if not success:
        print("ERROR: Cannot read camera")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # --------------------------------------
    # ALWAYS CREATE 126 FEATURES
    # 63 = first hand
    # 63 = second hand
    # --------------------------------------

    landmarks = [0.0] * 126

    hands_detected = 0

    if results.multi_hand_landmarks:

        hands_detected = len(results.multi_hand_landmarks)

        for i, hand_landmarks in enumerate(
            results.multi_hand_landmarks
        ):

            if i >= 2:
                break

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            hand_data = []

            for lm in hand_landmarks.landmark:

                hand_data.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            # Save first hand → 0 to 62
            # Save second hand → 63 to 125

            start = i * 63

            landmarks[start:start + 63] = hand_data

        # --------------------------------------
        # AUTOMATIC SAVE
        # --------------------------------------

        current_time = time.time()

        if current_time - last_save_time >= SAVE_INTERVAL:

            # 126 landmarks + label + collector
            data = landmarks + [label, collector]

            df = pd.DataFrame([data])

            df.to_csv(
                dataset_path,
                mode="a",
                header=not os.path.exists(dataset_path),
                index=False
            )

            sample_count += 1

            last_save_time = current_time

            print(
                f"✓ SAVED {sample_count}/{TOTAL_SAMPLES} | "
                f"Sign: {label} | "
                f"Hands detected: {hands_detected}"
            )

    # ==========================================
    # DISPLAY INFORMATION
    # ==========================================

    cv2.putText(
        frame,
        f"Collector: {collector}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Sign: {label}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {sample_count}/{TOTAL_SAMPLES}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Hands detected: {hands_detected}",
        (10, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press Q to stop",
        (10, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "ISL Data Collection",
        frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLOSE EVERYTHING
# ==========================================

cap.release()
hands.close()
cv2.destroyAllWindows()

print("\n===================================")
print("DATA COLLECTION COMPLETED!")
print("Total samples:", sample_count)
print("Saved in:", dataset_path)
print("===================================")