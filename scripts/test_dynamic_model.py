import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "models/dynamic_gesture_model.keras"
ENCODER_PATH = "models/dynamic_label_encoder.pkl"

SEQUENCE_LENGTH = 60
FEATURES = 63

CONFIDENCE_THRESHOLD = 70


# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

print("====================================")
print("DYNAMIC ASL MODEL TEST")
print("====================================")
print("Classes:", encoder.classes_)
print("Press Q to quit")


# ==========================================
# MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

sequence = deque(maxlen=SEQUENCE_LENGTH)


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)


    # ======================================
    # HAND DETECTED
    # ======================================

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


        if len(landmarks) == FEATURES:

            sequence.append(landmarks)


        # ==================================
        # PREDICT ONLY WHEN 60 REAL FRAMES
        # ==================================

        if len(sequence) == SEQUENCE_LENGTH:

            input_data = np.array(
                sequence,
                dtype=np.float32
            )

            input_data = np.expand_dims(
                input_data,
                axis=0
            )

            probabilities = model.predict(
                input_data,
                verbose=0
            )[0]

            class_index = np.argmax(
                probabilities
            )

            confidence = (
                probabilities[class_index] * 100
            )

            prediction = encoder.inverse_transform(
                [class_index]
            )[0]

            if confidence >= CONFIDENCE_THRESHOLD:

                display_prediction = prediction

            else:

                display_prediction = "?"


        else:

            display_prediction = "Collecting..."

            confidence = 0.0


        # ==================================
        # DISPLAY HAND STATUS
        # ==================================

        cv2.putText(
            frame,
            "Hand detected",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # ======================================
    # NO HAND DETECTED
    # ======================================

    else:

        # IMPORTANT:
        # Clear the sequence
        sequence.clear()

        display_prediction = "No hand"
        confidence = 0.0

        cv2.putText(
            frame,
            "No hand detected",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.putText(
        frame,
        "DYNAMIC ASL TEST",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 150, 136),
        2
    )

    cv2.putText(
        frame,
        f"Prediction: {display_prediction}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.1f}%",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
        (20, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Dynamic ASL Model",
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
hands.close()
cv2.destroyAllWindows()