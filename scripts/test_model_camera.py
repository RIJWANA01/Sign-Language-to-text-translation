import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load model
with open("models/gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

print("================================")
print("ASL CAMERA MODEL TEST")
print("================================")
print("Press Q to quit")

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

    if results.multi_hand_landmarks:

        features = []

        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            for lm in hand.landmark:
                features.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

        # Pad to 126 features
        if len(features) < 126:
            features.extend(
                [0.0] * (126 - len(features))
            )

        features = features[:126]

        X = np.array(features).reshape(1, -1)

        prediction = model.predict(X)[0]

        probabilities = model.predict_proba(X)[0]

        confidence = np.max(probabilities) * 100

        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}%",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Hands: {len(results.multi_hand_landmarks)}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "No hand detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "ASL Model Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()