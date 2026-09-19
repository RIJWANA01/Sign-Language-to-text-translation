import cv2
import mediapipe as mp
import pickle
import time

# ==========================================
# LOAD THE TRAINED TWO-HAND MODEL
# ==========================================

MODEL_PATH = "models/gesture_model_isl.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!")
print("Model expects:", model.n_features_in_, "features")


# ==========================================
# MEDIAPIPE HAND SETUP
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ==========================================
# OPEN WEBCAM
# ==========================================

print("Opening webcam...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open webcam.")
    hands.close()
    exit()

# Give the camera time to start
time.sleep(2)

print("Camera started successfully!")


# ==========================================
# PREDICTION LOOP
# ==========================================

while True:

    # Read frame from webcam
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Failed to read webcam frame")
        break


    # Mirror the camera
    frame = cv2.flip(frame, 1)


    # Convert BGR to RGB for MediaPipe
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Detect hands
    results = hands.process(rgb)


    # ==========================================
    # CREATE FIXED 126 FEATURES
    #
    # Hand 1 = 21 landmarks × 3 = 63
    # Hand 2 = 21 landmarks × 3 = 63
    #
    # Total = 126
    # ==========================================

    landmarks = [0.0] * 126


    # ==========================================
    # IF HAND(S) DETECTED
    # ==========================================

    if results.multi_hand_landmarks:

        number_of_hands = len(
            results.multi_hand_landmarks
        )


        # Process maximum 2 hands
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


            # Store x, y, z values
            hand_data = []

            for lm in hand_landmarks.landmark:

                hand_data.append(lm.x)
                hand_data.append(lm.y)
                hand_data.append(lm.z)


            # First hand:
            # features 0 to 62
            #
            # Second hand:
            # features 63 to 125

            start_index = i * 63

            landmarks[
                start_index:start_index + 63
            ] = hand_data


        # ==========================================
        # MAKE PREDICTION
        # ==========================================

        try:

            prediction = model.predict(
                [landmarks]
            )[0]


            # Show prediction
            cv2.putText(
                frame,
                f"Prediction: {prediction}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


            # Show number of hands detected
            cv2.putText(
                frame,
                f"Hands detected: {number_of_hands}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        except Exception as e:

            print("Prediction error:", e)

            cv2.putText(
                frame,
                "Prediction Error",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )


    # ==========================================
    # NO HAND DETECTED
    # ==========================================

    else:

        cv2.putText(
            frame,
            "No hand detected",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ==========================================
    # QUIT INSTRUCTION
    # ==========================================

    cv2.putText(
        frame,
        "Press Q to Quit",
        (10, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # ==========================================
    # SHOW CAMERA WINDOW
    # ==========================================

    cv2.imshow(
        "ISL Sign Language Prediction",
        frame
    )


    # Quit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Closing prediction...")
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

hands.close()

cv2.destroyAllWindows()

print("Program closed successfully!")