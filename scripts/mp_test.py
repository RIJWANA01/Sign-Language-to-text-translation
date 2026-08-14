import mediapipe as mp

print("Before Hands")

hands = mp.solutions.hands.Hands()

print("After Hands")

hands.close()