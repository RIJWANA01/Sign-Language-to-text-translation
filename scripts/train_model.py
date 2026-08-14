import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# -----------------------------
# Load Dataset
# -----------------------------
dataset = pd.read_csv("Data/dataset.csv", header=None)

# Features (63 landmark values)
X = dataset.iloc[:, :-1]

# Labels (A, B, C...)
y = dataset.iloc[:, -1]

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)

with open("models/gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained successfully!")
print("✅ Model saved as models/gesture_model.pkl")