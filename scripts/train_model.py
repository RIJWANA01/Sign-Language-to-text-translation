import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 1. Load clean ASL dataset
# ==============================

DATA_PATH = "Data/asl_dataset.csv"
MODEL_PATH = "models/gesture_model.pkl"

dataset = pd.read_csv(DATA_PATH, header=None)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

print("Dataset shape:", dataset.shape)
print("Features:", X.shape[1])
print("Classes:", sorted(y.unique()))

# ==============================
# 2. Split dataset
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==============================
# 3. Train Random Forest
# ==============================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ==============================
# 4. Evaluate
# ==============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# 5. Save model
# ==============================

os.makedirs("models", exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("\n================================")
print("✅ MODEL TRAINED SUCCESSFULLY")
print("================================")
print("Model saved to:")
print(MODEL_PATH)