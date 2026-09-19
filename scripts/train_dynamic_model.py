import os
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================
# SETTINGS
# ==========================================

DATA_PATH = "Data/dynamic_data"
MODEL_PATH = "models/dynamic_gesture_model.keras"

SEQUENCE_LENGTH = 60
FEATURES = 63


# ==========================================
# LOAD DATA
# ==========================================

X = []
y = []

for filename in sorted(os.listdir(DATA_PATH)):

    if not filename.endswith(".pkl"):
        continue

    filepath = os.path.join(DATA_PATH, filename)

    with open(filepath, "rb") as f:
        sequence = np.array(pickle.load(f))

    # Check shape
    if sequence.shape != (SEQUENCE_LENGTH, FEATURES):
        print(f"⚠️ Skipping {filename}: {sequence.shape}")
        continue

    # Get label from filename
    label = filename.split("_")[0].upper()

    if label not in ["J", "Z"]:
        continue

    X.append(sequence)
    y.append(label)


X = np.array(X)
y = np.array(y)

print("==========================================")
print("DYNAMIC DATASET")
print("==========================================")
print("X shape:", X.shape)
print("Labels:", {label: list(y).count(label) for label in ["J", "Z"]})


# ==========================================
# LABEL ENCODING
# ==========================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

y_categorical = to_categorical(
    y_encoded,
    num_classes=len(encoder.classes_)
)

print("Classes:", encoder.classes_)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Training sequences:", len(X_train))
print("Testing sequences:", len(X_test))


# ==========================================
# BUILD GRU MODEL
# ==========================================

model = Sequential([

    GRU(
        64,
        return_sequences=True,
        input_shape=(SEQUENCE_LENGTH, FEATURES)
    ),

    Dropout(0.3),

    GRU(32),

    Dropout(0.3),

    Dense(32, activation="relu"),

    Dense(
        len(encoder.classes_),
        activation="softmax"
    )
])


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ==========================================
# EARLY STOPPING
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# ==========================================
# TRAIN
# ==========================================

print("\n==========================================")
print("TRAINING GRU")
print("==========================================")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=4,
    callbacks=[early_stopping],
    verbose=1
)


# ==========================================
# EVALUATE
# ==========================================

y_probability = model.predict(
    X_test,
    verbose=0
)

y_prediction = np.argmax(
    y_probability,
    axis=1
)

y_actual = np.argmax(
    y_test,
    axis=1
)

accuracy = accuracy_score(
    y_actual,
    y_prediction
)

print("\n==========================================")
print("GRU PERFORMANCE")
print("==========================================")

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_actual,
        y_prediction,
        target_names=encoder.classes_
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

model.save(MODEL_PATH)

# Save label encoder
with open(
    "models/dynamic_label_encoder.pkl",
    "wb"
) as f:
    pickle.dump(encoder, f)


print("\n==========================================")
print("✅ DYNAMIC MODEL SAVED")
print("==========================================")

print(
    "Model:",
    MODEL_PATH
)

print(
    "Encoder:",
    "models/dynamic_label_encoder.pkl"
)
