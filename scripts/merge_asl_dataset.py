import pandas as pd
import os

files = [
    "Data/Rijwana/dataset_rijwana.csv",
    "Data/Monika/dataset_monika.csv"
]

all_data = []

for file in files:

    print(f"\nReading: {file}")

    df = pd.read_csv(file, header=None)

    # Take only the first 126 columns = hand landmarks
    X = df.iloc[:, :126]

    # Second-last column = ASL alphabet
    y = df.iloc[:, -2]

    # Keep only valid alphabet labels A-Z
    valid = y.astype(str).str.fullmatch(r"[A-Z]")

    clean = pd.concat(
        [X[valid].reset_index(drop=True),
         y[valid].reset_index(drop=True)],
        axis=1
    )

    print("Valid samples:", len(clean))
    print("Labels:", clean.iloc[:, -1].value_counts().sort_index().to_dict())

    all_data.append(clean)


# Combine datasets
final_data = pd.concat(
    all_data,
    ignore_index=True
)

os.makedirs("Data", exist_ok=True)

output = "Data/asl_dataset.csv"

final_data.to_csv(
    output,
    index=False,
    header=False
)

print("\n====================================")
print("✅ CLEAN ASL DATASET CREATED")
print("====================================")
print("File:", output)
print("Shape:", final_data.shape)

print("\nFinal label distribution:")
print(
    final_data.iloc[:, -1]
    .value_counts()
    .sort_index()
)

print("\n✅ Done!")