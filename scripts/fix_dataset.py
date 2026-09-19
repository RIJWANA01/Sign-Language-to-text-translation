import csv

input_file = "Data/dataset.csv"
output_file = "Data/dataset_fixed.csv"

with open(input_file, "r", newline="") as infile:
    reader = csv.reader(infile)
    rows = list(reader)

# Find the correct number of columns from the header
expected_columns = len(rows[0])

valid_rows = [rows[0]]
removed_rows = 0

for i, row in enumerate(rows[1:], start=2):
    if len(row) == expected_columns:
        valid_rows.append(row)
    else:
        print(f"Removed problematic row: {i} ({len(row)} columns)")
        removed_rows += 1

with open(output_file, "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(valid_rows)

print("\nDataset fixed successfully!")
print(f"Expected columns: {expected_columns}")
print(f"Problematic rows removed: {removed_rows}")
print(f"Fixed dataset saved as: {output_file}")