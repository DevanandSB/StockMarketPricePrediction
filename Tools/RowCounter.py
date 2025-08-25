import pandas as pd
import os

# Define the folder and file names
# The '../' tells the script to go UP one directory from /Tools/
folder_path = '../Processed_Data'
csv_files = [
    'train_data.csv',
    'val_data.csv',
    'test_data.csv',
    'full_processed_data.csv'
]

# --- Step 1: Get the total row count from full_processed_data.csv ---
full_data_path = os.path.join(folder_path, 'full_processed_data.csv')
total_rows = 0
if os.path.exists(full_data_path):
    try:
        full_df = pd.read_csv(full_data_path)
        total_rows = len(full_df)
    except Exception as e:
        print(f"Error reading {full_data_path}: {e}")
else:
    print(f"Error: Base file 'full_processed_data.csv' not found in '{folder_path}'. Cannot calculate percentages.")

# --- Step 2: Loop through all CSVs to count rows, display columns, and calculate percentages ---
print("=" * 60)
print("CSV Row Counter and Data Availability Report")
print("=" * 60)

for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            row_count = len(df)
            columns = df.columns.tolist()

            print(f"\n--- File: {file_name} ---")
            print(f"➡️  Row Count: {row_count:,}")

            # Calculate and display percentage if it's not the full dataset
            if file_name != 'full_processed_data.csv' and total_rows > 0:
                percentage = (row_count / total_rows) * 100
                print(f"📊 Percentage of full dataset: {percentage:.2f}%")

            print("📜 Columns:")
            # Print columns in a more readable format
            for i in range(0, len(columns), 4):
                print("    " + " | ".join(columns[i:i + 4]))

        except Exception as e:
            print(f"\n--- File: {file_name} ---")
            print(f"Could not process file. Error: {e}")
    else:
        print(f"\n--- File: {file_name} ---")
        print("File not found.")

print("\n" + "=" * 60)
print("Report Complete.")
print("=" * 60)