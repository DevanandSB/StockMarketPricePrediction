import pandas as pd
import os


def simple_row_count(folder_path="Processed_Data"):
    """Simple function to count rows in processed CSV files"""
    print("SIMPLE ROW COUNT")
    print("=" * 30)

    if not os.path.exists(folder_path):
        print("Processed_Data folder not found!")
        return

    files_to_check = [
        'train_data.csv',
        'val_data.csv',
        'test_data.csv',
        'full_processed_data.csv'
    ]

    for file_name in files_to_check:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"{file_name}: {len(df):,} rows")
            except Exception as e:
                print(f"{file_name}: Error - {e}")
        else:
            print(f"{file_name}: File not found")


# Run the simple version
simple_row_count()