import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import re
import shutil

# Initialize FinBERT for sentiment analysis
try:
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    print("✓ FinBERT model loaded successfully")
except Exception as e:
    print(f"Error loading FinBERT: {e}")
    print("Please check internet connection for first-time model download")
    exit()


def create_output_folder():
    """Create output folder for processed data"""
    output_folder = "Processed_Data"
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    return output_folder


def parse_date(date_str):
    """Parse different date formats consistently"""
    if pd.isna(date_str):
        return pd.NaT

    # Convert to string if it's not already
    date_str = str(date_str)

    # Remove timezone information if present
    if '+' in date_str:
        date_str = date_str.split('+')[0].strip()

    # Remove time component if present
    if ' ' in date_str:
        date_str = date_str.split(' ')[0]

    # Try different date formats
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d')
    except:
        try:
            return pd.to_datetime(date_str, format='%d-%m-%Y')
        except:
            try:
                return pd.to_datetime(date_str, format='%m/%d/%Y')
            except:
                return pd.to_datetime(date_str, errors='coerce')


def remove_empty_columns(df, threshold=0.9):
    """Remove columns with too many missing values"""
    if df.empty:
        return df

    initial_cols = len(df.columns)

    # Calculate missing percentage for each column
    missing_percentage = df.isnull().mean()

    # Remove columns with more than threshold missing values
    columns_to_remove = missing_percentage[missing_percentage > threshold].index.tolist()

    if columns_to_remove:
        print(f"Removing columns with >{threshold * 100}% missing values: {columns_to_remove}")
        df = df.drop(columns=columns_to_remove)

    print(f"Columns removed: {len(columns_to_remove)} | Remaining columns: {len(df.columns)}")
    return df


def analyze_sentiment(text):
    """Analyze sentiment using FinBERT"""
    if pd.isna(text) or text == '' or text is None:
        return 0.0

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        return probs[0][1].item() - probs[0][2].item()  # Positive - Negative
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return 0.0


def process_news_data(news_path):
    """Process news data and add sentiment scores"""
    print("\n" + "=" * 50)
    print("PROCESSING NEWS DATA")
    print("=" * 50)

    try:
        news_df = pd.read_csv(news_path)
        print(f"✓ Loaded news data with {len(news_df)} articles")

        # Convert date column with proper parsing
        print("Parsing news dates...")
        news_df['date'] = news_df['Date'].apply(parse_date)

        # Remove rows with invalid dates
        initial_count = len(news_df)
        news_df = news_df.dropna(subset=['date'])
        removed_count = initial_count - len(news_df)
        print(f"Removed {removed_count} rows with invalid dates")

        # Combine title and description for better sentiment analysis
        news_df['content'] = news_df['News Title'] + '. ' + news_df['News Description'].fillna('')

        # If sentiment column exists, use it; otherwise analyze
        if 'Sentiment' in news_df.columns and not news_df['Sentiment'].isna().all():
            print("Using existing sentiment column")
            news_df['sentiment'] = pd.to_numeric(news_df['Sentiment'], errors='coerce').fillna(0)
        else:
            print("Analyzing news sentiment...")
            sentiments = []
            for content in tqdm(news_df['content'], desc="Processing sentiment"):
                sentiments.append(analyze_sentiment(content))
            news_df['sentiment'] = sentiments

        # Aggregate daily sentiment
        daily_sentiment = news_df.groupby('date')['sentiment'].agg(['mean', 'count']).reset_index()
        daily_sentiment.columns = ['date', 'avg_sentiment', 'news_count']

        print(f"✓ Created daily sentiment data with {len(daily_sentiment)} days")
        print(f"Date range: {daily_sentiment['date'].min()} to {daily_sentiment['date'].max()}")
        return daily_sentiment

    except Exception as e:
        print(f"Error processing news data: {e}")
        return pd.DataFrame(columns=['date', 'avg_sentiment', 'news_count'])


def add_technical_indicators(df):
    """Add common technical indicators to technical data"""
    print("Adding technical indicators...")

    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['MA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs.replace([np.inf, -np.inf], 0)))

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Volume indicators
    df['volume_ma'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma'].replace(0, 1)

    # Price changes
    df['price_change'] = df['Close'].pct_change()
    df['volatility'] = df['Close'].rolling(window=20).std()

    print("✓ Technical indicators added")
    return df


def process_technical_data(technical_folder):
    """Process all technical data files"""
    print("\n" + "=" * 50)
    print("PROCESSING TECHNICAL DATA")
    print("=" * 50)

    try:
        technical_files = list(Path(technical_folder).glob("*.csv"))
        print(f"Found {len(technical_files)} technical files")

        all_technical_data = []

        for file in tqdm(technical_files, desc="Processing technical files"):
            try:
                tech_df = pd.read_csv(file)
                print(f"\nProcessing {file.stem}")

                # Convert date column with proper parsing
                tech_df['date'] = tech_df['Date'].apply(parse_date)

                # Remove rows with invalid dates
                initial_count = len(tech_df)
                tech_df = tech_df.dropna(subset=['date'])
                removed_count = initial_count - len(tech_df)
                if removed_count > 0:
                    print(f"Removed {removed_count} rows with invalid dates")

                # Add technical indicators
                tech_df = add_technical_indicators(tech_df)

                # Add stock symbol from filename if not in data
                if 'Symbol' not in tech_df.columns:
                    symbol = file.stem.replace('_historical', '')
                    tech_df['Symbol'] = symbol
                else:
                    # Ensure Symbol is consistent
                    tech_df['Symbol'] = tech_df['Symbol'].astype(str)

                all_technical_data.append(tech_df)

            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

        if all_technical_data:
            result_df = pd.concat(all_technical_data, ignore_index=True)
            result_df = remove_empty_columns(result_df)
            print(f"✓ Technical data processed: {len(result_df)} rows")
            print(f"Technical date range: {result_df['date'].min()} to {result_df['date'].max()}")
            return result_df
        else:
            print("No technical data processed successfully")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error in technical data processing: {e}")
        return pd.DataFrame()


def process_fundamental_data(fundamental_folder):
    """Process all fundamental data files"""
    print("\n" + "=" * 50)
    print("PROCESSING FUNDAMENTAL DATA")
    print("=" * 50)

    try:
        fundamental_files = list(Path(fundamental_folder).glob("*.csv"))
        print(f"Found {len(fundamental_files)} fundamental files")

        all_fundamental_data = []

        for file in tqdm(fundamental_files, desc="Processing fundamental files"):
            try:
                fund_df = pd.read_csv(file)
                print(f"\nProcessing {file.stem}")

                # Convert date column with proper parsing
                fund_df['date'] = fund_df['Date'].apply(parse_date)

                # Remove rows with invalid dates
                initial_count = len(fund_df)
                fund_df = fund_df.dropna(subset=['date'])
                removed_count = initial_count - len(fund_df)
                if removed_count > 0:
                    print(f"Removed {removed_count} rows with invalid dates")

                # Clean column names
                column_mapping = {
                    'Market Cap': 'market_cap',
                    'Current Price': 'current_price',
                    'Stock P/E': 'pe_ratio',
                    'Book Value': 'book_value',
                    'Dividend Yield': 'dividend_yield',
                    'Face Value': 'face_value',
                    'Industry PE': 'industry_pe',
                    'Debt to equity': 'debt_to_equity',
                    'PEG Ratio': 'peg_ratio',
                    'OPM': 'opm',
                    'Promoter holding': 'promoter_holding',
                    'Price to book value': 'price_to_book',
                    'Sales growth': 'sales_growth',
                    'Profit growth': 'profit_growth',
                    'Dividend Payout': 'dividend_payout',
                    'Debt': 'debt',
                    'Free Cash Flow': 'free_cash_flow'
                }

                # Only rename columns that actually exist
                existing_columns = {k: v for k, v in column_mapping.items() if k in fund_df.columns}
                fund_df = fund_df.rename(columns=existing_columns)

                # Convert numeric columns that exist
                numeric_cols = ['market_cap', 'current_price', 'High', 'Low', 'pe_ratio',
                                'book_value', 'dividend_yield', 'ROCE', 'ROE', 'face_value',
                                'EPS', 'industry_pe', 'debt_to_equity', 'peg_ratio', 'Reserves',
                                'opm', 'promoter_holding', 'price_to_book', 'sales_growth',
                                'profit_growth', 'dividend_payout', 'debt', 'Volume', 'free_cash_flow']

                for col in numeric_cols:
                    if col in fund_df.columns:
                        fund_df[col] = pd.to_numeric(fund_df[col], errors='coerce')

                # Ensure Symbol column is consistent
                if 'Symbol' in fund_df.columns:
                    fund_df['Symbol'] = fund_df['Symbol'].astype(str)
                else:
                    # Extract symbol from filename
                    symbol = file.stem.replace('_historical_fundamentals_2015_2025', '')
                    fund_df['Symbol'] = symbol

                # Remove empty columns from fundamentals
                fund_df = remove_empty_columns(fund_df, threshold=0.8)

                all_fundamental_data.append(fund_df)

            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

        if all_fundamental_data:
            result_df = pd.concat(all_fundamental_data, ignore_index=True)
            result_df = remove_empty_columns(result_df)
            print(f"✓ Fundamental data processed: {len(result_df)} rows")
            print(f"Fundamental date range: {result_df['date'].min()} to {result_df['date'].max()}")
            return result_df
        else:
            print("No fundamental data processed successfully")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error in fundamental data processing: {e}")
        return pd.DataFrame()


def handle_missing_values(df):
    """Handle missing values appropriately"""
    print("Handling missing values...")

    if df.empty:
        return df

    # Remove columns that are completely empty or have very high missingness
    df = remove_empty_columns(df, threshold=0.8)

    # Forward fill for fundamental data (grouped by symbol)
    fundamental_cols = [col for col in df.columns if col in [
        'market_cap', 'current_price', 'pe_ratio', 'book_value', 'dividend_yield',
        'ROCE', 'ROE', 'face_value', 'EPS', 'industry_pe', 'debt_to_equity',
        'peg_ratio', 'Reserves', 'opm', 'promoter_holding', 'price_to_book',
        'sales_growth', 'profit_growth', 'dividend_payout', 'debt', 'free_cash_flow'
    ]]

    if fundamental_cols:
        # Only use columns that actually exist in the dataframe
        existing_fund_cols = [col for col in fundamental_cols if col in df.columns]
        if existing_fund_cols:
            df[existing_fund_cols] = df.groupby('Symbol')[existing_fund_cols].ffill()

    # Fill remaining missing values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

    print("✓ Missing values handled")
    return df


def create_target_variable(df, lookahead_days=5):
    """Create target variable - future returns"""
    print("Creating target variable...")

    if df.empty:
        return df

    df = df.sort_values(['Symbol', 'date'])
    df['future_close'] = df.groupby('Symbol')['Close'].shift(-lookahead_days)
    df['target_return'] = (df['future_close'] / df['Close'] - 1) * 100

    # Remove rows where target is NaN (end of time series)
    initial_count = len(df)
    df = df.dropna(subset=['target_return'])
    removed_count = initial_count - len(df)

    print(f"✓ Target variable created. Removed {removed_count} rows with missing targets")
    return df


def split_temporal_data(df, test_size, val_size):
    """Split data temporally for time series"""
    print("Splitting data into train/validation/test sets...")

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Sort by date first
    df = df.sort_values('date')
    dates = df['date'].unique()

    n_test = int(len(dates) * test_size)
    n_val = int(len(dates) * val_size)

    test_dates = dates[-n_test:]
    val_dates = dates[-(n_test + n_val):-n_test]
    train_dates = dates[:-(n_test + n_val)]

    train_data = df[df['date'].isin(train_dates)]
    val_data = df[df['date'].isin(val_dates)]
    test_data = df[df['date'].isin(test_dates)]

    print(f"✓ Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data


def create_final_dataset(base_folder, test_size=0.2, val_size=0.1):
    """Create the final dataset for TFT training"""
    print("Starting data preparation pipeline...")

    # Process all data sources
    news_sentiment = process_news_data(f"{base_folder}/News/market_news.csv")
    technical_data = process_technical_data(f"{base_folder}/Technical/")
    fundamental_data = process_fundamental_data(f"{base_folder}/Fundamentals/")

    if technical_data.empty or fundamental_data.empty:
        print("Error: Technical or Fundamental data is empty. Check your CSV files.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Merge technical and fundamental data
    print("\nMerging data sources...")
    merged_data = pd.merge(
        technical_data,
        fundamental_data,
        on=['date', 'Symbol'],
        how='left',
        suffixes=('_tech', '_fund')
    )

    print(f"Merged data shape: {merged_data.shape}")

    # Merge with news sentiment if available
    if not news_sentiment.empty:
        final_data = pd.merge(
            merged_data,
            news_sentiment,
            on='date',
            how='left'
        )
        print(f"Final data with news: {final_data.shape}")
    else:
        final_data = merged_data
        print("News sentiment data not available, proceeding without it")

    # Handle missing values and remove empty columns
    final_data = handle_missing_values(final_data)

    # Create target variable
    final_data = create_target_variable(final_data)

    # Remove any remaining empty columns
    final_data = remove_empty_columns(final_data, threshold=0.5)

    # Split into train/validation/test sets
    train_data, val_data, test_data = split_temporal_data(final_data, test_size, val_size)

    return train_data, val_data, test_data, final_data


def main():
    """Main execution function"""
    print("STOCK PREDICTION DATA PREPARATION")
    print("=" * 60)

    base_folder = "RAW_CSV"
    output_folder = create_output_folder()

    # Check if folder exists
    if not os.path.exists(base_folder):
        print(f"Error: Folder '{base_folder}' not found!")
        print("Please make sure the folder exists in the same directory as this script")
        return

    # Check subfolders
    subfolders = ['Fundamentals', 'Technical', 'News']
    for folder in subfolders:
        folder_path = f"{base_folder}/{folder}"
        if not os.path.exists(folder_path):
            print(f"Error: Subfolder '{folder}' not found in {base_folder}!")
            return
        else:
            files = list(Path(folder_path).glob("*.csv"))
            print(f"Found {len(files)} files in {folder}")

    print(f"Using data from: {base_folder}")
    print(f"Output will be saved to: {output_folder}")
    print("This process may take time due to data processing...")

    try:
        start_time = time.time()

        train_data, val_data, test_data, full_data = create_final_dataset(
            base_folder,
            test_size=0.2,
            val_size=0.1
        )

        end_time = time.time()

        print("\n" + "=" * 50)
        print("DATA PREPARATION COMPLETED!")
        print("=" * 50)

        if not train_data.empty:
            print(f"Training data shape: {train_data.shape}")
            print(f"Validation data shape: {val_data.shape}")
            print(f"Test data shape: {test_data.shape}")
            print(f"Full data shape: {full_data.shape}")

            # Show remaining columns
            print(f"\nRemaining columns ({len(full_data.columns)}):")
            for i, col in enumerate(full_data.columns, 1):
                print(f"{i:2d}. {col}")

            # Save processed data to output folder
            train_data.to_csv(f"{output_folder}/train_data.csv", index=False)
            val_data.to_csv(f"{output_folder}/val_data.csv", index=False)
            test_data.to_csv(f"{output_folder}/test_data.csv", index=False)
            full_data.to_csv(f"{output_folder}/full_processed_data.csv", index=False)

            # Save column information
            with open(f"{output_folder}/column_info.txt", 'w') as f:
                f.write("COLUMNS IN PROCESSED DATASET\n")
                f.write("=" * 50 + "\n")
                for i, col in enumerate(full_data.columns, 1):
                    f.write(f"{i:2d}. {col}\n")

            print(f"✓ CSV files saved successfully to {output_folder}/")
            print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")

            # Show sample of the final data
            print("\nSample of final data:")
            sample_cols = ['date', 'Symbol', 'Close', 'target_return']
            if 'avg_sentiment' in full_data.columns:
                sample_cols.append('avg_sentiment')
            print(full_data[sample_cols].head())

        else:
            print("No data was processed. Please check your input files.")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    print("\nScript execution completed.")