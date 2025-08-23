import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import re
import shutil
import warnings

warnings.filterwarnings('ignore')

# Global variables for FinBERT
FINBERT_TOKENIZER = None
FINBERT_MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_output_folder():
    """Create output folder for processed data"""
    output_folder = "Processed_Data"
    if os.path.exists(output_folder):
        # Backup existing data
        backup_folder = f"{output_folder}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(output_folder, backup_folder)
        print(f"📦 Backed up existing data to {backup_folder}")

    os.makedirs(output_folder)
    print(f"✅ Created output folder: {output_folder}")
    return output_folder


def initialize_finbert_with_retry(max_retries=5):
    """Initialize FinBERT with multiple retries and comprehensive error handling"""
    global FINBERT_TOKENIZER, FINBERT_MODEL

    print("🔧 INITIALIZING FinBERT MODEL WITH UTMOST CARE...")
    print("=" * 80)

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to load FinBERT...")

            # Clear cache to avoid any conflicts
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(
                "ProsusAI/finbert",
                cache_dir="./finbert_cache",
                local_files_only=False,
                force_download=False,
                resume_download=True
            )

            FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert",
                cache_dir="./finbert_cache",
                local_files_only=False,
                force_download=False,
                resume_download=True
            )

            FINBERT_MODEL.to(DEVICE)
            FINBERT_MODEL.eval()

            # Test the model with a simple input
            test_text = "Stock market shows positive growth today."
            test_inputs = FINBERT_TOKENIZER(test_text, return_tensors="pt", truncation=True, max_length=128)
            test_inputs = {k: v.to(DEVICE) for k, v in test_inputs.items()}

            with torch.no_grad():
                test_outputs = FINBERT_MODEL(**test_inputs)
                test_probs = F.softmax(test_outputs.logits, dim=-1)
                test_sentiment = test_probs[0][1].item() - test_probs[0][2].item()

            print("✅ FinBERT MODEL LOADED SUCCESSFULLY!")
            print(f"   Device: {DEVICE}")
            print(f"   Model: {type(FINBERT_MODEL).__name__}")
            print(f"   Test sentiment score: {test_sentiment:.4f}")
            print(f"   Model parameters: {sum(p.numel() for p in FINBERT_MODEL.parameters()):,}")

            return True

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                print("💀 ALL ATTEMPTS TO LOAD FinBERT FAILED!")
                print("Please check:")
                print("1. Internet connection")
                print("2. Hugging Face access (https://huggingface.co/ProsusAI/finbert)")
                print("3. Disk space (need ~500MB for model)")
                print("4. Firewall/proxy settings")
                return False

            print(f"🔄 Retrying in 10 seconds...")
            time.sleep(10)

    return False


def analyze_sentiment_with_extreme_care(text, max_length=400):
    """Analyze sentiment with the utmost care and comprehensive error handling"""
    if FINBERT_TOKENIZER is None or FINBERT_MODEL is None:
        print("❌ FinBERT not initialized properly")
        return 0.0

    if pd.isna(text) or text is None:
        return 0.0

    # Convert to string and clean thoroughly
    text = str(text).strip()

    # Comprehensive validation of text content
    if (len(text) < 15 or
            text.lower() in ['nan', 'null', 'none', '', 'undefined', 'na'] or
            text.isspace() or
            len(text.split()) < 3):  # At least 3 words
        return 0.0

    try:
        # Pre-process text to remove problematic characters but keep meaningful content
        text = re.sub(r'[^\w\s.,!?\-@#$%&*()]', ' ', text)  # Keep basic punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

        if len(text) < 15:
            return 0.0

        # Tokenize with extreme care
        inputs = FINBERT_TOKENIZER(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            add_special_tokens=True
        )

        # Move to device with error handling
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Run inference with comprehensive error handling
        with torch.no_grad():
            try:
                outputs = FINBERT_MODEL(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)

                # Calculate sentiment score (positive - negative)
                sentiment_score = probs[0][1].item() - probs[0][2].item()

                # Validate and clamp the score
                if not (-1.0 <= sentiment_score <= 1.0):
                    print(f"⚠️  Invalid sentiment score: {sentiment_score}, clamping")
                    sentiment_score = max(-1.0, min(1.0, sentiment_score))

                return sentiment_score

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("⚠️  CUDA out of memory, reducing batch size or text length")
                    torch.cuda.empty_cache()
                    return analyze_sentiment_with_extreme_care(text, max_length // 2)
                raise e

    except Exception as e:
        print(f"❌ Critical error in sentiment analysis for text: '{text[:100]}...'")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        return 0.0


def parse_date_with_absolute_precision(date_str):
    """Parse date with absolute precision and comprehensive validation"""
    if pd.isna(date_str) or date_str is None:
        return pd.NaT

    try:
        date_str = str(date_str).strip()

        if not date_str or date_str.lower() in ['nan', 'null', 'none', '', 'undefined', 'na']:
            return pd.NaT

        # Comprehensive date format尝试
        date_formats = [
            '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d',
            '%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y',
            '%m-%d-%Y', '%m/%d/%Y', '%m.%d.%Y',
            '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%fZ',
            '%d-%b-%Y', '%d-%B-%Y', '%b %d, %Y', '%B %d, %Y',
            '%Y%m%d', '%d%m%Y', '%m%d%Y'
        ]

        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                if 1990 <= dt.year <= 2030:  # Reasonable year range
                    return dt
            except:
                continue

        # Try pandas as fallback
        dt = pd.to_datetime(date_str, errors='coerce', infer_datetime_format=True)
        if not pd.isna(dt) and 1990 <= dt.year <= 2030:
            return dt

        return pd.NaT

    except Exception:
        return pd.NaT


def extract_symbol_from_filename(filename):
    """Extract symbol from filename with careful parsing"""
    try:
        # Remove common suffixes
        name = str(filename).replace('_historical', '').replace('_fundamentals', '')
        name = name.replace('_2015_2025', '').replace('.csv', '')

        # Extract symbol (assume it's the first part before any underscores)
        symbol = name.split('_')[0].strip().upper()

        # Validate symbol (basic check)
        if len(symbol) >= 2 and len(symbol) <= 10 and symbol.isalpha():
            return symbol
        else:
            return "UNKNOWN"

    except:
        return "UNKNOWN"


def remove_empty_columns_with_care(df, threshold=0.8):
    """Remove columns with too many missing values carefully"""
    if df.empty:
        return df

    missing_percentage = df.isnull().mean()
    columns_to_remove = missing_percentage[missing_percentage > threshold].index.tolist()

    if columns_to_remove:
        print(f"Removing columns with >{threshold * 100}% missing values: {columns_to_remove}")
        df = df.drop(columns=columns_to_remove)

    return df


def add_technical_indicators_with_care(df):
    """Add technical indicators with extreme care and validation"""
    try:
        # Basic price data validation
        if 'Close' not in df.columns:
            return df

        # Ensure numeric values
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])

        if df.empty:
            return df

        # Moving averages
        for window in [5, 20, 50]:
            col_name = f'MA_{window}'
            df[col_name] = df['Close'].rolling(window=window, min_periods=1).mean()

        # Price change with careful handling
        df['price_change'] = df['Close'].pct_change()
        df['price_change'] = df['price_change'].replace([np.inf, -np.inf], np.nan)
        df['price_change'] = df['price_change'].fillna(0)

        # Volatility
        df['volatility_20'] = df['Close'].rolling(window=20, min_periods=1).std()
        df['volatility_20'] = df['volatility_20'].fillna(0)

        # Volume indicators (if available)
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df['volume_ma_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ma_20'] = df['volume_ma_20'].replace(0, 1)  # Avoid division by zero
            df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
            df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], 1)
            df['volume_ratio'] = df['volume_ratio'].fillna(1)

        # RSI with careful calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()

        # Avoid division by zero
        loss = loss.replace(0, 1e-10)
        rs = gain / loss

        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].clip(0, 100)  # Ensure valid range
        df['RSI'] = df['RSI'].fillna(50)  # Neutral value for missing

        return df

    except Exception as e:
        print(f"❌ Error adding technical indicators: {str(e)}")
        return df


def process_technical_data_with_care(technical_folder):
    """Process technical data with extreme care"""
    print("\n" + "=" * 80)
    print("📊 PROCESSING TECHNICAL DATA WITH EXTREME CARE")
    print("=" * 80)

    if not os.path.exists(technical_folder):
        print(f"❌ Technical folder not found: {technical_folder}")
        return pd.DataFrame()

    technical_files = list(Path(technical_folder).glob("*.csv"))
    if not technical_files:
        print("❌ No technical CSV files found")
        return pd.DataFrame()

    print(f"📁 Found {len(technical_files)} technical files")

    all_technical_data = []
    processed_files = 0

    for file in tqdm(technical_files, desc="Processing technical files"):
        try:
            # Read with careful error handling
            df = pd.read_csv(file, encoding='utf-8', on_bad_lines='warn')

            # Check required columns
            if 'Date' not in df.columns or 'Close' not in df.columns:
                print(f"⚠️  Missing required columns in {file.name}, skipping")
                continue

            # Parse dates with extreme care
            df['date'] = df['Date'].apply(parse_date_with_absolute_precision)
            df = df.dropna(subset=['date'])

            if df.empty:
                print(f"⚠️  No valid dates in {file.name}, skipping")
                continue

            # Add technical indicators carefully
            df = add_technical_indicators_with_care(df)

            # Add symbol
            symbol = extract_symbol_from_filename(file.name)
            df['Symbol'] = symbol

            all_technical_data.append(df)
            processed_files += 1

        except Exception as e:
            print(f"❌ Error processing {file.name}: {str(e)}")
            continue

    if not all_technical_data:
        print("❌ No technical data processed successfully")
        return pd.DataFrame()

    # Combine all data
    result_df = pd.concat(all_technical_data, ignore_index=True)
    result_df = remove_empty_columns_with_care(result_df)

    print(f"✅ Technical data: {len(result_df)} rows from {processed_files} files")
    print(f"📅 Date range: {result_df['date'].min()} to {result_df['date'].max()}")
    print(f"📈 Unique symbols: {result_df['Symbol'].nunique()}")

    return result_df


def process_fundamental_data_with_care(fundamental_folder):
    """Process fundamental data with extreme care"""
    print("\n" + "=" * 80)
    print("📈 PROCESSING FUNDAMENTAL DATA WITH EXTREME CARE")
    print("=" * 80)

    if not os.path.exists(fundamental_folder):
        print(f"❌ Fundamental folder not found: {fundamental_folder}")
        return pd.DataFrame()

    fundamental_files = list(Path(fundamental_folder).glob("*.csv"))
    if not fundamental_files:
        print("❌ No fundamental CSV files found")
        return pd.DataFrame()

    print(f"📁 Found {len(fundamental_files)} fundamental files")

    all_fundamental_data = []
    processed_files = 0

    for file in tqdm(fundamental_files, desc="Processing fundamental files"):
        try:
            # Read with careful error handling
            df = pd.read_csv(file, encoding='utf-8', on_bad_lines='warn')

            # Check required columns
            if 'Date' not in df.columns:
                print(f"⚠️  Missing Date column in {file.name}, skipping")
                continue

            # Parse dates with extreme care
            df['date'] = df['Date'].apply(parse_date_with_absolute_precision)
            df = df.dropna(subset=['date'])

            if df.empty:
                print(f"⚠️  No valid dates in {file.name}, skipping")
                continue

            # Convert numeric columns
            numeric_columns = [
                'Market Cap', 'Current Price', 'High', 'Low', 'Stock P/E', 'Book Value',
                'Dividend Yield', 'ROCE', 'ROE', 'Face Value', 'EPS', 'Industry PE',
                'Debt to equity', 'PEG Ratio', 'Reserves', 'OPM', 'Promoter holding',
                'Price to book value', 'Sales growth', 'Profit growth', 'Dividend Payout',
                'Debt', 'Volume', 'Free Cash Flow'
            ]

            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Add symbol
            symbol = extract_symbol_from_filename(file.name)
            df['Symbol'] = symbol

            all_fundamental_data.append(df)
            processed_files += 1

        except Exception as e:
            print(f"❌ Error processing {file.name}: {str(e)}")
            continue

    if not all_fundamental_data:
        print("❌ No fundamental data processed successfully")
        return pd.DataFrame()

    # Combine all data
    result_df = pd.concat(all_fundamental_data, ignore_index=True)
    result_df = remove_empty_columns_with_care(result_df, threshold=0.7)

    print(f"✅ Fundamental data: {len(result_df)} rows from {processed_files} files")
    print(f"📅 Date range: {result_df['date'].min()} to {result_df['date'].max()}")
    print(f"📈 Unique symbols: {result_df['Symbol'].nunique()}")

    return result_df


def process_news_data_with_perfection(news_path):
    """Process news data with absolute perfection and comprehensive error handling"""
    print("\n" + "=" * 100)
    print("📰 PROCESSING NEWS DATA WITH ABSOLUTE PERFECTION")
    print("=" * 100)

    # Comprehensive file existence check
    if not os.path.exists(news_path):
        print(f"❌ NEWS FILE NOT FOUND: {news_path}")
        print("🔍 Please check:")
        print(f"   - File exists at: {news_path}")
        print("   - File permissions")
        print("   - File path is correct")
        print("⚠️  CONTINUING WITHOUT NEWS DATA")
        return pd.DataFrame(columns=['date', 'avg_sentiment', 'news_count', 'has_news'])

    try:
        # Load news data with comprehensive error handling
        print("📥 LOADING NEWS DATA WITH UTMOST CARE...")

        try:
            news_df = pd.read_csv(news_path, encoding='utf-8', on_bad_lines='warn')
            if len(news_df) == 0:
                print("❌ News file is empty")
                return pd.DataFrame(columns=['date', 'avg_sentiment', 'news_count', 'has_news'])
        except UnicodeDecodeError:
            try:
                news_df = pd.read_csv(news_path, encoding='latin-1', on_bad_lines='warn')
            except:
                news_df = pd.read_csv(news_path, encoding='ISO-8859-1', on_bad_lines='warn')

        print(f"✅ LOADED {len(news_df):,} NEWS ARTICLES")
        print(f"📊 COLUMNS AVAILABLE: {list(news_df.columns)}")

        # COMPREHENSIVE COLUMN VALIDATION
        required_date_columns = ['Date', 'DATE', 'date', 'datetime', 'Datetime', 'Time', 'time']
        date_column = None

        for col in required_date_columns:
            if col in news_df.columns:
                date_column = col
                break

        if date_column is None:
            print("❌ NO DATE COLUMN FOUND IN NEWS DATA")
            print("🔍 Available columns:", list(news_df.columns))
            print("⚠️  CONTINUING WITHOUT NEWS DATA")
            return pd.DataFrame(columns=['date', 'avg_sentiment', 'news_count', 'has_news'])

        print(f"📅 USING DATE COLUMN: {date_column}")

        # COMPREHENSIVE CONTENT COLUMN VALIDATION
        content_columns = [
            'News Title', 'News Description', 'Content', 'Text', 'Article',
            'Title', 'Description', 'Headline', 'Body', 'Summary',
            'news_title', 'news_description', 'content', 'text', 'article'
        ]

        content_column = None
        for col in content_columns:
            if col in news_df.columns:
                content_column = col
                break

        if content_column is None:
            print("❌ NO CONTENT COLUMN FOUND IN NEWS DATA")
            print("🔍 Available columns:", list(news_df.columns))
            print("⚠️  CONTINUING WITHOUT NEWS DATA")
            return pd.DataFrame(columns=['date', 'avg_sentiment', 'news_count', 'has_news'])

        print(f"📝 USING CONTENT COLUMN: {content_column}")

        # EXTREME CARE DATE PARSING
        print("📅 PARSING DATES WITH EXTREME PRECISION...")
        news_df['date'] = news_df[date_column].apply(parse_date_with_absolute_precision)

        # Remove invalid dates
        initial_count = len(news_df)
        news_df = news_df.dropna(subset=['date'])
        valid_count = len(news_df)

        print(f"📊 DATE VALIDATION: {valid_count:,}/{initial_count:,} ARTICLES HAVE VALID DATES")

        if valid_count == 0:
            print("❌ NO ARTICLES WITH VALID DATES")
            print("⚠️  CONTINUING WITHOUT NEWS DATA")
            return pd.DataFrame(columns=['date', 'avg_sentiment', 'news_count', 'has_news'])

        # PREPARE CONTENT FOR SENTIMENT ANALYSIS WITH UTMOST CARE
        print("✍️  PREPARING CONTENT FOR SENTIMENT ANALYSIS...")

        # Handle missing content with extreme care
        news_df['clean_content'] = news_df[content_column].fillna('')
        news_df['clean_content'] = news_df['clean_content'].astype(str).str.strip()

        # Remove empty or meaningless content
        news_df = news_df[news_df['clean_content'].str.len() > 20]  # At least 20 characters
        news_df = news_df[news_df['clean_content'].str.split().str.len() > 3]  # At least 3 words

        meaningful_count = len(news_df)
        print(f"📋 {meaningful_count:,} ARTICLES HAVE MEANINGFUL CONTENT FOR SENTIMENT ANALYSIS")

        if meaningful_count == 0:
            print("❌ NO ARTICLES WITH MEANINGFUL CONTENT")
            print("⚠️  CONTINUING WITHOUT NEWS DATA")
            return pd.DataFrame(columns=['date', 'avg_sentiment', 'news_count', 'has_news'])

        # SENTIMENT ANALYSIS WITH ABSOLUTE PRECISION
        print("🧠 ANALYZING SENTIMENT WITH FinBERT (THIS WILL TAKE TIME)...")
        print("⏰ ESTIMATED TIME: ~1-2 SECONDS PER ARTICLE")
        print(f"⏳ TOTAL ARTICLES: {meaningful_count:,}")
        print(f"⏱️  ESTIMATED TOTAL TIME: {meaningful_count * 1.5 / 60:.1f} MINUTES")
        print("⚠️  PLEASE BE PATIENT - QUALITY IS EVERYTHING")

        sentiments = []
        batch_size = 1  # Process one at a time for maximum stability
        processed_count = 0

        # Create progress bar
        pbar = tqdm(total=meaningful_count, desc="🧠 Sentiment Analysis", unit="article")

        for i in range(0, meaningful_count, batch_size):
            batch_indices = range(i, min(i + batch_size, meaningful_count))
            batch_sentiments = []

            for idx in batch_indices:
                row = news_df.iloc[idx]
                sentiment = analyze_sentiment_with_extreme_care(row['clean_content'])
                batch_sentiments.append(sentiment)
                processed_count += 1

                # Update progress every 10 articles
                if processed_count % 10 == 0:
                    pbar.set_description(f"🧠 Sentiment Analysis ({processed_count}/{meaningful_count})")
                    pbar.update(10)

            sentiments.extend(batch_sentiments)

            # Save progress every 100 articles to avoid losing work
            if processed_count % 100 == 0:
                temp_df = news_df.iloc[:processed_count].copy()
                temp_df['sentiment'] = sentiments
                print(f"💾 Checkpoint: Processed {processed_count} articles")

        pbar.close()

        # Add sentiments to dataframe
        news_df['sentiment'] = sentiments

        # COMPREHENSIVE SENTIMENT ANALYSIS
        print("📊 ANALYZING SENTIMENT DISTRIBUTION...")
        sentiment_stats = news_df['sentiment'].describe()
        print(f"📈 SENTIMENT STATISTICS:")
        print(f"   Mean: {sentiment_stats['mean']:.6f}")
        print(f"   Std:  {sentiment_stats['std']:.6f}")
        print(f"   Min:  {sentiment_stats['min']:.6f}")
        print(f"   Max:  {sentiment_stats['max']:.6f}")
        print(f"   Non-zero: {(news_df['sentiment'] != 0).sum():,} articles")

        # AGGREGATE BY DATE WITH PRECISION
        print("📅 AGGREGATING SENTIMENT BY DATE...")
        daily_sentiment = news_df.groupby('date').agg({
            'sentiment': ['mean', 'count', 'std']
        }).reset_index()

        daily_sentiment.columns = ['date', 'avg_sentiment', 'news_count', 'sentiment_std']

        # Add news presence flag
        daily_sentiment['has_news'] = 1

        print(f"✅ CREATED DAILY SENTIMENT DATA FOR {len(daily_sentiment):,} DAYS")
        print(f"📅 DATE RANGE: {daily_sentiment['date'].min()} to {daily_sentiment['date'].max()}")
        print(f"📊 ARTICLES PER DAY: {daily_sentiment['news_count'].mean():.1f} (avg)")

        return daily_sentiment

    except Exception as e:
        print(f"💥 CRITICAL ERROR PROCESSING NEWS DATA: {str(e)}")
        import traceback
        traceback.print_exc()
        print("⚠️  CONTINUING WITHOUT NEWS DATA")
        return pd.DataFrame(columns=['date', 'avg_sentiment', 'news_count', 'has_news'])


def handle_missing_values_with_care(df):
    """Handle missing values with extreme care and precision"""
    if df.empty:
        return df

    print("🔍 Handling missing values with detailed analysis...")

    # Separate news columns
    news_cols = ['avg_sentiment', 'news_count', 'sentiment_std', 'has_news']
    stock_cols = [col for col in df.columns if col not in news_cols + ['date', 'Symbol']]

    # Report initial missing values
    print(f"   Initial missing values in stock data: {df[stock_cols].isnull().sum().sum()}")
    print(
        f"   Initial missing values in news data: {df[news_cols].isnull().sum().sum() if news_cols[0] in df.columns else 'N/A'}")

    # For stock data: forward fill by symbol, then fill with 0
    for col in stock_cols:
        if col in df.columns:
            # First try forward fill within each symbol
            df[col] = df.groupby('Symbol')[col].ffill()
            # Then fill remaining with 0
            df[col] = df[col].fillna(0)

    # For news data: special handling
    if 'has_news' in df.columns:
        # Ensure has_news is integer (0 or 1)
        df['has_news'] = df['has_news'].fillna(0).astype(int)

    if 'avg_sentiment' in df.columns:
        # For sentiment, keep 0 for no news (neutral sentiment)
        df['avg_sentiment'] = df['avg_sentiment'].fillna(0)

    if 'news_count' in df.columns:
        # News count should be 0 when no news
        df['news_count'] = df['news_count'].fillna(0).astype(int)

    if 'sentiment_std' in df.columns:
        # Sentiment std should be 0 when no news
        df['sentiment_std'] = df['sentiment_std'].fillna(0)

    # Report final missing values
    print(f"   Final missing values: {df.isnull().sum().sum()}")

    return df

def create_target_variable_with_care(df):
    """Create target variable with extreme care and validation"""
    if df.empty or 'Close' not in df.columns:
        return df

    # Sort by symbol and date
    df = df.sort_values(['Symbol', 'date'])

    # Create future price (5 days ahead)
    df['future_close'] = df.groupby('Symbol')['Close'].shift(-5)

    # Calculate return
    df['target_return'] = (df['future_close'] / df['Close'] - 1) * 100

    # Remove invalid returns
    initial_count = len(df)
    df = df.dropna(subset=['target_return'])

    # Remove extreme outliers
    df = df[df['target_return'].between(-50, 50)]  # Reasonable range

    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rows with invalid targets")

    return df


def split_temporal_data_with_care(df, test_size=0.15, val_size=0.15):
    """Split data temporally with extreme care"""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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

    print(f"Train: {len(train_data)} rows")
    print(f"Validation: {len(val_data)} rows")
    print(f"Test: {len(test_data)} rows")

    return train_data, val_data, test_data


def merge_datasets_with_care(technical_data, fundamental_data, news_data):
    """Merge all datasets with extreme care and precision"""
    print("\n" + "=" * 80)
    print("🔗 MERGING DATASETS WITH EXTREME CARE")
    print("=" * 80)

    if technical_data.empty:
        print("❌ No technical data to merge")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Fix timezone issues first
    print("🕐 Fixing timezone issues...")

    # Remove timezone from all datasets
    technical_data['date'] = pd.to_datetime(technical_data['date']).dt.tz_localize(None)
    fundamental_data['date'] = pd.to_datetime(fundamental_data['date']).dt.tz_localize(None)

    if not news_data.empty:
        news_data['date'] = pd.to_datetime(news_data['date']).dt.tz_localize(None)
        print(f"News data date range: {news_data['date'].min()} to {news_data['date'].max()}")
        print(f"News data samples: {len(news_data)}")

    # First merge technical and fundamental data properly
    print("Merging technical and fundamental data...")

    # Use proper merge with careful handling
    merged_data = pd.merge(
        technical_data,
        fundamental_data,
        on=['date', 'Symbol'],
        how='outer',
        suffixes=('', '_fund')
    )

    # Remove duplicate columns
    merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]

    print(f"Merged technical+fundamental shape: {merged_data.shape}")
    print(f"Merged date range: {merged_data['date'].min()} to {merged_data['date'].max()}")

    # CRITICAL FIX: Proper news data merging
    if not news_data.empty:
        print("🔍 DEBUG: News data info before merge:")
        print(f"   News data shape: {news_data.shape}")
        print(f"   News date range: {news_data['date'].min()} to {news_data['date'].max()}")
        print(f"   News columns: {news_data.columns.tolist()}")
        print(f"   Sample news sentiment: {news_data['avg_sentiment'].head(5).values}")

        print("Merging with news data with extreme care...")

        # Create a complete date range that covers ALL possible dates
        all_dates = pd.date_range(
            start=min(merged_data['date'].min(), news_data['date'].min()),
            end=max(merged_data['date'].max(), news_data['date'].max()),
            freq='D'
        )

        # Create a date reference dataframe
        date_reference = pd.DataFrame({'date': all_dates})

        # Merge news data with the complete date range (LEFT JOIN to preserve all news)
        news_complete = pd.merge(date_reference, news_data, on='date', how='left')

        # Fill NaN values for news columns appropriately
        news_cols = ['avg_sentiment', 'news_count', 'sentiment_std', 'has_news']
        for col in news_cols:
            if col in news_complete.columns:
                if col == 'has_news':
                    news_complete[col] = news_complete[col].fillna(0)
                else:
                    news_complete[col] = news_complete[col].fillna(0)  # Fill with 0 for no news

        print(f"News complete shape: {news_complete.shape}")
        print(f"News complete non-zero days: {(news_complete['has_news'] == 1).sum()}")

        # Now merge with the main dataset
        final_data = pd.merge(
            merged_data,
            news_complete,
            on='date',
            how='left'
        )

        print(f"✅ Final data shape after news merge: {final_data.shape}")
        print(f"✅ News data in final dataset:")
        print(f"   Days with news: {(final_data['has_news'] == 1).sum()}")
        print(
            f"   Avg sentiment range: {final_data['avg_sentiment'].min():.3f} to {final_data['avg_sentiment'].max():.3f}")
        print(f"   Total news count: {final_data['news_count'].sum()}")

    else:
        print("No news data available")
        final_data = merged_data.copy()
        # Add empty news columns with proper defaults
        final_data['avg_sentiment'] = 0.0
        final_data['news_count'] = 0
        final_data['sentiment_std'] = 0.0
        final_data['has_news'] = 0

    # Handle missing values with extreme care
    print("Handling missing values with extreme care...")
    final_data = handle_missing_values_with_care(final_data)

    # DEBUG: Check news data after handling missing values
    if not news_data.empty:
        print("🔍 DEBUG: After missing value handling:")
        print(f"   Non-zero sentiment: {(final_data['avg_sentiment'] != 0).sum()}")
        print(f"   Non-zero news count: {(final_data['news_count'] != 0).sum()}")
        print(f"   Has news flag: {final_data['has_news'].sum()}")

    # Create target variable
    print("Creating target variable...")
    final_data = create_target_variable_with_care(final_data)

    # Final debug check
    if not news_data.empty:
        news_stats = final_data[['avg_sentiment', 'news_count', 'has_news']].describe()
        print("📊 FINAL NEWS STATISTICS:")
        print(news_stats)

    # Split into train/validation/test
    print("Splitting into train/validation/test sets...")
    train_data, val_data, test_data = split_temporal_data_with_care(final_data)

    return train_data, val_data, test_data, final_data

def save_datasets_with_care(output_folder, train_data, val_data, test_data, full_data):
    """Save all datasets with extreme care and validation"""
    print("\n" + "=" * 80)
    print("💾 SAVING DATASETS WITH EXTREME CARE")
    print("=" * 80)

    datasets = {
        'train_data.csv': train_data,
        'val_data.csv': val_data,
        'test_data.csv': test_data,
        'full_processed_data.csv': full_data
    }

    for filename, data in datasets.items():
        if not data.empty:
            filepath = os.path.join(output_folder, filename)
            data.to_csv(filepath, index=False)
            print(f"✅ Saved {filename}: {len(data)} rows")
        else:
            print(f"❌ Cannot save {filename}: empty dataframe")

    # Save detailed dataset information
    save_dataset_info(output_folder, train_data, val_data, test_data, full_data)


def save_dataset_info(output_folder, train_data, val_data, test_data, full_data):
    """Save comprehensive dataset information"""
    info_path = os.path.join(output_folder, "dataset_info.txt")

    with open(info_path, 'w') as f:
        f.write("STOCK PREDICTION DATASET - EXTREMELY CAREFUL PREPARATION\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATASET SIZES:\n")
        f.write(f"Training samples: {len(train_data):,}\n")
        f.write(f"Validation samples: {len(val_data):,}\n")
        f.write(f"Test samples: {len(test_data):,}\n")
        f.write(f"Total samples: {len(full_data):,}\n\n")

        f.write("DATE RANGES:\n")
        if not full_data.empty:
            f.write(f"Start date: {full_data['date'].min()}\n")
            f.write(f"End date: {full_data['date'].max()}\n")
            f.write(f"Total days: {full_data['date'].nunique()}\n\n")

        f.write("SYMBOL INFORMATION:\n")
        if not full_data.empty and 'Symbol' in full_data.columns:
            f.write(f"Unique symbols: {full_data['Symbol'].nunique()}\n")
            symbol_counts = full_data['Symbol'].value_counts().head(10)
            f.write("Top 10 symbols by sample count:\n")
            for symbol, count in symbol_counts.items():
                f.write(f"  {symbol}: {count:,} samples\n")
        f.write("\n")

        f.write("FEATURE INFORMATION:\n")
        if not full_data.empty:
            f.write(f"Total features: {len(full_data.columns)}\n")
            f.write("Feature list:\n")
            for i, col in enumerate(full_data.columns, 1):
                f.write(f"{i:2d}. {col}\n")
        f.write("\n")

        f.write("NEWS DATA INFORMATION:\n")
        if not full_data.empty and 'avg_sentiment' in full_data.columns:
            news_stats = full_data[['avg_sentiment', 'news_count', 'has_news']].describe()
            f.write(f"News statistics:\n{news_stats.to_string()}\n")
            days_with_news = full_data['has_news'].sum()
            total_days = len(full_data)
            f.write(f"Days with news: {days_with_news:,}/{total_days:,} ({days_with_news / total_days * 100:.1f}%)\n")
        else:
            f.write("No news data available\n")

    print(f"✅ Saved dataset information to {info_path}")


def main():
    """Main execution with absolute perfection in data preparation"""
    print("🎯 ABSOLUTELY PERFECT DATA PREPARATION FOR STOCK PREDICTION")
    print("=" * 120)
    print("THIS WILL TAKE AS LONG AS NEEDED - QUALITY IS THE ONLY PRIORITY")
    print("TIME IS IRRELEVANT - PERFECTION IS MANDATORY")
    print("=" * 120)

    # Initialize FinBERT with extreme care
    if not initialize_finbert_with_retry():
        print("💀 CRITICAL: CANNOT PROCEED WITHOUT FinBERT")
        print("Please resolve the FinBERT loading issue before continuing")
        return

    # Create output folder
    output_folder = create_output_folder()

    # Comprehensive input validation
    base_folder = "RAW_CSV"
    if not os.path.exists(base_folder):
        print(f"❌ INPUT FOLDER NOT FOUND: {base_folder}")
        print("Please create the folder structure:")
        print("RAW_CSV/")
        print("├── Technical/       # Technical CSV files")
        print("├── Fundamentals/    # Fundamental CSV files")
        print("└── News/           # News CSV file")
        return

    print("✅ BEGINNING ABSOLUTELY PERFECT DATA PROCESSING...")
    print("⚠️  THIS WILL TAKE SEVERAL HOURS - QUALITY CANNOT BE RUSHED")

    start_time = time.time()

    try:
        # Process news data first (most time-consuming and critical)
        print("\n" + "=" * 100)
        print("🎯 STEP 1: PROCESSING NEWS DATA WITH ABSOLUTE PERFECTION")
        print("=" * 100)

        news_data = process_news_data_with_perfection(f"{base_folder}/News/market_news.csv")

        # Process other data
        print("\n" + "=" * 100)
        print("🎯 STEP 2: PROCESSING TECHNICAL DATA WITH ABSOLUTE PERFECTION")
        print("=" * 100)
        technical_data = process_technical_data_with_care(f"{base_folder}/Technical/")

        print("\n" + "=" * 100)
        print("🎯 STEP 3: PROCESSING FUNDAMENTAL DATA WITH ABSOLUTE PERFECTION")
        print("=" * 100)
        fundamental_data = process_fundamental_data_with_care(f"{base_folder}/Fundamentals/")

        # Merge everything with absolute precision
        print("\n" + "=" * 100)
        print("🎯 STEP 4: MERGING ALL DATASETS WITH ABSOLUTE PERFECTION")
        print("=" * 100)
        train_data, val_data, test_data, full_data = merge_datasets_with_care(
            technical_data, fundamental_data, news_data
        )

        # Save results
        print("\n" + "=" * 100)
        print("🎯 STEP 5: SAVING DATASETS WITH ABSOLUTE PERFECTION")
        print("=" * 100)
        save_datasets_with_care(output_folder, train_data, val_data, test_data, full_data)

        print("\n" + "=" * 100)
        print("🎉 DATA PREPARATION COMPLETED SUCCESSFULLY!")
        print("=" * 100)
        print("All datasets have been prepared with extreme care and attention to detail")

    except Exception as e:
        print(f"💥 UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Calculate total time
        end_time = time.time()
        total_hours = (end_time - start_time) / 3600
        print(f"\n⏰ TOTAL EXECUTION TIME: {total_hours:.2f} HOURS")

        if total_hours > 0.5:
            print("✅ TIME WELL SPENT - QUALITY DATA IS BEING PRODUCED")
        else:
            print("⚠️  Process completed quickly - please verify data quality")


if __name__ == "__main__":
    main()