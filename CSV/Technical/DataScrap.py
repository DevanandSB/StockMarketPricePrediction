import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
import time
import os


def get_nifty50_companies():
    """Get the list of Nifty 50 companies from NSE website"""
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"

    try:
        # Download the CSV file
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Read the CSV data
        df = pd.read_csv(pd.compat.StringIO(response.text))

        # Extract company symbols (NSE symbols with .NS suffix for yfinance)
        symbols = df['Symbol'].tolist()
        company_names = df['Company Name'].tolist()

        print(f"Found {len(symbols)} Nifty 50 companies")
        return symbols, company_names

    except Exception as e:
        print(f"Error downloading Nifty 50 list: {e}")
        # Fallback: Hardcoded Nifty 50 symbols
        fallback_symbols = ['BEL', 'M&M', 'SUNPHARMA', 'ETERNAL', 'TITAN', 'TRENT', 'MARUTI', 'APOLLOHOSP', 'BAJFINANCE', 'CIPLA', 'JIOFIN', 'INDUSINDBK', 'DRREDDY', 'WIPRO', 'INFY', 'BHARTIARTL', 'HINDALCO', 'LT', 'ONGC', 'TATACONSUM', 'EICHERMOT', 'SBIN', 'BAJAJ-AUTO', 'NTPC', 'SHRIRAMFIN', 'BAJAJFINSV', 'AXISBANK', 'HINDUNILVR', 'SBILIFE', 'TECHM', 'HDFCLIFE', 'RELIANCE', 'TATAMOTORS', 'ICICIBANK', 'POWERGRID', 'TATASTEEL', 'COALINDIA', 'TCS', 'NESTLEIND', 'HDFCBANK', 'KOTAKBANK', 'ADANIPORTS', 'HCLTECH', 'ITC', 'ADANIENT', 'ULTRACEMCO', 'JSWSTEEL', 'HEROMOTOCO', 'ASIANPAINT', 'GRASIM']
        print("Using fallback Nifty 50 symbols")
        return fallback_symbols, fallback_symbols


def download_stock_data(symbol, start_date, end_date):
    """Download historical stock data for a single symbol"""
    try:
        # Add .NS suffix for NSE stocks in yfinance
        yf_symbol = symbol + '.NS'

        # Download data using yfinance
        stock = yf.Ticker(yf_symbol)
        data = stock.history(start=start_date, end=end_date)

        if not data.empty:
            # Add symbol as a column
            data['Symbol'] = symbol
            return data
        else:
            print(f"✗ No data found for {symbol}")
            return None

    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return None


def main():
    # Date range
    start_date = "2015-01-01"
    end_date = "2025-08-21"

    print("Starting Nifty 50 data download...")
    print(f"Date range: {start_date} to {end_date}")
    print("Files will be saved in current directory\n")

    # Get Nifty 50 companies
    symbols, company_names = get_nifty50_companies()

    if not symbols:
        print("Failed to get Nifty 50 companies list")
        return

    success_count = 0

    for i, symbol in enumerate(symbols, 1):
        try:
            print(f"[{i:2d}/50] Downloading {symbol}...", end=" ", flush=True)

            # Download stock data
            data = download_stock_data(symbol, start_date, end_date)

            if data is not None:
                # Save to CSV in current directory
                filename = f"{symbol}_historical.csv"
                data.to_csv(filename)
                success_count += 1
                print(f"✓ Saved {filename} ({len(data)} records)")
            else:
                print(f"✗ Failed")

            # Add delay to avoid rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"✗ Error processing {symbol}: {e}")

    print(f"\n{'=' * 50}")
    print(f"Download completed!")
    print(f"Successfully downloaded: {success_count} out of 50 companies")
    print(f"CSV files are saved in current directory: {os.getcwd()}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()