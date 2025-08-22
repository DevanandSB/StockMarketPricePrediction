import os
import time
import warnings
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')
from typing import Dict


class HistoricalFundamentalData:
    def __init__(self):
        self.symbols = [
            'BEL', 'M&M', 'SUNPHARMA', 'ETERNAL', 'TITAN', 'TRENT', 'MARUTI',
            'APOLLOHOSP', 'BAJFINANCE', 'CIPLA', 'JIOFIN', 'INDUSINDBK', 'DRREDDY',
            'WIPRO', 'INFY', 'BHARTIARTL', 'HINDALCO', 'LT', 'ONGC', 'TATACONSUM',
            'EICHERMOT', 'SBIN', 'BAJAJ-AUTO', 'NTPC', 'SHRIRAMFIN', 'BAJAJFINSV',
            'AXISBANK', 'HINDUNILVR', 'SBILIFE', 'TECHM', 'HDFCLIFE', 'RELIANCE',
            'TATAMOTORS', 'ICICIBANK', 'POWERGRID', 'TATASTEEL', 'COALINDIA', 'TCS',
            'NESTLEIND', 'HDFCBANK', 'KOTAKBANK', 'ADANIPORTS', 'HCLTECH', 'ITC',
            'ADANIENT', 'ULTRACEMCO', 'JSWSTEEL', 'HEROMOTOCO', 'ASIANPAINT', 'GRASIM'
        ]

        self.required_columns = [
            'Date', 'Symbol', 'Market Cap', 'Current Price', 'High', 'Low',
            'Stock P/E', 'Book Value', 'Dividend Yield', 'ROCE', 'ROE', 'Face Value',
            'EPS', 'Industry PE', 'Debt to equity', 'PEG Ratio', 'Reserves', 'OPM',
            'Promoter holding', 'Price to book value', 'Sales growth', 'Profit growth',
            'Dividend Payout', 'Debt', 'Sector', 'Volume', 'Free Cash Flow'
        ]

        self.start_date = "2015-01-01"
        self.end_date = "2025-08-22"

    def get_yahoo_historical_data(self, symbol: str) -> pd.DataFrame:
        """Get historical price data from Yahoo Finance"""
        try:
            yf_symbol = symbol + '.NS'
            data = yf.download(yf_symbol, start=self.start_date, end=self.end_date)
            if not data.empty:
                data = data.reset_index()
                data['Symbol'] = symbol
                data.rename(columns={'Close': 'Current Price', 'High': 'High',
                                     'Low': 'Low', 'Volume': 'Volume'}, inplace=True)
                return data[['Date', 'Symbol', 'Current Price', 'High', 'Low', 'Volume']]
        except:
            pass
        return pd.DataFrame()

    def get_nse_historical_data(self, symbol: str) -> pd.DataFrame:
        """Get historical data from NSE"""
        try:
            url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Process NSE data format
                df = pd.DataFrame(data['data'])
                df['Date'] = pd.to_datetime(df['TIMESTAMP'])
                df['Symbol'] = symbol
                df.rename(columns={'CH_CLOSING_PRICE': 'Current Price',
                                   'CH_TRADE_HIGH_PRICE': 'High',
                                   'CH_TRADE_LOW_PRICE': 'Low',
                                   'CH_TOT_TRADED_QTY': 'Volume'}, inplace=True)
                return df[['Date', 'Symbol', 'Current Price', 'High', 'Low', 'Volume']]
        except:
            pass
        return pd.DataFrame()

    def get_screener_fundamentals(self, symbol: str) -> Dict:
        """Get current fundamentals from Screener.in"""
        try:
            url = f"https://www.screener.in/company/{symbol}/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')

            fundamentals = {}

            # Extract ratios
            ratios_section = soup.find('section', id='ratios')
            if ratios_section:
                items = ratios_section.find_all('li', class_='flex flex-space-between')
                for item in items:
                    name = item.find('span', class_='name').text.strip()
                    value = item.find('span', class_='number').text.strip()
                    fundamentals[name] = value

            # Extract sector
            sector_info = soup.find('div', class_='company-profile')
            if sector_info:
                sector = sector_info.find('p', class_='sub')
                if sector:
                    fundamentals['Sector'] = sector.text.strip()

            return fundamentals
        except:
            return {}

    def map_fundamentals(self, raw_data: Dict) -> Dict:
        """Map raw fundamental data to required columns"""
        mapping = {
            'Market Cap': ['Market Cap', 'Mkt Cap'],
            'Stock P/E': ['P/E', 'Stock P/E', 'PE Ratio'],
            'Book Value': ['Book Value', 'BV'],
            'Dividend Yield': ['Dividend Yield', 'Div Yield'],
            'ROCE': ['ROCE', 'Return on Capital Employed'],
            'ROE': ['ROE', 'Return on Equity'],
            'Face Value': ['Face Value', 'FV'],
            'EPS': ['EPS', 'Earnings Per Share'],
            'Industry PE': ['Industry PE', 'Industry P/E'],
            'Debt to equity': ['Debt to equity', 'D/E Ratio'],
            'PEG Ratio': ['PEG Ratio', 'PEG'],
            'Reserves': ['Reserves', 'Total Reserves'],
            'OPM': ['OPM', 'Operating Profit Margin'],
            'Promoter holding': ['Promoter holding', 'Promoter Holding'],
            'Price to book value': ['Price to book value', 'P/B', 'PB Ratio'],
            'Dividend Payout': ['Dividend Payout', 'Payout Ratio'],
            'Debt': ['Debt', 'Total Debt', 'Borrowings'],
            'Sector': ['Sector', 'Industry'],
        }

        result = {}
        for target, sources in mapping.items():
            for source in sources:
                if source in raw_data:
                    result[target] = self.clean_value(raw_data[source])
                    break

        return result

    def clean_value(self, value):
        """Clean and convert values"""
        if not value or value in ['-', '—']:
            return None

        # Remove commas, percentage signs, etc.
        value = str(value).replace(',', '').replace('%', '').replace('₹', '').strip()

        # Handle Cr/Lac
        if 'Cr' in value:
            value = value.replace(' Cr', '')
            try:
                return float(value) * 10000000
            except:
                return None
        elif 'Lac' in value:
            value = value.replace(' Lac', '')
            try:
                return float(value) * 100000
            except:
                return None

        try:
            return float(value)
        except:
            return value

    def calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics like growth rates"""
        if len(df) < 2:
            return df

        # Calculate price-based growth metrics
        df['Price growth'] = df['Current Price'].pct_change() * 100

        # Simple moving averages for smoothing
        for window in [30, 90, 180]:
            df[f'MA_{window}'] = df['Current Price'].rolling(window=window).mean()

        return df

    def get_yahoo_fundamentals(self, symbol: str) -> Dict:
        """Get fundamentals from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol + '.NS')
            info = ticker.info

            return {
                'Market Cap': info.get('marketCap'),
                'Stock P/E': info.get('trailingPE'),
                'Book Value': info.get('bookValue'),
                'Dividend Yield': info.get('dividendYield'),
                'ROE': info.get('returnOnEquity'),
                'EPS': info.get('trailingEps'),
                'PEG Ratio': info.get('pegRatio'),
                'Debt to equity': info.get('debtToEquity'),
                'Price to book value': info.get('priceToBook'),
                'Sector': info.get('sector')
            }
        except:
            return {}

    def generate_historical_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Generate historical fundamental data by combining multiple sources"""
        print(f"Processing {symbol}...")

        # Get historical price data
        price_data = self.get_yahoo_historical_data(symbol)
        if price_data.empty:
            price_data = self.get_nse_historical_data(symbol)

        if price_data.empty:
            print(f"No price data found for {symbol}")
            return pd.DataFrame()

        # Get current fundamentals from multiple sources
        screener_data = self.get_screener_fundamentals(symbol)
        yahoo_data = self.get_yahoo_fundamentals(symbol)

        # Combine and map fundamentals
        all_fundamentals = {**screener_data, **yahoo_data}
        mapped_fundamentals = self.map_fundamentals(all_fundamentals)

        # Add fundamentals to each row (assuming they change slowly)
        for col, value in mapped_fundamentals.items():
            if col not in price_data.columns:
                price_data[col] = value

        # Calculate derived metrics
        price_data = self.calculate_derived_metrics(price_data)

        # Fill forward fundamental data (assuming they change quarterly)
        fundamental_cols = ['Market Cap', 'Stock P/E', 'Book Value', 'ROCE', 'ROE',
                            'EPS', 'Debt to equity', 'OPM', 'Promoter holding', 'Debt']

        for col in fundamental_cols:
            if col in price_data.columns:
                price_data[col] = price_data[col].ffill()

        # Ensure all required columns are present
        for col in self.required_columns:
            if col not in price_data.columns:
                price_data[col] = None

        return price_data[self.required_columns]

    def download_all_companies(self):
        """Download historical data for all companies"""
        os.makedirs("historical_fundamentals", exist_ok=True)

        success_count = 0
        all_summary = []

        for i, symbol in enumerate(self.symbols, 1):
            try:
                df = self.generate_historical_fundamentals(symbol)

                if not df.empty:
                    # Save individual company data
                    filename = f"historical_fundamentals/{symbol}_historical_fundamentals_2015_2025.csv"
                    df.to_csv(filename, index=False)

                    # Create summary
                    summary = {
                        'Symbol': symbol,
                        'Records': len(df),
                        'Start Date': df['Date'].min().strftime('%Y-%m-%d'),
                        'End Date': df['Date'].max().strftime('%Y-%m-%d'),
                        'Data Quality': f"{df.notna().mean().mean():.1%}"
                    }
                    all_summary.append(summary)

                    success_count += 1
                    print(f"✓ [{i:2d}/{len(self.symbols)}] Saved {symbol} - {len(df)} records")
                else:
                    print(f"✗ [{i:2d}/{len(self.symbols)}] Failed {symbol}")

                time.sleep(2)  # Rate limiting

            except Exception as e:
                print(f"✗ [{i:2d}/{len(self.symbols)}] Error processing {symbol}: {e}")
                time.sleep(1)

        # Save summary
        summary_df = pd.DataFrame(all_summary)
        summary_df.to_csv("historical_fundamentals/download_summary.csv", index=False)

        return success_count, summary_df


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("HISTORICAL FUNDAMENTAL DATA DOWNLOADER (2015-2025)")
    print("=" * 80)
    print("Downloading historical fundamental data for 50 companies...")
    print("Data sources: Yahoo Finance + NSE + Screener.in")
    print(f"Date range: {HistoricalFundamentalData().start_date} to {HistoricalFundamentalData().end_date}")
    print("=" * 80)

    downloader = HistoricalFundamentalData()
    success_count, summary = downloader.download_all_companies()

    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Successful downloads: {success_count}/{len(downloader.symbols)}")
    print(f"Data saved in: historical_fundamentals/ directory")
    print(f"Each company has a CSV file with daily historical data")

    if not summary.empty:
        print(f"\nAverage records per company: {summary['Records'].mean():.0f}")
        print(f"Average data quality: {summary['Data Quality'].mean():.1%}")

    print("=" * 80)