import pandas as pd
import requests
from bs4 import BeautifulSoup
import feedparser
import datetime
import time
import random
from textblob import TextBlob
import re
import json
from datetime import timedelta
import xml.etree.ElementTree as ET
import warnings

warnings.filterwarnings('ignore')


class HistoricalIndianNewsScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
        }
        self.session = requests.Session()

    def get_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        if not text:
            return 'Neutral'

        try:
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity

            if polarity > 0.1:
                return 'Positive'
            elif polarity < -0.1:
                return 'Negative'
            else:
                return 'Neutral'
        except:
            return 'Neutral'

    def clean_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def generate_synthetic_historical_data(self, start_date="2015-01-01", end_date="2025-08-22"):
        """Generate synthetic historical market news data"""
        news_items = []

        # Create a date range from start to end
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Market events that might influence news generation
        market_events = {
            "2016-11-08": "Demonetization announced by Indian government",
            "2017-07-01": "GST implementation in India",
            "2020-03-25": "COVID-19 lockdown announced in India",
            "2021-01-16": "COVID-19 vaccination drive begins in India",
            "2022-02-24": "Russia-Ukraine conflict impacts global markets",
            "2023-01-01": "New year market optimism",
        }

        # Sample news templates
        news_templates = [
            "Sensex {} points, Nifty {} as {}",
            "Stock market {}: {} leads to {} in indices",
            "Market {}: {} sector shows {} performance",
            "Indian shares {} amid {} developments",
            "{}: Market reacts to {} with {} trend",
            "BSE, NSE {}: {} influences trading",
            "Rupee {} against dollar as markets {}",
            "{} companies report {} earnings, impacting {}",
            "RBI policy decision: {} rates, markets {}",
            "Global cues lead to {} in Indian markets: {}"
        ]

        # Market movements
        movements = ["gains", "falls", "rises", "declines", "jumps", "dips", "surges", "plunges"]
        directions = ["positive", "negative", "mixed", "volatile", "steady"]
        points_range = [50, 100, 200, 300, 400, 500, 600, 700, 800]

        # News sources
        sources = ["Economic Times", "MoneyControl", "Business Standard", "Live Mint", "NDTV Profit",
                   "Reuters", "Bloomberg", "CNBC TV18", "ET Now", "Financial Express"]

        # News categories
        categories = [
            "earnings reports", "economic data", "global markets", "political developments",
            "monetary policy", "corporate announcements", "sector performance", "commodity prices",
            "currency movements", "foreign investments", "regulatory changes", "budget announcements"
        ]

        print(f"Generating synthetic historical data from {start_date} to {end_date}...")

        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')

            # Determine if it's a trading day (skip weekends)
            if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                continue

            # Generate 2-5 news items per day
            num_news = random.randint(2, 5)

            for i in range(num_news):
                # Check if there's a significant event on this date
                event_text = ""
                if date_str in market_events:
                    event_text = market_events[date_str]

                # Select random template and fill it
                template = random.choice(news_templates)

                # Generate random values for the template
                movement = random.choice(movements)
                points = random.choice(points_range)
                direction = random.choice(directions)
                category = random.choice(categories)
                source = random.choice(sources)

                # Fill the template
                if "{}" in template:
                    if template.count("{}") == 2:
                        title = template.format(movement, points, category)
                    elif template.count("{}") == 3:
                        title = template.format(movement, category, direction)
                    else:
                        title = template.format(movement, category)
                else:
                    title = template

                # Add event text if available
                if event_text:
                    title = f"{title} - {event_text}"

                # Generate description
                description = f"On {date_str}, Indian stock markets showed {direction} movement. {title}. "
                description += f"The {random.choice(['BSE Sensex', 'NSE Nifty'])} showed {movement} of {points} points. "
                description += f"This was primarily driven by {category}."

                # Add more context for significant events
                if event_text:
                    description += f" Market participants reacted to {event_text.lower()}."

                news_items.append({
                    'Date': date_str,
                    'News Title': title,
                    'News Description': description,
                    'Source': source,
                    'Sentiment': self.get_sentiment(f"{title} {description}"),
                    'URL': f"https://example.com/news/{date_str.replace('-', '')}/{i}",
                    'Type': 'Synthetic Historical'
                })

        return news_items

    def scrape_available_historical_data(self):
        """Scrape whatever historical data is available from public sources"""
        news_items = []

        # Try to get some real historical data from archive.org or other sources
        # This is a placeholder for actual historical scraping implementation

        return news_items

    def combine_with_real_time_data(self, historical_data):
        """Combine historical data with real-time current data"""
        # Create a scraper for current data
        current_scraper = HybridIndianNewsScraper()

        # Get current news
        print("Fetching current news...")
        current_news = current_scraper.scrape_rss_feeds()
        current_news.extend(current_scraper.scrape_nse_announcements())
        current_news.extend(current_scraper.scrape_bse_announcements())

        # Add to historical data
        historical_data.extend(current_news)

        return historical_data


def main():
    scraper = HistoricalIndianNewsScraper()

    print("=" * 60)
    print("INDIAN STOCK MARKET HISTORICAL NEWS DATASET GENERATOR")
    print("=" * 60)
    print("Generating data from 1st Jan 2015 to 22nd Aug 2025")
    print("=" * 60)

    # Generate synthetic historical data
    historical_data = scraper.generate_synthetic_historical_data(
        start_date="2015-01-01",
        end_date="2025-08-22"
    )

    # Try to get some real historical data if available
    real_historical_data = scraper.scrape_available_historical_data()
    historical_data.extend(real_historical_data)

    # Add current real-time data
    historical_data = scraper.combine_with_real_time_data(historical_data)

    # Create DataFrame
    df = pd.DataFrame(historical_data)

    # Remove duplicates
    df = df.drop_duplicates(subset=['Date', 'News Title'], keep='first')

    # Sort by date
    df = df.sort_values('Date', ascending=False)

    if df.empty:
        print("\n❌ No articles were collected.")
        return

    # Save to CSV
    filename = f"indian_stock_market_news_historical_2015_2025.csv"
    df.to_csv(filename, index=False, encoding='utf-8')

    print("\n" + "=" * 60)
    print("📊 HISTORICAL DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"📁 File saved: {filename}")
    print(f"📈 Total articles: {len(df)}")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"🎯 Sources: {df['Source'].nunique()} unique sources")
    print(f"😊 Sentiment distribution:")
    print(df['Sentiment'].value_counts())

    print("\n💡 Note: This dataset contains synthetic historical data")
    print("   combined with real-time current news for completeness.")


# Add the original HybridIndianNewsScraper class here
class HybridIndianNewsScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
        }
        self.session = requests.Session()

    def get_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        if not text:
            return 'Neutral'

        try:
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity

            if polarity > 0.1:
                return 'Positive'
            elif polarity < -0.1:
                return 'Negative'
            else:
                return 'Neutral'
        except:
            return 'Neutral'

    def clean_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def scrape_rss_feeds(self):
        """Scrape news from RSS feeds (most reliable)"""
        news_items = []
        rss_feeds = [
            {
                'name': 'Economic Times Markets',
                'url': 'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
                'source': 'Economic Times'
            },
            {
                'name': 'MoneyControl Markets',
                'url': 'https://www.moneycontrol.com/rss/MCmarketnews.xml',
                'source': 'MoneyControl'
            },
            {
                'name': 'Business Standard Markets',
                'url': 'https://www.business-standard.com/rss/markets-101.rss',
                'source': 'Business Standard'
            },
            {
                'name': 'Live Mint Markets',
                'url': 'https://www.livemint.com/rss/market',
                'source': 'Live Mint'
            },
            {
                'name': 'NDTV Profit',
                'url': 'https://feeds.feedburner.com/ndtvprofit-latest',
                'source': 'NDTV Profit'
            }
        ]

        for feed in rss_feeds:
            try:
                print(f"Fetching RSS feed: {feed['name']}")
                feed_data = feedparser.parse(feed['url'])

                for entry in feed_data.entries[:15]:  # Limit to 15 entries per feed
                    try:
                        title = self.clean_text(entry.title)
                        description = self.clean_text(entry.get('description', ''))
                        pub_date = entry.get('published', '')

                        # Try to parse date
                        try:
                            date_obj = datetime.datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
                            formatted_date = date_obj.strftime('%Y-%m-%d')
                        except:
                            formatted_date = datetime.datetime.now().strftime('%Y-%m-%d')

                        news_items.append({
                            'Date': formatted_date,
                            'News Title': title,
                            'News Description': description,
                            'Source': feed['source'],
                            'Sentiment': self.get_sentiment(f"{title} {description}"),
                            'URL': entry.get('link', ''),
                            'Type': 'RSS'
                        })
                    except Exception as e:
                        continue

                time.sleep(1)

            except Exception as e:
                print(f"Error parsing RSS feed {feed['name']}: {e}")
                continue

        return news_items

    def scrape_nse_announcements(self):
        """Scrape NSE announcements"""
        news_items = []
        try:
            print("Fetching NSE announcements...")
            url = "https://www.nseindia.com/api/corporate-announcements"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Referer': 'https://www.nseindia.com/company-listing/corporate-filings-announcements'
            }

            response = self.session.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('data', [])[:20]:
                    title = self.clean_text(item.get('subject', ''))
                    description = self.clean_text(item.get('attachmentText', ''))

                    news_items.append({
                        'Date': item.get('dateTime', datetime.datetime.now().strftime('%Y-%m-%d')),
                        'News Title': title,
                        'News Description': description,
                        'Source': 'NSE India',
                        'Sentiment': self.get_sentiment(title),
                        'URL': f"https://www.nseindia.com{item.get('url', '')}",
                        'Type': 'NSE Announcement'
                    })

        except Exception as e:
            print(f"NSE announcements error: {e}")

        return news_items

    def scrape_bse_announcements(self):
        """Scrape BSE announcements"""
        news_items = []
        try:
            print("Fetching BSE announcements...")
            url = "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w?strCat=-1&strPrevDate=&strScrip=&strSearch=P&strToDate=&strType=C"

            response = self.session.get(url, headers=self.headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('Table', [])[:15]:
                    title = self.clean_text(item.get('NEWS_SUBJECT', ''))
                    description = self.clean_text(item.get('NEWS_DTLS', ''))

                    news_items.append({
                        'Date': item.get('DT_TM', datetime.datetime.now().strftime('%Y-%m-%d')),
                        'News Title': title,
                        'News Description': description,
                        'Source': 'BSE India',
                        'Sentiment': self.get_sentiment(title),
                        'URL': 'https://www.bseindia.com/corporates/ann.html',
                        'Type': 'BSE Announcement'
                    })

        except Exception as e:
            print(f"BSE announcements error: {e}")

        return news_items

    def scrape_alternative_web_sources(self):
        """Scrape from alternative web sources"""
        news_items = []

        # Source 1: Investing.com India News
        try:
            print("Scraping Investing.com...")
            url = "https://www.investing.com/news/india-news"
            response = self.session.get(url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')

            articles = soup.select('.largeTitle .articleItem')
            for article in articles[:10]:
                try:
                    title_elem = article.select_one('a.title')
                    if title_elem:
                        title = self.clean_text(title_elem.get_text())
                        link = "https://www.investing.com" + title_elem['href']

                        date_elem = article.select_one('.date')
                        date_str = date_elem.get_text() if date_elem else datetime.datetime.now().strftime('%Y-%m-%d')

                        news_items.append({
                            'Date': date_str,
                            'News Title': title,
                            'News Description': '',
                            'Source': 'Investing.com',
                            'Sentiment': self.get_sentiment(title),
                            'URL': link,
                            'Type': 'Web Scrape'
                        })
                except:
                    continue

            time.sleep(2)
        except Exception as e:
            print(f"Investing.com error: {e}")

        # Source 2: Reuters India Business
        try:
            print("Scraping Reuters India...")
            url = "https://www.reuters.com/news/archive/india-business"
            response = self.session.get(url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')

            articles = soup.select('.story-content')
            for article in articles[:10]:
                try:
                    title_elem = article.select_one('a')
                    if title_elem:
                        title = self.clean_text(title_elem.get_text())
                        link = "https://www.reuters.com" + title_elem['href']

                        news_items.append({
                            'Date': datetime.datetime.now().strftime('%Y-%m-%d'),
                            'News Title': title,
                            'News Description': '',
                            'Source': 'Reuters',
                            'Sentiment': self.get_sentiment(title),
                            'URL': link,
                            'Type': 'Web Scrape'
                        })
                except:
                    continue

            time.sleep(2)
        except Exception as e:
            print(f"Reuters error: {e}")

        return news_items

    def use_newsapi_fallback(self, api_key=None):
        """Fallback to NewsAPI if available"""
        news_items = []

        if not api_key or api_key == "YOUR_API_KEY":
            return news_items

        try:
            print("Using NewsAPI fallback...")
            url = f"https://newsapi.org/v2/everything?q=indian+stock+market+NSE+BSE&language=en&sortBy=publishedAt&apiKey={api_key}"

            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for article in data.get('articles', [])[:20]:
                    title = self.clean_text(article.get('title', ''))
                    description = self.clean_text(article.get('description', ''))

                    news_items.append({
                        'Date': article.get('publishedAt', '')[:10],
                        'News Title': title,
                        'News Description': description,
                        'Source': article.get('source', {}).get('name', 'NewsAPI'),
                        'Sentiment': self.get_sentiment(f"{title} {description}"),
                        'URL': article.get('url', ''),
                        'Type': 'NewsAPI'
                    })

        except Exception as e:
            print(f"NewsAPI error: {e}")

        return news_items

    def generate_historical_dataset(self, days_back=30):
        """Generate dataset for recent historical period"""
        all_news = []

        print(f"Generating dataset for last {days_back} days...")

        # Get RSS feeds (current news)
        rss_news = self.scrape_rss_feeds()
        all_news.extend(rss_news)
        print(f"RSS feeds: {len(rss_news)} articles")

        # Get exchange announcements
        nse_news = self.scrape_nse_announcements()
        all_news.extend(nse_news)
        print(f"NSE announcements: {len(nse_news)} articles")

        bse_news = self.scrape_bse_announcements()
        all_news.extend(bse_news)
        print(f"BSE announcements: {len(bse_news)} articles")

        # Get web sources
        web_news = self.scrape_alternative_web_sources()
        all_news.extend(web_news)
        print(f"Web sources: {len(web_news)} articles")

        # Remove duplicates based on title
        df = pd.DataFrame(all_news)
        if not df.empty:
            df = df.drop_duplicates(subset=['News Title'], keep='first')
            df = df.sort_values('Date', ascending=False)

            # Filter for recent news
            recent_date = (datetime.datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            df = df[df['Date'] >= recent_date]

        return df

if __name__ == "__main__":
    main()
