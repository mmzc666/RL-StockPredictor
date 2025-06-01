import os
import sys
import json
from datetime import datetime, timedelta
import finnhub


# Finnhub API Key (Replace with your actual key)
FINNHUB_API_KEY = "D0thjk9r01qlvahc6ll0d0thjk9r01qlvahc6llg" # You need to get your own API key from finnhub.io
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Configuration
STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMD", "META"]
NEWS_OUTLETS = {
    "AAPL": ["Finnhub"],
    "MSFT": ["Finnhub"],
    "GOOGL": ["Finnhub"],
    "NVDA": ["Finnhub"],
    "TSLA": ["Finnhub"],
    "AMD": ["Finnhub"],
    "META": ["Finnhub"]
}
DATA_DIR = "sentiment_news_data"

def get_historical_news_data(symbol: str, start_date: datetime, end_date: datetime):
    """
    Fetches historical news data for a given stock symbol using NewspaperScraper.
    """
    all_news = []
    for outlet in NEWS_OUTLETS.get(symbol, []):
        print(f"Fetching news for {symbol} from {outlet} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        if outlet == "Finnhub":
            try:
                # Finnhub API call
                # Finnhub API free tier limits historical data to 1 year per call.
                # The current implementation already sets start_date to one year ago, so this should be fine.
                finnhub_news = finnhub_client.company_news(symbol, _from=start_date.strftime('%Y-%m-%d'), to=end_date.strftime('%Y-%m-%d'))
                for news_item in finnhub_news:
                    all_news.append({
                        'title': news_item.get('headline'),
                        'date_published': datetime.fromtimestamp(news_item.get('datetime')).strftime('%Y-%m-%d %H:%M:%S') if news_item.get('datetime') else None,
                        'news_outlet': news_item.get('source'),
                        'article_link': news_item.get('url'),
                        'summary': news_item.get('summary'),
                        'text': news_item.get('summary') # Finnhub summary is often the full text
                    })
                print(f"Successfully fetched {len(finnhub_news)} articles from Finnhub for {symbol}")
            except Exception as e:
                print(f"Error fetching news from Finnhub for {symbol}: {e}")
        
    return all_news

def save_news_data(symbol: str, news_data: list):
    """
    Saves the fetched news data to a JSON file.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    file_path = os.path.join(DATA_DIR, f"{symbol}_news.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(news_data, f, ensure_ascii=False, indent=4)
    print(f"Saved news data for {symbol} to {file_path}")

if __name__ == "__main__":
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365) # Set start date to one year ago

    for symbol in STOCK_SYMBOLS:
        news = get_historical_news_data(symbol, start_date, end_date)
        if news:
            save_news_data(symbol, news)
        else:
            print(f"未获取到 {symbol} 的新闻数据。")