import requests
import csv
from datetime import datetime
import os

# 定义股票代码和日期范围
ticker_symbols = ["TSLA", "AAPL", "GOOGL", "MSFT", "NVDA", "META", "AMD"]
start_date = "2010-06-29"  # 特斯拉 IPO 日期
end_date = datetime.now().strftime("%Y-%m-%d")

# 替换为您的 Marketstack API 密钥
API_KEY = "5e102f567fd93bc60b0a666189b70d38" # 请替换为您的 Marketstack API 密钥

# 定义数据存储文件夹
DATA_DIR = "stock_data"

# 创建数据存储文件夹（如果不存在）
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_stock_data(symbol, api_key, start_date, end_date):
    print(f"正在获取 {symbol} 的股票数据...")
    all_data = []
    offset = 0
    while True:
        url = f"http://api.marketstack.com/v1/eod?access_key={api_key}&symbols={symbol}&date_from={start_date}&date_to={end_date}&limit=1000&offset={offset}"
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
        data = response.json()

        if data and 'data' in data and data['data']:
            for item in data['data']:
                all_data.append({
                    'Date': item['date'].split('T')[0],
                    'Open': item['open'],
                    'High': item['high'],
                    'Low': item['low'],
                    'Close': item['close'],
                    'Volume': item['volume']
                })
            
            # 检查是否还有更多数据
            if data['pagination']['count'] < data['pagination']['limit']:
                break  # 没有更多数据了
            offset += data['pagination']['count']
        else:
            print(f"未获取到 {symbol} 的数据，请检查股票代码、日期范围或 API 密钥")
            break
    return all_data

# 获取并保存所有股票数据
for symbol in ticker_symbols:
    try:
        stock_data = get_stock_data(symbol, API_KEY, start_date, end_date)
        if stock_data:
            csv_file_path = os.path.join(DATA_DIR, f'{symbol.lower()}_stock_data.csv')
            with open(csv_file_path, 'w', newline='') as f:
                fieldnames = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(stock_data)
            print(f"{symbol} 数据已成功保存到 {csv_file_path}")
        else:
            print(f"未能保存 {symbol} 的数据。")
    except requests.exceptions.RequestException as e:
        print(f"请求 Marketstack API 时发生错误: {e}")
    except Exception as e:
        print(f"处理 {symbol} 数据时发生错误: {e}")

print("所有股票数据获取完成。")