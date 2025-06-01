import requests
import os
import requests
from datetime import datetime, timedelta

# 定义股票代码
ticker_symbols = ["TSLA", "AAPL", "GOOGL", "MSFT", "NVDA", "META", "AMD"]
# ticker_symbols = ["MSFT"]

# 定义数据存储文件夹
FINANCIAL_DATA_DIR = "financial_data"

# 创建数据存储文件夹（如果不存在）
if not os.path.exists(FINANCIAL_DATA_DIR):
    os.makedirs(FINANCIAL_DATA_DIR)

# 请替换为您的 Finnhub API 密钥
# 您可以在这里获取 API 密钥: https://finnhub.io/dashboard
FINNHUB_API_KEY = "D0thjk9r01qlvahc6ll0d0thjk9r01qlvahc6llg"

def get_finnhub_financials_reported(symbol, api_key, freq):
    """
    使用 Finnhub API 获取公司的财务报告数据。
    freq 可以是 'annual' 或 'quarterly'
    """
    print(f"正在获取 {symbol} 的 Finnhub {freq} 财务报告数据...")
    url = f"https://finnhub.io/api/v1/stock/financials-reported?symbol={symbol}&freq={freq}&token={api_key}"
    response = requests.get(url)
    response.raise_for_status() # 检查请求是否成功
    data = response.json()
    return data

def save_financial_data(symbol, data, filename_suffix):
    """
    将财报数据保存为 JSON 文件。
    """
    if data:
        company_dir = os.path.join(FINANCIAL_DATA_DIR, symbol.lower())
        os.makedirs(company_dir, exist_ok=True)
        file_path = os.path.join(company_dir, f'{symbol.lower()}_{filename_suffix}.json')
        with open(file_path, 'w') as f:
            import json
            json.dump(data, f, indent=4)
        print(f"{symbol} 的 {filename_suffix} 数据已成功保存到 {file_path}")
    else:
        print(f"未获取到 {symbol} 的 {filename_suffix} 数据。")

if __name__ == "__main__":
    if FINNHUB_API_KEY == "YOUR_FINNHUB_API_KEY":
        print("请在 financial_data_collector.py 文件中设置您的 Finnhub API 密钥。")
    else:
        for symbol in ticker_symbols:
            try:
                # 获取年度财报
                annual_financials = get_finnhub_financials_reported(symbol, FINNHUB_API_KEY, 'annual')
                save_financial_data(symbol, annual_financials, 'finnhub_annual_financials')

                # 获取季度财报
                quarterly_financials = get_finnhub_financials_reported(symbol, FINNHUB_API_KEY, 'quarterly')
                save_financial_data(symbol, quarterly_financials, 'finnhub_quarterly_financials')

            except requests.exceptions.RequestException as e:
                print(f"请求 Finnhub API 时发生错误: {e}")
            except Exception as e:
                print(f"处理 {symbol} 财报数据时发生错误: {e}")

    print("所有 Finnhub 财报数据获取完成。")