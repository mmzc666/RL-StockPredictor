# market_sentiment_analyzer.py

import os
import json
import pandas as pd
from snownlp import SnowNLP
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 定义数据存储文件夹
SENTIMENT_NEWS_DATA_DIR = "sentiment_news_data"

def analyze_sentiment_from_file(file_path: str) -> pd.DataFrame:
    """
    从JSON文件读取新闻数据，进行情感分析，并返回带有情感分数的数据。
    情感分数范围为0-1，越接近1表示越积极。
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return pd.DataFrame()

    with open(file_path, 'r', encoding='utf-8') as f:
        news_data = json.load(f)

    if not news_data:
        print(f"文件 {file_path} 中没有新闻数据。")
        return pd.DataFrame()

    news_df = pd.DataFrame(news_data)

    if 'title' not in news_df.columns or 'summary' not in news_df.columns:
        print("新闻数据中缺少 'title' 或 'summary' 字段，无法进行情感分析。")
        return news_df

    sentiment_scores = []
    for index, row in news_df.iterrows():
        text_to_analyze = f"{row['title']}. {row['summary']}"
        s = SnowNLP(text_to_analyze)
        sentiment_scores.append(s.sentiments)
    news_df['Sentiment_Score'] = sentiment_scores
    return news_df

def save_analyzed_sentiment_data(news_df: pd.DataFrame, output_file_path: str):
    """
    将情感分析后的新闻数据保存为CSV文件。
    """
    if news_df.empty:
        print("没有情感分析数据可保存。")
        return

    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    news_df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"情感分析数据已保存到 {output_file_path}")

def analyze_sentiment_with_finbert(text: str, tokenizer, model) -> float:
    """
    使用FinBERT模型对文本进行情感分析。
    返回一个介于0和1之间的分数，其中接近1表示积极，接近0表示消极。
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # FinBERT通常输出负面、中性和积极的logits
    # 我们需要将其转换为一个单一的积极/消极分数
    # 假设输出顺序是 [negative, neutral, positive]
    logits = outputs.logits[0]
    probabilities = torch.softmax(logits, dim=0)
    
    # 简单地将积极概率作为情感分数，或者可以加权计算
    # 例如：positive_score - negative_score + 0.5 (归一化到0-1)
    # 这里我们直接使用积极的概率作为分数
    positive_score = probabilities[2].item() # 假设积极情感在索引2
    negative_score = probabilities[0].item() # 假设消极情感在索引0
    neutral_score = probabilities[1].item() # 假设中性情感在索引1

    # 我们可以将积极中和性情感加权，或者直接使用积极情感
    # 这里我们尝试一个更精细的计算，将分数归一化到0-1
    # 积极分数越高，越接近1；消极分数越高，越接近0
    # (positive_score - negative_score + 1) / 2 可以将范围从 [-1, 1] 映射到 [0, 1]
    # 但FinBERT的输出是概率，所以直接使用positive_score可能更直观
    # 考虑到金融新闻的特点，中性情感也很重要，可以尝试 (positive_score + neutral_score * 0.5)
    # 这里我们使用 (positive_score - negative_score + 1) / 2 的变体，确保中性情感的影响
    sentiment_score = (positive_score - negative_score + 1) / 2
    return sentiment_score

def analyze_sentiment_from_file_finbert(file_path: str, tokenizer, model) -> pd.DataFrame:
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return pd.DataFrame()

    with open(file_path, 'r', encoding='utf-8') as f:
        news_data = json.load(f)

    if not news_data:
        print(f"文件 {file_path} 中没有新闻数据。")
        return pd.DataFrame()

    news_df = pd.DataFrame(news_data)

    if 'title' not in news_df.columns or 'summary' not in news_df.columns:
        print("新闻数据中缺少 'title' 或 'summary' 字段，无法进行情感分析。")
        return news_df

    sentiment_scores = []
    for index, row in news_df.iterrows():
        text_to_analyze = f"{row['title']}. {row['summary']}"
        score = analyze_sentiment_with_finbert(text_to_analyze, tokenizer, model)
        sentiment_scores.append(score)
    news_df['Sentiment_Score'] = sentiment_scores
    return news_df

if __name__ == "__main__":
    # 加载FinBERT模型和分词器
    print("正在加载FinBERT模型...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    print("FinBERT模型加载完成。")

    # 定义新闻数据和情感分析结果的根目录
    NEWS_DATA_ROOT_DIR = "sentiment_news_data"
    # 新建一个文件夹来存储情感分析结果
    ANALYZED_SENTIMENT_OUTPUT_DIR = "analyzed_sentiment_results"

    # 确保情感分析结果的根目录存在
    if not os.path.exists(ANALYZED_SENTIMENT_OUTPUT_DIR):
        os.makedirs(ANALYZED_SENTIMENT_OUTPUT_DIR)

    # 遍历 NEWS_DATA_ROOT_DIR 下的所有公司文件夹
    for company_folder in os.listdir(NEWS_DATA_ROOT_DIR):
        company_folder_path = os.path.join(NEWS_DATA_ROOT_DIR, company_folder)
        if os.path.isdir(company_folder_path):
            # 构建新闻数据文件路径
            # 假设新闻数据文件名为 {公司名}_news.json
            company_news_file = os.path.join(company_folder_path, f"{company_folder.upper()}_news.json")

            # 构建情感分析结果的输出目录和文件路径
            # 在新的 ANALYZED_SENTIMENT_OUTPUT_DIR 下为每个公司创建子文件夹
            company_output_dir = os.path.join(ANALYZED_SENTIMENT_OUTPUT_DIR, company_folder)
            if not os.path.exists(company_output_dir):
                os.makedirs(company_output_dir)
            output_analyzed_file = os.path.join(company_output_dir, f"{company_folder.upper()}_news_sentiment.csv")

            print(f"\n正在处理 {company_folder.upper()} 的新闻数据...")
            analyzed_news_df = analyze_sentiment_from_file_finbert(company_news_file, tokenizer, model)
            if not analyzed_news_df.empty:
                print(f"{company_folder.upper()} 情感分析结果（部分）：")
                print(analyzed_news_df.head())
                save_analyzed_sentiment_data(analyzed_news_df, output_analyzed_file)
            else:
                print(f"未能对 {company_folder.upper()} 的新闻进行情感分析。")

    # 示例用法：
    # 假设 market_sentiment_news_collector.py 已经生成了新闻数据文件
    # 例如：tsla_news_20230101_20230107.json
    # 修改为使用实际的TSLA新闻文件
    # sample_news_file = os.path.join(NEWS_DATA_ROOT_DIR, 'TSLA_news.json')
    # output_analyzed_file = os.path.join(NEWS_DATA_ROOT_DIR, 'TSLA_news_sentiment.csv')

    # 为了运行示例，这里创建一个假的示例新闻文件
    # 注释掉或删除创建示例数据的代码
    # if not os.path.exists(SENTIMENT_NEWS_DATA_DIR):
    #     os.makedirs(SENTIMENT_NEWS_DATA_DIR)
    
    # dummy_news_data = [
    #     {'Date': '2023-01-01', 'Title': '特斯拉股价大涨', 'Description': '特斯拉发布了新的电动汽车型号，市场反应积极。', 'URL': 'http://example.com/1'},
    #     {'Date': '2023-01-02', 'Title': '电动汽车市场竞争加剧', 'Description': '多家传统汽车制造商宣布进军电动汽车领域。', 'URL': 'http://example.com/2'},
    #     {'Date': '2023-01-03', 'Title': '特斯拉工厂扩建', 'Description': '特斯拉宣布将在德国建设新的超级工厂，以满足欧洲市场需求。', 'URL': 'http://example.com/3'}
    # ]
    # with open(sample_news_file, 'w', encoding='utf-8') as f:
    #     json.dump(dummy_news_data, f, indent=4)
    # print(f"已创建示例新闻文件: {sample_news_file}")

    # analyzed_news_df = analyze_sentiment_from_file_finbert(sample_news_file, tokenizer, model)
    # if not analyzed_news_df.empty:
    #     print("情感分析结果：")
    #     print(analyzed_news_df.head())
    #     save_analyzed_sentiment_data(analyzed_news_df, output_analyzed_file)
    # else:
    #     print("未能进行情感分析。")

    # 清理示例文件
    # os.remove(sample_news_file)
    # print(f"已删除示例新闻文件: {sample_news_file}")