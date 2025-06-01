# 基于强化学习的股票预测系统项目概述

这个项目是一个使用强化学习技术构建的股票预测系统，它整合了多种数据源（股票历史数据、财务报告和市场情绪数据）来预测股票的未来走势。系统提供了两个版本的强化学习预测器，分别实现了不同的预测策略。

## 核心功能

- **多源数据整合**：结合股票OHLCV数据、公司财务报告和市场新闻情绪数据
- **强化学习环境**：基于gymnasium库构建自定义的股票交易环境
- **特征工程**：从原始数据中提取并构建适合RL模型的观察特征
- **PPO模型训练**：使用stable-baselines3库的PPO算法训练智能体
- **预测功能**：提供接口预测未来股票价格变化或OHLCV序列
- **数据预处理工具**：包含收集、清洗和处理各类数据的辅助脚本

## 项目结构

项目包含多个关键组件：

1. **数据收集模块**：
   - <mcfile name="stock_data_collector.py" path="/Users/apple/PycharmProjects/Stock/stock_data_collector.py"></mcfile>：收集股票历史数据
   - <mcfile name="financial_data_collector.py" path="/Users/apple/PycharmProjects/Stock/financial_data_collector.py"></mcfile>：收集公司财务报告数据
   - <mcfile name="market_sentiment_news_collector.py" path="/Users/apple/PycharmProjects/Stock/market_sentiment_news_collector.py"></mcfile>：收集市场新闻数据

2. **数据处理模块**：
   - <mcfile name="financial_data_processor.py" path="/Users/apple/PycharmProjects/Stock/financial_data_processor.py"></mcfile>：处理财务数据
   - <mcfile name="market_sentiment_analyzer.py" path="/Users/apple/PycharmProjects/Stock/market_sentiment_analyzer.py"></mcfile>：分析新闻情感
   - <mcfile name="data_cleaner.py" path="/Users/apple/PycharmProjects/Stock/data_cleaner.py"></mcfile>：清洗和预处理数据

3. **预测模型**：
   - <mcfile name="rl_stock_predictor.py" path="/Users/apple/PycharmProjects/Stock/rl_stock_predictor.py"></mcfile>：第一版预测器（预测百分比变化）
   - <mcfile name="rl_stock_predictor_v2.py" path="/Users/apple/PycharmProjects/Stock/rl_stock_predictor_v2.py"></mcfile>：第二版预测器（预测OHLCV序列）
   - <mcfile name="rl_stock_predictor_inference.py" path="/Users/apple/PycharmProjects/Stock/rl_stock_predictor_inference.py"></mcfile>：用于模型推理的脚本

4. **数据存储**：
   - 原始数据存储在相应目录<mcfolder name="stock_data/" path="/Users/apple/PycharmProjects/Stock/stock_data/"></mcfolder><mcfolder name="financial_data/" path="/Users/apple/PycharmProjects/Stock/financial_data/"></mcfolder>、<mcfolder name="sentiment_news_data/" path="/Users/apple/PycharmProjects/Stock/sentiment_news_data/"></mcfolder>
   - 处理后的数据存储在对应目录<mcfolder name="processed_financial_data/" path="/Users/apple/PycharmProjects/Stock/processed_financial_data/"></mcfolder><mcfolder name="analyzed_sentiment_results/" path="/Users/apple/PycharmProjects/Stock/analyzed_sentiment_results/"></mcfolder>
   - 预测结果存储在<mcfolder name="predictions_v2/" path="/Users/apple/PycharmProjects/Stock/predictions_v2/"></mcfolder>目录
   - 训练好的模型存储在<mcfolder name="trained_rl_models_v2/" path="/Users/apple/PycharmProjects/Stock/trained_rl_models_v2/"></mcfolder>目录

## 使用方法

本项目的使用流程主要分为数据准备、模型训练和模型预测三个阶段。

### 1. 数据获取

数据准备是整个预测系统的基础，涉及到股票历史数据、公司财务数据和市场情绪数据的收集、清洗和处理。以下是主要的数据准备脚本及其功能：

#### 1.1 股票历史数据

- **脚本**: `stock_data_collector.py`
- **功能**: 收集指定股票的历史 OHLCV（开盘价、最高价、最低价、收盘价、成交量）数据。该脚本通过 Marketstack API 获取股票的日线数据。
- **数据来源**: Marketstack API。
- **输出路径**: 原始股票数据将保存到 `stock_data/{公司代码}_stock_data.csv`。
- **使用方法**: 
  ```bash
  python stock_data_collector.py
  ```
  该脚本会自动为预设的公司（例如：AAPL, MSFT, GOOGL, NVDA, TSLA, AMD, META）收集股票历史数据。您可以根据需要在脚本中修改 `ticker_symbols` 列表和 `start_date`、`end_date` 来调整收集的公司和日期范围。
- **API密钥配置**: 需要配置 Marketstack API 密钥。请在 `stock_data_collector.py` 文件中找到 `API_KEY` 变量，并替换为您的实际密钥。

#### 1.2 公司财务数据

- **脚本**: `financial_data_collector.py`
- **功能**: 收集指定公司的年度和季度财务报告数据。该脚本通过 Finnhub API 获取公司的财务报表数据，包括年度和季度报告。
- **数据来源**: Finnhub API。
- **输出路径**: 原始财务数据将保存到 `financial_data/{公司代码}/{公司代码}_finnhub_annual_financials.json`（年度报告）和 `financial_data/{公司代码}/{公司代码}_finnhub_quarterly_financials.json`（季度报告）。
- **使用方法**: 
  ```bash
  python financial_data_collector.py
  ```
  该脚本会自动为预设的公司（例如：AAPL, MSFT, GOOGL, NVDA, TSLA, AMD, META）收集财务数据。您可以根据需要在脚本中修改 `ticker_symbols` 列表来调整收集的公司。
- **API密钥配置**: 需要配置 Finnhub API 密钥。请在 `financial_data_collector.py` 文件中找到 `FINNHUB_API_KEY` 变量，并替换为您的实际密钥。

#### 1.3 市场情绪新闻数据

- **脚本**: `market_sentiment_news_collector.py`
- **功能**: 收集指定公司的市场新闻数据，为情感分析做准备。该脚本通过 Finnhub API 获取公司新闻，并支持配置多个新闻来源（目前默认使用 Finnhub）。
- **数据来源**: Finnhub API。
- **输出路径**: 原始新闻数据将保存到 `sentiment_news_data/{公司代码}_news.json`。
- **使用方法**:
  ```bash
  python market_sentiment_news_collector.py
  ```
  该脚本会自动为预设的公司（例如：AAPL, MSFT, GOOGL, NVDA, TSLA, AMD, META）收集过去一年的新闻数据。您可以根据需要在脚本中修改 `STOCK_SYMBOLS` 列表和 `start_date`、`end_date` 的计算逻辑来调整收集的公司和日期范围。
- **API密钥配置**: 需要配置 Finnhub API 密钥。请在 `market_sentiment_news_collector.py` 文件中找到 `FINNHUB_API_KEY` 变量，并替换为您的实际密钥。

#### 1.4 市场情绪分析

- **脚本**: `market_sentiment_analyzer.py`
- **功能**: 对收集到的市场新闻数据进行情感分析，量化市场情绪。该脚本支持两种情感分析模型：
  - **SnowNLP**: 适用于中文文本的情感分析。
  - **FinBERT**: 适用于英文金融文本的情感分析（默认使用）。
  脚本会读取 `sentiment_news_data` 目录下每个公司的新闻数据（JSON格式），提取新闻标题和摘要进行情感分析，并将分析结果（情感分数）保存为CSV文件。
- **输入**: 原始新闻数据文件，位于 `sentiment_news_data/{公司代码}/{公司代码}_news.json`。
- **输出路径**: 情感分析结果将保存到 `analyzed_sentiment_results/{公司代码}/{公司代码}_news_sentiment.csv`。
- **使用方法**:
  ```bash
  python market_sentiment_analyzer.py
  ```
  运行此脚本，它将自动遍历 `sentiment_news_data` 目录下所有公司的新闻数据，使用FinBERT模型进行情感分析，并将结果保存到 `analyzed_sentiment_results` 目录下的相应公司文件夹中。


### 2. 数据清洗与整合

- **脚本**：
  - <mcfile name="financial_data_processor.py" path="financial_data_processor.py"></mcfile>
  - <mcfile name="data_cleaner.py" path="data_cleaner.py"></mcfile>
- **功能**：这些脚本协同工作，对原始数据进行清洗、格式化、特征提取和整合，生成可用于模型训练的结构化数据。
  - <mcfile name="financial_data_processor.py" path="financial_data_processor.py"></mcfile>：该脚本负责处理从Finnhub API收集到的原始财务数据（JSON格式），从中提取关键的年度和季度财务指标，如营收（revenue）、净利润（netIncome）、每股收益（eps）、总资产（totalAssets）、总负债（totalLiabilities）和股东权益（equity）。它会读取 <mcfolder name="financial_data/" path="financial_data/"></mcfolder> 目录下的JSON文件，并将其整理成结构化的CSV格式。
  - <mcfile name="data_cleaner.py" path="data_cleaner.py"></mcfile>：整合股票、财务和情感数据，处理缺失值，确保数据对齐和格式统一。这是数据准备的最后一步，它将不同来源的数据合并成一个统一的、时间序列对齐的数据集，为模型训练做好准备。
- **输出路径**：
  - 处理后的财务数据存储在 <mcfolder name="processed_financial_data/" path="processed_financial_data/"></mcfolder> 目录下，按公司代码分类（例如，`processed_financial_data/MSFT/msft_financial_metrics.csv`）。
  - 整合后的最终数据集将由 <mcfile name="data_cleaner.py" path="data_cleaner.py"></mcfile> 生成，具体输出路径请参考该脚本的实现。
- **使用方法**：
  1. 确保已完成股票、财务和新闻数据的收集，以及市场情绪分析。
  2. 运行处理脚本（建议按以下顺序执行，以确保数据依赖性）：
     - `python financial_data_processor.py`
     - `python data_cleaner.py`
  3. 脚本将按顺序处理数据并生成最终的训练数据集，这些数据集将用于后续的模型训练。

### 3. 模型训练

该部分主要包含以下脚本：

-   `rl_stock_predictor_v2.py`: 这是一个基于强化学习（Reinforcement Learning, RL）的股票预测模型，利用PPO（Proximal Policy Optimization）算法进行训练，旨在预测股票未来的价格走势。

#### 3.1 模型概述

`rl_stock_predictor_v2.py` 实现了基于强化学习的股票价格预测模型。该模型采用 **近端策略优化 (PPO)** 算法，通过与自定义的股票交易环境 `StockTradingEnvV2` 交互来学习最优的预测策略。模型的目标是预测未来一段时间内股票的开盘价、最高价、最低价、收盘价和交易量 (OHLCV) 数据。

#### 3.2 数据准备与特征工程

在模型训练之前，需要对收集到的原始数据进行整合和预处理。`rl_stock_predictor_v2.py` 中的 `load_and_preprocess_data_v2` 函数负责此项工作：

-   **数据加载**：
    -   从 `stock_data` 目录加载股票历史 OHLCV 数据（通过 `load_stock_data_v2` 函数）。
    -   从 `processed_financial_data` 目录加载已处理的财务指标数据（通过 `load_financial_data_v2` 函数）。
    -   从 `analyzed_sentiment_results` 目录加载市场情绪分析结果数据（通过 `load_sentiment_data_v2` 函数）。
-   **数据合并**：将股票数据、财务数据和情感数据按日期进行合并。财务数据和情感数据会向前填充 (forward fill)，以确保每个交易日都有对应的财务和情感信息。对于初始的缺失值，会用0填充。
-   **特征标准化**：所有用于模型训练的特征（OHLCV、情感分数、财务指标）都会被标准化，以确保模型训练的稳定性和效率。标准化器 (`MinMaxScaler`) 会在环境初始化时根据所有公司的历史数据进行拟合。
-   **特征维度**：最终的观察特征包括 OHLCV (5维)、情感分数 (1维) 和多个财务指标。这些特征共同构成了强化学习环境的观察空间。

#### 3.3 强化学习环境 (`StockTradingEnvV2`)

`StockTradingEnvV2` 是一个基于 `gymnasium` 库的自定义强化学习环境，用于模拟股票交易过程，为PPO智能体提供训练平台。环境的核心组成部分包括：

-   **观察空间 (Observation Space)**：
    -   定义为 `Box(low=-np.inf, high=np.inf, shape=(self.lookback_window * self.observation_feature_dim + 1,), dtype=np.float32)`。
    -   `lookback_window`：模型观察过去多少天的历史数据（默认为60天）。
    -   `observation_feature_dim`：每个时间步的特征维度，包括OHLCV、情感分数和财务指标。
    -   `+1`：额外的一个特征用于指示当前的预测周期类型（短期或长期）。
    -   观察数据经过 `MinMaxScaler` 进行标准化处理。

-   **动作空间 (Action Space)**：
    -   定义为 `Box(low=-1.0, high=1.0, shape=(self.max_pred_horizon, self.num_ohlcv_features), dtype=np.float32)`。
    -   智能体的动作是预测未来 `max_pred_horizon` 天（默认为180天，即最长预测周期）的标准化OHLCV序列。
    -   `num_ohlcv_features`：OHLCV特征的数量（5个）。

-   **奖励机制 (Reward Mechanism)**：
    -   奖励旨在鼓励智能体做出更准确的预测。
    -   **主要奖励**：基于预测的收盘价与实际收盘价之间的平均百分比误差 (`mean_error_pct`)。
        -   `mean_error_pct < 1%`：奖励 `100` 分（高精度）。
        -   `1% <= mean_error_pct < 5%`：奖励 `50` 分（中等精度）。
        -   `5% <= mean_error_pct < 10%`：奖励 `20` 分（低精度）。
        -   `mean_error_pct >= 10%`：施加惩罚，惩罚值为 `-( (mean_error_pct - 0.10) * 100 )`。
    -   **方向奖励**：额外奖励或惩罚基于预测的最后一天收盘价与观察期最后一天收盘价相比的涨跌方向是否与实际方向一致。
        -   方向一致：额外奖励 `15` 分。
        -   方向不一致：额外惩罚 `-15` 分。
    -   **惩罚**：
        -   数据不足以形成有效观察或目标序列：` -200` 或 ` -150`。
        -   模型或环境配置错误（如Scaler未拟合、动作形状不匹配）：` -250` 或 ` -1000`。

-   **重置 (`reset`)**：
    -   每个回合开始时，环境会随机选择一个公司和预测周期（短期7天或长期180天）。
    -   随机选择一个起始时间步，确保有足够的历史数据用于观察，并有足够的未来数据用于评估预测。
    -   返回初始观察和环境信息。

-   **步进 (`step`)**：
    -   接收智能体预测的标准化OHLCV序列作为动作。
    -   根据当前预测周期截取动作中相关部分。
    -   获取实际的未来OHLCV序列。
    -   计算奖励。
    -   时间步进1天。
    -   判断回合是否结束（如果数据不足以进行下一次观察或预测）。
    -   返回新的观察、奖励、回合是否结束的标志 (`terminated`, `truncated`) 和额外信息。

#### 3.4 模型训练 (`PPO` 算法)

-   **算法选择**：使用 `stable_baselines3` 库中的 `PPO` (Proximal Policy Optimization) 算法。
-   **网络架构**：策略网络和价值网络均采用多层感知机 (MLP)，每层包含128个神经元，共2层 (`pi=[128, 128], vf=[128, 128]`)。
-   **训练参数**：
    -   `learning_rate=0.0003`：学习率。
    -   `n_steps=2048`：每次更新收集的样本数。
    -   `batch_size=64`：每次梯度更新的mini-batch大小。
    -   `n_epochs=10`：每次策略更新的迭代次数。
    -   `gamma=0.99`：折扣因子，用于平衡即时奖励和未来奖励。
    -   `gae_lambda=0.95`：广义优势估计 (GAE) 参数。
    -   `clip_range=0.2`：PPO裁剪参数，用于限制策略更新幅度。
    -   `ent_coef=0.0`：熵系数，用于鼓励探索（此处设置为0，表示不鼓励）。
    -   `vf_coef=0.5`：价值函数损失系数。
    -   `max_grad_norm=0.5`：梯度裁剪的最大范数。
-   **训练过程**：
    -   数据加载和预处理：通过 `load_and_preprocess_data_v2` 函数加载所有公司的股票、财务和情感数据。
    -   环境创建：使用 `DummyVecEnv` 包装 `StockTradingEnvV2`，并用 `VecNormalize` 进行奖励归一化（`norm_reward=True`），观察值不进行额外归一化（`norm_obs=False`），因为环境内部已处理。
    -   模型加载/创建：如果存在预训练模型，则加载继续训练；否则，创建新的PPO模型。
    -   回调函数：使用 `CheckpointCallback` 每隔 `50000` 步保存一次模型检查点。
    -   训练步数：总训练步数为 `200000`。
    -   TensorBoard日志：训练过程中的指标会记录到 `ppo_stock_tensorboard_v2/` 目录，可用于可视化训练进度。
-   **模型保存**：训练完成后，最终模型 (`ppo_stock_predictor_v2.zip`)、观察和动作标准化器 (`scalers_v2.pkl`) 以及 `VecNormalize` 的统计信息 (`vec_normalize_v2.pkl`) 会保存到 `trained_rl_models_v2/` 目录下。

### 4. 模型推理与预测

`rl_stock_predictor_inference.py` 脚本中的 `predict_stock_ohlcv` 函数是专门用于模型推理预测的接口。它加载训练好的模型，并对新的数据进行未来OHLCV序列的预测。其主要步骤和使用方法如下：

#### 4.1 函数签名

```python
def predict_stock_ohlcv(
    company_symbol: str,
    prediction_horizon_key: str = 'short', # 'short' or 'long'
    output_csv_path: str = None
):
```

-   `company_symbol` (str): 股票代码，例如 `'AAPL'`, `'MSFT'`。
-   `prediction_horizon_key` (str): 预测周期，`'short'` 表示预测未来7天，`'long'` 表示预测未来30天。默认为 `'short'`。
-   `output_csv_path` (str, optional): 预测结果CSV文件的保存路径。如果为 `None`，则结果将打印到控制台。默认为 `None`。

#### 4.2 详细使用步骤

1.  **数据输入准备**：
    -   确保您需要预测的公司的最新股票历史数据、财务指标数据和市场情绪分析数据已放置在 `input_data/` 目录下。
    -   文件命名约定：
        -   股票数据：`{company_symbol.lower()}_stock_data.csv` (例如: `msft_stock_data.csv`)
        -   财务数据：`{company_symbol.lower()}_financial_metrics.csv` (例如: `msft_financial_metrics.csv`)
        -   情绪数据：`{company_symbol.upper()}_news_sentiment.csv` (例如: `MSFT_news_sentiment.csv`)
    -   这些数据应包含至少 `HISTORICAL_LOOKBACK_DAYS` (默认为60天) 的历史数据，以便构建观察窗口。

2.  **模型加载**：
    -   `predict_stock_ohlcv` 函数会自动从 `trained_rl_models_v2/` 目录下加载以下文件：
        -   训练好的PPO模型：`ppo_stock_predictor_v2.zip`
        -   观察和动作的标准化器：`scalers_v2.pkl`
        -   `VecNormalize` 的统计信息（用于观察值的归一化）：`vec_normalize_v2.pkl`
    -   请确保这些文件在运行推理前已经通过模型训练脚本生成并放置在正确的位置。

3.  **数据预处理与观察构建**：
    -   函数会加载指定公司的最新股票、财务和情感数据。
    -   它会从这些数据中提取最近 `HISTORICAL_LOOKBACK_DAYS` 天的数据作为模型的观察输入。
    -   所有特征（OHLCV、情感分数、财务指标）的列顺序和数量会与训练时保持一致，并进行前向填充和缺失值处理，以确保数据完整性。
    -   原始观察数据会通过加载的 `observation_scaler` 进行标准化。
    -   标准化后的数据会被展平，并拼接上预测周期指示器（短期或长期），形成最终的观察向量。
    -   最后，通过加载的 `VecNormalize` 实例对最终的观察向量进行归一化，使其符合模型训练时的输入范围。

4.  **模型预测**：
    -   使用加载的PPO模型对归一化后的观察向量进行预测，得到标准化后的未来OHLCV序列动作。

5.  **结果反标准化与输出**：
    -   模型输出的标准化OHLCV序列会通过 `action_ohlcv_scaler` 进行反标准化，恢复到原始的价格和交易量范围。
    -   预测结果会被整理成包含未来日期和预测OHLCV值的 `pandas.DataFrame`。
    -   如果提供了 `output_csv_path`，预测结果将保存为CSV文件到指定路径（例如 `predictions_v2/` 目录下）。否则，结果将打印到控制台。

#### 4.3 示例用法

您可以通过直接运行 `rl_stock_predictor_inference.py` 脚本来执行预测，脚本中包含了示例代码：

```bash
python rl_stock_predictor_inference.py
```

脚本内部的 `if __name__ == '__main__':` 块展示了如何调用 `predict_stock_ohlcv` 函数：

```python
import os
from rl_stock_predictor_inference import predict_stock_ohlcv, PREDICTION_OUTPUT_DIR

# 示例1：对 MSFT 进行短期（7天）预测
predicted_df_msft_short = predict_stock_ohlcv(
    company_symbol='MSFT',
    prediction_horizon_key='short',
    output_csv_path=os.path.join(PREDICTION_OUTPUT_DIR, "MSFT_short_term_prediction.csv")
)

if predicted_df_msft_short is not None:
    print("\nMSFT 短期预测结果:")
    print(predicted_df_msft_short)

# 示例2：对 MSFT 进行长期（30天）预测
predicted_df_msft_long = predict_stock_ohlcv(
    company_symbol='MSFT',
    prediction_horizon_key='long',
    output_csv_path=os.path.join(PREDICTION_OUTPUT_DIR, "MSFT_long_term_prediction.csv")
)

if predicted_df_msft_long is not None:
    print("\nMSFT 长期预测结果:")
    print(predicted_df_msft_long)
```

请根据您的实际需求修改 `company_symbol`、`prediction_horizon_key` 和 `output_csv_path` 参数。
      output_csv_path=os.path.join(PREDICTION_OUTPUT_DIR, f"{example_symbol}_short_term_prediction_v2.csv")
  )
  if predicted_df_short is not None:
      print(f"\nShort-term predictions for {example_symbol}:\n{predicted_df_short}")
  ```
  对于`rl_stock_predictor_inference.py`，其`if __name__ == '__main__':`块中也提供了类似的示例用法。

## 注意事项

- **数据一致性**：确保用于预测的数据（股票、财务、情绪）的格式和特征与训练模型时使用的数据一致。
- **特征工程**：`predict_for_new_data`和`predict_for_new_data_v2`函数内部会进行与训练环境相似的特征构建和归一化，请确保输入数据的完整性，以便正确提取特征。
- **模型路径**：在进行预测时，请确保`model_load_path`、`scaler_load_path`和`vec_normalize_load_path`指向正确的已训练模型和归一化器文件。
- **预测周期**：`rl_stock_predictor.py`支持'micro'（短期）和'macro'（长期）预测，而`rl_stock_predictor_v2.py`支持'short'和'long'预测，请根据您的需求选择合适的预测周期。

该项目展示了如何将强化学习应用于金融市场预测，通过整合多源数据和先进的机器学习技术，为投资决策提供参考。
