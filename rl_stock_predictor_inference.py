import os
import pandas as pd
import numpy as np
import pickle
import gymnasium as gym
from datetime import timedelta

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sklearn.preprocessing import MinMaxScaler
from gymnasium.spaces import Box # Import Box for environment definition

# --- Constants (should match training script) ---
OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
SENTIMENT_FEATURES_NAMES = ['SentimentScore'] # Assuming one sentiment score

# Parameters for data processing with available data
HISTORICAL_LOOKBACK_DAYS = 60 # Must match TRAINING_LOOKBACK_DAYS for valid observation generation

# Parameters assumed from the trained model/VecNormalize stats
TRAINING_LOOKBACK_DAYS = 60 # Adjusted to match expected obs_space (361,) for VecNormalize
TRAINING_NUM_FIN_FEATURES = 0  # Adjusted to match expected obs_space (361,) for VecNormalize
PREDICTION_HORIZONS = {
    'short': 7,  # Predict 7 days into the future
    'long': 30   # Predict 30 days into the future
}

# --- Directories (adjust as needed) ---
INPUT_DATA_DIR = './input_data/' # Consolidated input directory
TRAINED_MODEL_DIR_V2 = './trained_rl_models_v2/'
PREDICTION_OUTPUT_DIR = './predictions_v2/'

os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

# --- Data Loading and Preprocessing Functions (Copied/Adapted from rl_stock_predictor_v2.py) ---
def load_stock_data_v2(symbol, input_data_dir):
    file_path = os.path.join(input_data_dir, f"{symbol.lower()}_stock_data.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date') # Corrected back to 'Date'
        df.columns = df.columns.str.lower() # Ensure column names are lowercase
        df = df.sort_index() # Sort by date in ascending order
        return df
    return pd.DataFrame()

def load_financial_data_v2(symbol, input_data_dir):
    file_path = os.path.join(input_data_dir, f"{symbol.lower()}_financial_metrics.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Filter out non-numeric or irrelevant columns, keep only actual financial metrics
        financial_cols = [col for col in df.columns if col not in ['year', 'period', 'symbol'] and 'Unnamed' not in col]
        # Create a 'Date' column from 'year' and 'period' for potential future use, though not used as index here
        # This part is tricky as 'period' is like 'FY', 'Q1', 'Q2', 'Q3'. For simplicity, we might ignore direct date conversion here
        # or make assumptions. For now, we'll just load the data without a date index.
        return df[financial_cols], financial_cols
    return pd.DataFrame(), []

def load_sentiment_data_v2(symbol, input_data_dir):
    file_path = os.path.join(input_data_dir, f"{symbol.upper()}_news_sentiment.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['date_published'], index_col='date_published') # Corrected to 'date_published'
        # Ensure 'Sentiment_Score' column exists (as per CSV header)
        if 'Sentiment_Score' in df.columns:
            return df[['Sentiment_Score']]
    return pd.DataFrame()

# --- StockTradingEnvV2 (Minimal for observation space definition) ---
# We only need the structure of the environment to define the observation space
# for VecNormalize loading and observation scaling.
class StockTradingEnvV2:
    def __init__(self, financial_feature_names, lookback_window, pred_horizons):
        self.financial_feature_names = financial_feature_names
        self.lookback_window = lookback_window
        self.pred_horizons = pred_horizons
        
        self.num_ohlcv_features = len(OHLCV_COLUMNS)
        self.num_sentiment_features = len(SENTIMENT_FEATURES_NAMES)
        self.num_financial_features = len(self.financial_feature_names)

        self.observation_feature_dim = self.num_ohlcv_features + self.num_sentiment_features + self.num_financial_features
        
        # Observation space: (lookback_window * observation_feature_dim) + horizon_indicator
        # The horizon_indicator is a single float at the end.
        obs_space_shape = (self.lookback_window * self.observation_feature_dim) + 1
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_space_shape,), dtype=np.float32)

        # Action space: (max_pred_horizon * num_ohlcv_features)
        self.max_pred_horizon = max(self.pred_horizons.values())
        action_space_shape = (self.max_pred_horizon, self.num_ohlcv_features)
        self.action_space = Box(low=0, high=1, shape=action_space_shape, dtype=np.float32) # Actions are normalized OHLCV

        # Initialize scalers (they will be loaded from file, but need to exist for structure)
        self.observation_scaler = MinMaxScaler()
        self.action_ohlcv_scaler = MinMaxScaler()

    def _get_observation(self, current_data_df, current_pred_horizon_days, observation_scaler):
        # This function is adapted for direct use in inference, not as part of the Env class
        # It takes a dataframe and scalers as input.
        obs_df_base = current_data_df.iloc[-self.lookback_window:].copy()
        
        if obs_df_base.shape[0] != self.lookback_window:
            raise ValueError(f"Input data for observation window has incorrect length. Expected {self.lookback_window}, got {obs_df_base.shape[0]}")

        raw_obs_features = obs_df_base.values
        scaled_obs_features = observation_scaler.transform(raw_obs_features)
        
        horizon_indicator = 0.0 if current_pred_horizon_days == self.pred_horizons['short'] else 1.0
        obs = np.concatenate((scaled_obs_features.flatten(), np.array([horizon_indicator], dtype=np.float32)))
        return obs.astype(np.float32)


# --- Main Prediction Function ---
def predict_stock_ohlcv(
    company_symbol: str,
    prediction_horizon_key: str = 'short', # 'short' or 'long'
    output_csv_path: str = None
):
    """
    Loads a trained RL model and predicts future stock OHLCV prices for a given company.

    Args:
        company_symbol (str): The stock symbol (e.g., 'AAPL', 'MSFT').
        prediction_horizon_key (str): 'short' for 7 days, 'long' for 30 days prediction.
        output_csv_path (str, optional): Path to save the predictions CSV. If None, prints to console.

    Returns:
        pd.DataFrame: DataFrame containing predicted OHLCV values and dates.
    """
    print(f"--- Starting Prediction for {company_symbol} ({prediction_horizon_key} term) ---")

    model_load_path = os.path.join(TRAINED_MODEL_DIR_V2, "ppo_stock_predictor_v2.zip")
    scaler_load_path = os.path.join(TRAINED_MODEL_DIR_V2, "scalers_v2.pkl")
    vec_normalize_load_path = os.path.join(TRAINED_MODEL_DIR_V2, "vec_normalize_v2.pkl")

    if not all(os.path.exists(p) for p in [model_load_path, scaler_load_path, vec_normalize_load_path]):
        print("Error: Trained model, scalers, or VecNormalize file not found. Please ensure the model is trained.")
        if not os.path.exists(model_load_path): print(f"Missing: {model_load_path}")
        if not os.path.exists(scaler_load_path): print(f"Missing: {scaler_load_path}")
        if not os.path.exists(vec_normalize_load_path): print(f"Missing: {vec_normalize_load_path}")
        return None

    # 1. Load model, scalers, and VecNormalize stats
    try:
        model = PPO.load(model_load_path)
        with open(scaler_load_path, 'rb') as f:
            scalers = pickle.load(f)
        observation_scaler = scalers['observation_scaler']
        action_ohlcv_scaler = scalers['action_ohlcv_scaler']

        # Determine financial feature names from the loaded observation_scaler's n_features_in_
        # This assumes the scaler was fitted on the full observation feature set (OHLCV + Sentiment + Financial)
        num_total_obs_features_trained = observation_scaler.n_features_in_
        num_ohlcv_sentiment_features = len(OHLCV_COLUMNS) + len(SENTIMENT_FEATURES_NAMES)
        num_financial_features_trained = num_total_obs_features_trained - num_ohlcv_sentiment_features
        
        # Create dummy financial feature names that match how the scaler was trained
        dummy_financial_features_for_env = [f'fin_feat_{i}' for i in range(num_financial_features_trained)]

        # Create a dummy environment instance to correctly load VecNormalize stats
        # Define a simple Gymnasium-compatible environment for VecNormalize to load its stats
        # This environment only needs to define observation_space and action_space correctly.
        class DummyGymEnv(gym.Env):
            def __init__(self, observation_space, action_space):
                super().__init__()
                self.observation_space = observation_space
                self.action_space = action_space

            def step(self, action):
                return self.observation_space.sample(), 0, False, False, {}

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                return self.observation_space.sample(), {}

        # 1. Initialize a dummy environment with the observation and action spaces that match the TRAINED model.
        #    This is needed for VecNormalize.load() to work correctly.
        #    The observation space shape is assumed to be (TRAINING_LOOKBACK_DAYS * len(OHLCV_COLUMNS)) + TRAINING_NUM_FIN_FEATURES = 361
        dummy_training_financial_features = [f'train_fin_feat_{i}' for i in range(TRAINING_NUM_FIN_FEATURES)]
        env_for_stats = StockTradingEnvV2(
            financial_feature_names=dummy_training_financial_features, 
            lookback_window=TRAINING_LOOKBACK_DAYS, 
            pred_horizons=PREDICTION_HORIZONS
        )
        dummy_env = DummyGymEnv(observation_space=env_for_stats.observation_space, action_space=env_for_stats.action_space)
        venv = DummyVecEnv([lambda: dummy_env]) # This is the DummyVecEnv to be wrapped by VecNormalize
        
        # Load the VecNormalize statistics
        vec_env_for_norm_load = VecNormalize.load(vec_normalize_load_path, venv) # Pass venv here
        vec_env_for_norm_load.training = False # Set to inference mode
        vec_env_for_norm_load.norm_reward = False # Not needed for prediction

    except Exception as e:
        print(f"Error loading model/scalers/VecNormalize: {e}")
        return None
    print("Model, scalers, and VecNormalize stats loaded.")

    # 2. Load and preprocess LATEST data for the observation window
    print(f"Loading latest data for {company_symbol} from {INPUT_DATA_DIR}...")
    stock_df = load_stock_data_v2(company_symbol, INPUT_DATA_DIR)
    financial_df, fin_feats_actual = load_financial_data_v2(company_symbol, INPUT_DATA_DIR)
    sentiment_df = load_sentiment_data_v2(company_symbol, INPUT_DATA_DIR)

    if stock_df.empty or len(stock_df) < HISTORICAL_LOOKBACK_DAYS:
        print(f"Not enough stock data for {company_symbol} for lookback ({len(stock_df)} days). Prediction aborted.")
        return None

    # Combine into a single DataFrame for observation
    # Start with the latest stock data (last HISTORICAL_LOOKBACK_DAYS rows)
    obs_df_base = stock_df.iloc[-HISTORICAL_LOOKBACK_DAYS:][OHLCV_COLUMNS].copy()
    current_obs_dates = obs_df_base.index # Dates for alignment

    # Add financial features
    aligned_financial_df = pd.DataFrame(index=current_obs_dates)
    if not financial_df.empty:
        temp_financial_df = financial_df.copy()
        # Ensure 'Date' column is set as index if it exists and is not already the index name
        if (temp_financial_df.index.name is None or 'Date' not in temp_financial_df.index.name) and 'Date' in temp_financial_df.columns:
             temp_financial_df.set_index('Date', inplace=True)
        
        # Check if the index is named 'Date' (or 'date' due to potential lowercasing elsewhere, though financial_df is not lowercased yet)
        if temp_financial_df.index.name == 'Date' or temp_financial_df.index.name == 'date':
            reindexed_fin = temp_financial_df.reindex(current_obs_dates, method='ffill')
            for i, dummy_col_name in enumerate(dummy_financial_features_for_env):
                if i < len(fin_feats_actual) and fin_feats_actual[i] in reindexed_fin.columns:
                    aligned_financial_df[dummy_col_name] = reindexed_fin[fin_feats_actual[i]]
                else:
                    aligned_financial_df[dummy_col_name] = 0.0 # Default if not enough actual features or mismatch
        else:
            for dummy_col_name in dummy_financial_features_for_env:
                aligned_financial_df[dummy_col_name] = 0.0
    else:
        for dummy_col_name in dummy_financial_features_for_env:
            aligned_financial_df[dummy_col_name] = 0.0
    obs_df_base = pd.concat([obs_df_base, aligned_financial_df.fillna(0.0)], axis=1)

    # Add sentiment features
    if not sentiment_df.empty:
        temp_sentiment_df = sentiment_df.copy()
        if 'Date' not in temp_sentiment_df.index.name and 'Date' in temp_sentiment_df.columns:
            temp_sentiment_df.set_index('Date', inplace=True)
        
        if 'Date' in temp_sentiment_df.index.name:
            reindexed_sentiment = temp_sentiment_df.reindex(current_obs_dates, method='ffill')
            obs_df_base['SentimentScore'] = reindexed_sentiment.get('SentimentScore', 0.0)
        else:
            obs_df_base['SentimentScore'] = 0.0
    else:
        obs_df_base['SentimentScore'] = 0.0
    obs_df_base['SentimentScore'].fillna(0.0, inplace=True)

    # Ensure final column order and fill any remaining NaNs
    final_ordered_cols_for_obs = OHLCV_COLUMNS + SENTIMENT_FEATURES_NAMES + dummy_financial_features_for_env
    obs_input_df = obs_df_base.reindex(columns=final_ordered_cols_for_obs, fill_value=0.0).fillna(0.0)

    # Validate the shape of the input observation data
    expected_obs_feature_dim = len(OHLCV_COLUMNS) + len(SENTIMENT_FEATURES_NAMES) + len(dummy_financial_features_for_env)
    if obs_input_df.shape[0] != HISTORICAL_LOOKBACK_DAYS or obs_input_df.shape[1] != expected_obs_feature_dim:
        print(f"Error: Observation DataFrame shape mismatch. Expected: ({HISTORICAL_LOOKBACK_DAYS}, {expected_obs_feature_dim}), Got: {obs_input_df.shape}")
        return None

    # 3. Construct observation vector
    pred_horizon_days_value = PREDICTION_HORIZONS.get(prediction_horizon_key)
    if pred_horizon_days_value is None:
        print(f"Invalid prediction_horizon_key: {prediction_horizon_key}. Use 'short' or 'long'."); return None

    # Use the adapted _get_observation function
    # Note: This requires the Box import from gym.spaces or stable_baselines3.common.vec_env
    from gymnasium.spaces import Box # Import Box for environment definition
    
    # Create a temporary dummy env instance to call its _get_observation method
    # This is a bit hacky but reuses the logic from the training environment.
    temp_dummy_env = StockTradingEnvV2(
        financial_feature_names=dummy_financial_features_for_env,
        lookback_window=HISTORICAL_LOOKBACK_DAYS,
        pred_horizons=PREDICTION_HORIZONS
    )
    # Manually set the observation_scaler for the temporary env
    temp_dummy_env.observation_scaler = observation_scaler

    single_obs_flat = temp_dummy_env._get_observation(obs_input_df, pred_horizon_days_value, observation_scaler)
    
    # Normalize the observation using the loaded VecNormalize stats
    normalized_obs_for_model = vec_env_for_norm_load.normalize_obs(single_obs_flat.reshape(1, -1))

    # 4. Use model.predict()
    action, _states = model.predict(normalized_obs_for_model, deterministic=True)
    # Flatten the action array if it's 2D (e.g., from a batch prediction)
    action = action.flatten()
    # Reshape the predicted normalized OHLCV data
    # The action space is (pred_horizon_days * num_ohlcv_features)
    # So we reshape the action to (pred_horizon_days, num_ohlcv_features)
    predicted_ohlcv_norm = action[:pred_horizon_days_value * len(OHLCV_COLUMNS)].reshape(pred_horizon_days_value, len(OHLCV_COLUMNS))

    # 5. Denormalize actions
    predicted_ohlcv_denorm = action_ohlcv_scaler.inverse_transform(predicted_ohlcv_norm)

    # 6. Format and return predictions
    last_date_in_input = obs_input_df.index[-1]
    future_dates = pd.to_datetime([last_date_in_input + timedelta(days=i+1) for i in range(pred_horizon_days_value)])
    
    predictions_df = pd.DataFrame(predicted_ohlcv_denorm, columns=[f"Predicted_{col}" for col in OHLCV_COLUMNS])
    predictions_df['Date'] = future_dates
    predictions_df = predictions_df[['Date'] + [f"Predicted_{col}" for col in OHLCV_COLUMNS]]

    if output_csv_path:
        try:
            output_dir = os.path.dirname(output_csv_path)
            if output_dir: os.makedirs(output_dir, exist_ok=True)
            predictions_df.to_csv(output_csv_path, index=False)
            print(f"Predictions saved to {output_csv_path}")
        except Exception as e: print(f"Error saving predictions to CSV: {e}")

    print(f"--- Prediction for {company_symbol} Complete ---")
    return predictions_df

# --- Example Usage (Run this script directly) ---
if __name__ == '__main__':
    print("Starting RL Stock Predictor Inference...")

    # Example: Predict for AAPL, short-term
    # Ensure you have 'AAPL_stock_data.csv', 'aapl_financial_metrics.csv',
    # and 'AAPL_news_sentiment.csv' in their respective input directories.
    # These files should contain at least HISTORICAL_LOOKBACK_DAYS of data.
    
    # You can specify custom paths for input data if they are not in the default locations
    # stock_data_path = './input_data/my_aapl_stock.csv'
    # financial_data_path = './input_data/my_aapl_financial.csv'
    # sentiment_data_path = './input_data/my_aapl_sentiment.csv'

    # Example 1: Short-term prediction for MSFT (using data from input_data folder)
    predicted_df_msft_short = predict_stock_ohlcv(
        company_symbol='MSFT',
        prediction_horizon_key='short',
        output_csv_path=os.path.join(PREDICTION_OUTPUT_DIR, "MSFT_short_term_prediction.csv")
    )
    if predicted_df_msft_short is not None:
        print("\nMSFT Short-Term Predictions:")
        print(predicted_df_msft_short)


    # Example 2: Long-term prediction for MSFT (using data from input_data folder)
    predicted_df_msft_long = predict_stock_ohlcv(
        company_symbol='MSFT',
        prediction_horizon_key='long',
        output_csv_path=os.path.join(PREDICTION_OUTPUT_DIR, "MSFT_long_term_prediction.csv")
    )
    if predicted_df_msft_long is not None:
        print("\nMSFT Long-Term Predictions:")
        print(predicted_df_msft_long)

    print("\nInference process complete.")