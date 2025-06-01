import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import os
import pickle
import joblib # Keep for now if other parts might use it, but scalers will use pickle
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# --- Constants ---
STOCK_DATA_DIR = 'stock_data'
PROCESSED_FINANCIAL_DATA_DIR = 'processed_financial_data'
SENTIMENT_DATA_DIR = 'analyzed_sentiment_results'
TRAINED_MODEL_DIR_V2 = 'trained_rl_models_v2' # New directory for V2 models
PREDICTION_OUTPUT_DIR = 'predictions_v2' # Directory for prediction outputs
COMPANY_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMD']
LOAD_PRETRAINED_MODEL_IF_EXISTS = True # Flag to load a pre-trained model if available

# Ensure directories exist
if not os.path.exists(TRAINED_MODEL_DIR_V2):
    os.makedirs(TRAINED_MODEL_DIR_V2)
if not os.path.exists(PREDICTION_OUTPUT_DIR):
    os.makedirs(PREDICTION_OUTPUT_DIR)

# Observation and Prediction constants
HISTORICAL_LOOKBACK_DAYS = 60  # How many past days of data to observe
PREDICTION_HORIZONS = {'short': 7, 'long': 180} # Days to predict: 7 days or 180 days (approx. 6 months)
OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume'] # Standardized to lowercase
SENTIMENT_FEATURES_NAMES = ['sentimentscore'] # Standardized to lowercase

# Reward mechanism constants
ERROR_THRESHOLD_MAX_REWARD = 0.01 # Error < 1% for max reward
ERROR_THRESHOLD_MID_REWARD = 0.05 # Error < 5% for mid reward
ERROR_THRESHOLD_MIN_REWARD = 0.10 # Error < 10% for min reward (or start of penalty)

# --- Data loading and preprocessing ---
def load_stock_data_v2(symbol, stock_dir):
    file_path = os.path.join(stock_dir, f"{symbol.lower()}_stock_data.csv")
    if not os.path.exists(file_path):
        print(f"Stock data file not found for {symbol}: {file_path}")
        return pd.DataFrame()
    try:
        # Try reading with 'Date' first
        df = pd.read_csv(file_path, low_memory=False) # Add low_memory=False for mixed types
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            date_col_to_use = 'Date'
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            date_col_to_use = 'date'
        else:
            print(f"Date column ('Date' or 'date') not found in {file_path}. Returning empty DataFrame.")
            return pd.DataFrame()
        df.dropna(subset=[date_col_to_use], inplace=True) # Drop rows where date conversion failed
    except Exception as e:
        print(f"Error reading or processing date for {file_path}: {e}. Returning empty DataFrame.")
        return pd.DataFrame()

    df.rename(columns={date_col_to_use: 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df.set_index('date', inplace=True)
    df.columns = [col.lower() for col in df.columns]
    
    # Ensure all required OHLCV columns exist, otherwise return empty
    if not all(col in df.columns for col in OHLCV_COLUMNS):
        missing_cols = [col for col in OHLCV_COLUMNS if col not in df.columns]
        print(f"Missing required stock data columns after processing {file_path}: {missing_cols}. Returning empty DataFrame.")
        return pd.DataFrame()
    return df[OHLCV_COLUMNS]

def load_financial_data_v2(symbol, financial_dir):
    file_path = os.path.join(financial_dir, symbol.lower(), f"{symbol.lower()}_financial_metrics.csv")
    if not os.path.exists(file_path):
        print(f"Financial data file not found for {symbol}: {file_path}")
        return pd.DataFrame(), []
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        date_col_to_use = 'Date'
        if 'Date' not in df.columns and 'date' in df.columns:
            df = pd.read_csv(file_path, parse_dates=['date'])
            date_col_to_use = 'date'
        elif 'Date' not in df.columns and 'date' not in df.columns:
            print(f"Date column ('Date' or 'date') not found in {file_path}. Returning empty DataFrame.")
            return pd.DataFrame(), []
    except Exception as e:
        print(f"Error reading or parsing date for {file_path}: {e}. Returning empty DataFrame.")
        return pd.DataFrame(), []

    df.rename(columns={date_col_to_use: 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df.set_index('date', inplace=True)
    df.columns = [col.lower() for col in df.columns]
    
    financial_feature_names = [col for col in df.columns if col not in ['unnamed: 0', 'symbol'] and 'symbol' not in col.lower()]
    # Ensure only existing columns are returned
    existing_financial_features = [col for col in financial_feature_names if col in df.columns]
    if not existing_financial_features and len(financial_feature_names) > 0:
        print(f"No financial features found after processing {file_path} from potential: {financial_feature_names}")
    return df[existing_financial_features], existing_financial_features

def load_sentiment_data_v2(symbol, sentiment_dir):
    file_path = os.path.join(sentiment_dir, symbol.lower(), f"{symbol.upper()}_news_sentiment.csv")
    if not os.path.exists(file_path):
        print(f"Sentiment data file not found for {symbol}: {file_path}")
        return pd.DataFrame()
    try:
        # Try reading with 'Date' first
        df = pd.read_csv(file_path, low_memory=False) # Add low_memory=False for mixed types
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            date_col_to_use = 'Date'
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            date_col_to_use = 'date'
        else:
            print(f"Date column ('Date' or 'date') not found in {file_path}. Returning empty DataFrame.")
            return pd.DataFrame()
        df.dropna(subset=[date_col_to_use], inplace=True) # Drop rows where date conversion failed
    except Exception as e:
        print(f"Error reading or processing date for {file_path}: {e}. Returning empty DataFrame.")
        return pd.DataFrame()

    df.rename(columns={date_col_to_use: 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df.set_index('date', inplace=True)

    # Standardize sentiment score column name to 'sentimentscore'
    if 'overall_sentiment_score' in df.columns:
        df.rename(columns={'overall_sentiment_score': 'sentimentscore'}, inplace=True)
    elif 'SentimentScore' in df.columns:
        df.rename(columns={'SentimentScore': 'sentimentscore'}, inplace=True)
    elif 'sentimentscore' in df.columns: # Already correct
        pass 
    else:
        print(f"Sentiment score column ('overall_sentiment_score' or 'SentimentScore') not found in {file_path}. Returning empty DataFrame.")
        return pd.DataFrame()

    df.columns = [col.lower() for col in df.columns] # Ensure all other columns are lowercase too

    if SENTIMENT_FEATURES_NAMES[0] in df.columns:
        return df[[SENTIMENT_FEATURES_NAMES[0]]]
    else:
        print(f"Final sentiment score column '{SENTIMENT_FEATURES_NAMES[0]}' not found after processing {file_path}.")
        return pd.DataFrame()

def load_and_preprocess_data_v2(company_symbols, stock_dir, financial_dir, sentiment_dir):
    all_companies_data = {}
    all_financial_feature_names = set()

    for symbol in company_symbols:
        print(f"Processing data for {symbol}...")
        stock_df = load_stock_data_v2(symbol, stock_dir)
        financial_df, current_fin_features = load_financial_data_v2(symbol, financial_dir)
        sentiment_df = load_sentiment_data_v2(symbol, sentiment_dir)

        if stock_df.empty:
            print(f"Skipping {symbol} due to missing stock data.")
            continue

        all_financial_feature_names.update(current_fin_features)
        merged_df = stock_df.copy()

        if not financial_df.empty:
            merged_df = pd.merge(merged_df, financial_df, on='Date', how='left')
            # Forward fill financial data: crucial as it's reported periodically
            merged_df[current_fin_features] = merged_df[current_fin_features].ffill()

        if not sentiment_df.empty:
            merged_df = pd.merge(merged_df, sentiment_df, on='Date', how='left')
            # Forward fill sentiment, then fill remaining NaNs (e.g. at start) with neutral 0
            merged_df['SentimentScore'] = merged_df['SentimentScore'].ffill().fillna(0) 
        else:
            # If no sentiment data, create a column of zeros
            merged_df['SentimentScore'] = 0.0

        # Fill NaNs for OHLCV (e.g. from market holidays if not handled in source)
        merged_df[OHLCV_COLUMNS] = merged_df[OHLCV_COLUMNS].ffill().bfill()
        # For financial features, ffill is done. Fill initial NaNs with 0.
        for fin_col in current_fin_features:
            if fin_col in merged_df.columns:
                 merged_df[fin_col] = merged_df[fin_col].fillna(0)
        
        merged_df.dropna(subset=OHLCV_COLUMNS, inplace=True) # Critical drop after all merges & fills

        if not merged_df.empty:
            all_companies_data[symbol] = merged_df
        else:
            print(f"No data remaining for {symbol} after preprocessing.")

    final_financial_feature_names = sorted(list(all_financial_feature_names))

    for symbol in list(all_companies_data.keys()):
        df = all_companies_data[symbol]
        # Ensure all dataframes have all financial columns, fill with 0 if a company lacks a specific feature
        for fin_col_master in final_financial_feature_names:
            if fin_col_master not in df.columns:
                df[fin_col_master] = 0.0
        
        # Define the final column order for the observation space features
        # This order must be consistent for the scaler and the environment
        ordered_feature_cols = OHLCV_COLUMNS + SENTIMENT_FEATURES_NAMES + final_financial_feature_names
        
        # Ensure SentimentScore column exists if it wasn't created (e.g. no sentiment files at all)
        if 'SentimentScore' not in df.columns:
            df['SentimentScore'] = 0.0
            if 'SentimentScore' not in ordered_feature_cols: # Should be there from SENTIMENT_FEATURES_NAMES
                 ordered_feature_cols = OHLCV_COLUMNS + ['SentimentScore'] + final_financial_feature_names

        # Select and reorder columns
        # Only keep columns that are part of our defined feature set
        df_reordered = df.reindex(columns=ordered_feature_cols, fill_value=0.0)
        
        if df_reordered.isnull().values.any():
            print(f"Warning: NaNs detected in final preprocessed data for {symbol} before storing. Filling with 0.")
            df_reordered.fillna(0, inplace=True)
        
        all_companies_data[symbol] = df_reordered

    return all_companies_data, final_financial_feature_names

# --- Placeholder for RL Environment V2 ---
class StockTradingEnvV2(gym.Env):

    def __init__(self, data_dict, financial_features, lookback_window=HISTORICAL_LOOKBACK_DAYS, pred_horizons=PREDICTION_HORIZONS):
        super(StockTradingEnvV2, self).__init__()
        self.data_dict = data_dict # Dict: {symbol: DataFrame}
        self.company_symbols = list(data_dict.keys())
        self.financial_features = financial_features
        self.lookback_window = lookback_window
        self.pred_horizons = pred_horizons
        self.num_ohlcv_features = len(OHLCV_COLUMNS)
        self.num_sentiment_features = len(SENTIMENT_FEATURES_NAMES)
        self.num_financial_features = len(self.financial_features)

        # Action space: Predict N days of (O,H,L,C,V) - normalized values
        # We'll use the longest prediction horizon to define the action space shape
        self.max_pred_horizon = max(self.pred_horizons.values())
        self.action_space = spaces.Box(low=-1.0, high=1.0, 
                                       shape=(self.max_pred_horizon, self.num_ohlcv_features), 
                                       dtype=np.float32)

        # Observation space: lookback_window days of (O,H,L,C,V, Sent, FinFeats) + current horizon type
        self.observation_feature_dim = (self.num_ohlcv_features + 
                                        self.num_sentiment_features + 
                                        self.num_financial_features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.lookback_window * self.observation_feature_dim + 1,), # +1 for horizon type
                                            dtype=np.float32)
        
        self.current_company_symbol = None
        self.current_data = None # This will be a DataFrame for the selected company
        self.current_step = 0 # Current time step index within self.current_data
        self.current_pred_horizon_days = 0 # 7 or 180 for the current episode
        self.episode_start_step = 0 # The first step of the lookback window in current_data

        self.observation_scaler = MinMaxScaler()
        self.action_ohlcv_scaler = MinMaxScaler() 
        self._prepare_scalers()

    def _prepare_scalers(self):
        all_feature_data_list = []
        all_ohlcv_data_list = []

        if not self.data_dict or not self.company_symbols:
            print("Warning: Data dictionary or company symbols empty, cannot prepare scalers.")
            # Define dummy scalers to prevent errors if no data is loaded (e.g. during init before data load)
            # This is a fallback; ideally, data should always be present for a functional env.
            self.observation_scaler.fit(np.zeros((1, self.observation_feature_dim)))
            self.action_ohlcv_scaler.fit(np.zeros((1, self.num_ohlcv_features)))
            return

        # The columns for observation scaling are OHLCV + Sentiment + Financials
        # Their order is defined by how load_and_preprocess_data_v2 structures the DataFrames
        # and subsequently by self.observation_feature_dim calculation.
        # The DataFrame columns should already be in the correct order from preprocessing.

        for symbol in self.company_symbols:
            if symbol not in self.data_dict or self.data_dict[symbol].empty:
                print(f"Skipping scaler prep for {symbol} due to missing/empty data.")
                continue
            df = self.data_dict[symbol]
            # Assuming df columns are already correctly ordered: OHLCV, SentimentScore, Financials
            # The number of these columns should match self.observation_feature_dim
            if df.shape[1] != self.observation_feature_dim:
                print(f"Warning: Column count mismatch for {symbol}. Expected {self.observation_feature_dim}, got {df.shape[1]}. Scaler fitting might be incorrect.")
                # This indicates an issue in data preprocessing or feature definition consistency.
                # For robustness, try to use available columns if a subset matches, or skip.
                # For now, we proceed, but this needs careful checking.
            all_feature_data_list.append(df.iloc[:, :self.observation_feature_dim]) # Use all columns intended for observation
            all_ohlcv_data_list.append(df[OHLCV_COLUMNS])

        if not all_feature_data_list:
            print("Warning: No feature data collected to fit observation_scaler. Using zeros.")
            self.observation_scaler.fit(np.zeros((1, self.observation_feature_dim)))
        else:
            full_feature_df = pd.concat(all_feature_data_list)
            if not full_feature_df.empty:
                self.observation_scaler.fit(full_feature_df.values)
            else:
                 self.observation_scaler.fit(np.zeros((1, self.observation_feature_dim)))
        
        if not all_ohlcv_data_list:
            print("Warning: No OHLCV data collected to fit action_ohlcv_scaler. Using zeros.")
            self.action_ohlcv_scaler.fit(np.zeros((1, self.num_ohlcv_features)))
        else:
            full_ohlcv_df = pd.concat(all_ohlcv_data_list)
            if not full_ohlcv_df.empty:
                 self.action_ohlcv_scaler.fit(full_ohlcv_df.values)
            else:
                self.action_ohlcv_scaler.fit(np.zeros((1, self.num_ohlcv_features)))
        print("Scalers prepared.")

    def _get_observation(self):
        obs_start_idx = self.current_step - self.lookback_window
        obs_end_idx = self.current_step # Observation is data UP TO self.current_step (exclusive)

        if obs_start_idx < 0:
            # This can happen if current_step is too close to the beginning after a reset.
            # Pad with zeros or handle as an invalid state.
            # For simplicity, let's assume reset logic ensures current_step >= lookback_window.
            # If it still happens, it's an issue with episode start logic.
            print(f"Critical Error: obs_start_idx < 0 for {self.current_company_symbol} at step {self.current_step}. Reset logic needs check.")
            # Fallback to a zero observation to avoid crashing, but this state is problematic.
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Columns are already ordered: OHLCV, Sentiment, Financials
        raw_obs_features_df = self.current_data.iloc[obs_start_idx:obs_end_idx]
        
        if raw_obs_features_df.shape[0] != self.lookback_window or raw_obs_features_df.shape[1] != self.observation_feature_dim:
            print(f"Observation shape error for {self.current_company_symbol} at step {self.current_step}.")
            print(f"Expected lookback: {self.lookback_window}, got: {raw_obs_features_df.shape[0]}")
            print(f"Expected feature_dim: {self.observation_feature_dim}, got: {raw_obs_features_df.shape[1]}")
            # This indicates a problem, possibly with data length or preprocessing consistency.
            # Fallback to zeros, but this needs investigation.
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        raw_obs_features = raw_obs_features_df.values
        scaled_obs_features = self.observation_scaler.transform(raw_obs_features)
        
        horizon_indicator = 0.0 if self.current_pred_horizon_days == self.pred_horizons['short'] else 1.0
        obs = np.concatenate((scaled_obs_features.flatten(), np.array([horizon_indicator], dtype=np.float32)))
        return obs.astype(np.float32)

    def _get_target_ohlcv_sequence(self, start_idx, length):
        target_end_idx = start_idx + length
        if target_end_idx > len(self.current_data):
            # Not enough future data available for the full prediction horizon
            # This is a common case for termination.
            return None 
        
        # Ensure we only select OHLCV columns for the target
        actual_ohlcv_raw = self.current_data[OHLCV_COLUMNS].iloc[start_idx : target_end_idx].values
        
        if actual_ohlcv_raw.shape[0] != length:
            # This means we couldn't get the full length, e.g., at the very end of data
            return None
            
        return actual_ohlcv_raw

    def _calculate_reward(self, predicted_ohlcv_norm, actual_ohlcv_raw):
        if actual_ohlcv_raw is None or predicted_ohlcv_norm.shape[0] != actual_ohlcv_raw.shape[0] or predicted_ohlcv_norm.shape[1] != len(OHLCV_COLUMNS):
            print(f"Reward calc error: Shape mismatch or None actuals. Pred shape: {predicted_ohlcv_norm.shape}, Actual shape: {actual_ohlcv_raw.shape if actual_ohlcv_raw is not None else 'None'}")
            return -200 

        try:
            if not hasattr(self.action_ohlcv_scaler, 'scale_') or self.action_ohlcv_scaler.scale_ is None:
                 print("Action OHLCV scaler not fitted. Returning high penalty.")
                 return -250 # Scaler not fitted
            if self.action_ohlcv_scaler.n_features_in_ != predicted_ohlcv_norm.shape[1]:
                print(f"Action scaler feature mismatch: scaler expects {self.action_ohlcv_scaler.n_features_in_}, got {predicted_ohlcv_norm.shape[1]}")
                return -250 
            predicted_ohlcv_denorm = self.action_ohlcv_scaler.inverse_transform(predicted_ohlcv_norm)
        except Exception as e:
            print(f"Error during inverse_transform of predicted_ohlcv_norm: {e}")
            print(f"predicted_ohlcv_norm shape: {predicted_ohlcv_norm.shape}, min: {predicted_ohlcv_norm.min()}, max: {predicted_ohlcv_norm.max()}")
            return -200 

        # Ensure 'close' is used for indexing, matching the lowercase convention
        close_idx = -1
        try:
            close_idx = OHLCV_COLUMNS.index('close')
        except ValueError:
            # Fallback if 'close' is not found, try 'Close' - though this shouldn't happen with proper standardization
            try:
                close_idx = OHLCV_COLUMNS.index('Close') 
                print("Warning: Using 'Close' for indexing in reward calculation. OHLCV_COLUMNS should be standardized to lowercase.")
            except ValueError:
                print("Critical Error: Neither 'close' nor 'Close' found in OHLCV_COLUMNS for reward calculation.")
                return -300 # Indicate a severe configuration error

        pred_close_denorm = predicted_ohlcv_denorm[:, close_idx]
        actual_close_raw_prices = actual_ohlcv_raw[:, close_idx]

        if len(actual_close_raw_prices) == 0:
             print("Reward calc error: actual_close_raw_prices is empty.")
             return -150

        errors_pct = np.abs((pred_close_denorm - actual_close_raw_prices) / (actual_close_raw_prices + 1e-9)) 
        mean_error_pct = np.mean(errors_pct)

        reward = 0
        if mean_error_pct < ERROR_THRESHOLD_MAX_REWARD: 
            reward = 100  
        elif mean_error_pct < ERROR_THRESHOLD_MID_REWARD: 
            reward = 50   
        elif mean_error_pct < ERROR_THRESHOLD_MIN_REWARD: 
            reward = 20   
        else: 
            reward = - ( (mean_error_pct - ERROR_THRESHOLD_MIN_REWARD) * 100 ) 

        # Last observed close price is from self.current_data at index self.current_step - 1
        # This is the close price of the day just before the prediction period starts.
        if self.current_step > 0 and (self.current_step -1) < len(self.current_data) :
            last_observed_close = self.current_data['close'].iloc[self.current_step - 1] # Use lowercase 'close'
            
            pred_final_day_direction = np.sign(pred_close_denorm[-1] - last_observed_close)
            actual_final_day_direction = np.sign(actual_close_raw_prices[-1] - last_observed_close)

            if pred_final_day_direction == actual_final_day_direction and actual_final_day_direction != 0: 
                reward += 15 
            elif pred_final_day_direction != actual_final_day_direction and actual_final_day_direction != 0: 
                reward -= 15 
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        valid_episode_found = False
        max_reset_attempts = len(self.company_symbols) * len(self.pred_horizons) * 2 # Try each combo twice
        attempts = 0

        while not valid_episode_found and attempts < max_reset_attempts:
            attempts += 1
            self.current_company_symbol = np.random.choice(self.company_symbols)
            self.current_data = self.data_dict.get(self.current_company_symbol)
            self.current_pred_horizon_days = np.random.choice(list(self.pred_horizons.values()))

            if self.current_data is None or self.current_data.empty:
                print(f"Warning: Data for {self.current_company_symbol} is missing or empty. Resampling company.")
                continue

            # Minimum length required for one full step (lookback + prediction horizon)
            min_data_len_for_episode = self.lookback_window + self.current_pred_horizon_days
            if len(self.current_data) < min_data_len_for_episode:
                # print(f"Data for {self.current_company_symbol} ({len(self.current_data)} days) is too short for lookback {self.lookback_window} + horizon {self.current_pred_horizon_days}. Resampling.")
                continue # Try another company or horizon

            # Valid range for self.current_step (which is the END of the observation window)
            # Earliest possible current_step is self.lookback_window (so obs is from 0 to lookback_window-1)
            # Latest possible current_step is len(self.current_data) - self.current_pred_horizon_days
            # (so target sequence ends at len(self.current_data)-1)
            min_current_step_val = self.lookback_window 
            max_current_step_val = len(self.current_data) - self.current_pred_horizon_days
            
            if min_current_step_val > max_current_step_val:
                # This means lookback_window + current_pred_horizon_days > len(self.current_data)
                # This check is similar to min_data_len_for_episode, but more direct for picking step
                # print(f"Cannot find valid start step for {self.current_company_symbol} with horizon {self.current_pred_horizon_days}. Data len: {len(self.current_data)}. Resampling.")
                continue
            
            self.current_step = np.random.randint(min_current_step_val, max_current_step_val + 1)
            self.episode_start_step = self.current_step - self.lookback_window # Start of the historical data window for the first obs
            valid_episode_found = True
        
        if not valid_episode_found:
            # This is a critical failure - no valid company/horizon/start_step combination found
            # Could happen if all data is too short, or lookback/horizons are too large
            print("CRITICAL: Could not find a valid episode start after multiple attempts. Check data and parameters.")
            # Fallback: create a dummy observation and info, and likely terminate immediately in step()
            # Or, raise an error to halt training if this state is unrecoverable.
            # For now, let's set up a minimal state that might lead to immediate termination.
            self.current_company_symbol = self.company_symbols[0] if self.company_symbols else None
            self.current_data = self.data_dict.get(self.current_company_symbol) if self.current_company_symbol else pd.DataFrame()
            self.current_pred_horizon_days = self.pred_horizons['short']
            self.current_step = self.lookback_window if self.current_data is not None and len(self.current_data) >= self.lookback_window else 0
            self.episode_start_step = max(0, self.current_step - self.lookback_window)
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = self._get_info()
            info['error'] = "Failed to initialize a valid episode."
            return observation, info

        observation = self._get_observation()
        info = self._get_info()
        # print(f"Reset successful: Company {self.current_company_symbol}, Horizon {self.current_pred_horizon_days}, Start Step {self.current_step}, Obs Shape {observation.shape}")
        return observation, info

    def step(self, action):
        # Action is the normalized predicted OHLCV sequence.
        # The policy always outputs for self.max_pred_horizon.
        # We need to take the relevant part for self.current_pred_horizon_days.
        # Action shape from policy is (self.max_pred_horizon * self.num_ohlcv_features,)
        # or (self.max_pred_horizon, self.num_ohlcv_features) if already shaped by SB3.
        
        # First, ensure action is a flat array if it comes shaped from SB3 for some reason
        # (though typically it's flat for MultiDiscrete/Box actions that are then reshaped by env)
        # The action_space is defined as Box(shape=(self.max_pred_horizon, self.num_ohlcv_features)),
        # so 'action' should arrive with this shape.

        if action.shape[0] != self.max_pred_horizon or action.shape[1] != self.num_ohlcv_features:
            print(f"Warning: Action received in step has unexpected shape {action.shape}. Expected ({self.max_pred_horizon}, {self.num_ohlcv_features}). Attempting to proceed by taking first {self.num_ohlcv_features} columns if 1D or handling as error.")
            # If action is 1D and its size is max_pred_horizon * num_ohlcv_features, reshape it first.
            if len(action.shape) == 1 and action.size == self.max_pred_horizon * self.num_ohlcv_features:
                action = action.reshape(self.max_pred_horizon, self.num_ohlcv_features)
            elif len(action.shape) == 2 and action.shape[0] == self.max_pred_horizon and action.shape[1] == self.num_ohlcv_features:
                pass # Correct shape already
            else:
                # This is an unexpected shape, log error and terminate episode with high penalty
                print(f"Critical Error: Unhandled action shape {action.shape} in step.")
                observation = np.zeros(self.observation_space.shape, dtype=np.float32)
                reward = -1000
                terminated = True
                truncated = False
                info = {'error': 'Critical action shape mismatch', 'action_shape': str(action.shape)}
                return observation, reward, terminated, truncated, info

        # Take the slice relevant to the current prediction horizon
        action_for_current_horizon = action[:self.current_pred_horizon_days, :]

        try:
            # The action_for_current_horizon should now have shape (self.current_pred_horizon_days, self.num_ohlcv_features)
            predicted_ohlcv_norm = action_for_current_horizon
            if predicted_ohlcv_norm.shape != (self.current_pred_horizon_days, self.num_ohlcv_features):
                # This should not happen if slicing is correct
                raise ValueError(f"Internal reshape error. Shape after slice is {predicted_ohlcv_norm.shape}")
        except ValueError as e:
            print(f"Error reshaping action in step: {e}. Action shape: {action.shape}, Expected: ({self.current_pred_horizon_days}, {self.num_ohlcv_features})")
            # This is a critical error, likely a mismatch between action space def and policy output
            # Return a very low reward and terminate to signal a problem.
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = -1000
            terminated = True
            truncated = False
            info = {'error': 'Action reshape failed', 'action_shape': str(action.shape)}
            return observation, reward, terminated, truncated, info

        # Actual OHLCV sequence starts from self.current_step (day after last observed day)
        actual_ohlcv_raw = self._get_target_ohlcv_sequence(start_idx=self.current_step, length=self.current_pred_horizon_days)

        if actual_ohlcv_raw is None:
            # This means we don't have enough future data to evaluate the action for the full horizon.
            # This should ideally lead to termination. The reward function handles None actuals.
            # print(f"Step {self.current_step} for {self.current_company_symbol}: Actual OHLCV is None. Terminating.")
            reward = self._calculate_reward(predicted_ohlcv_norm, None) # Should be a large penalty
            terminated = True
            truncated = False
            observation = self._get_observation() # Get obs for the current state before termination
            info = self._get_info()
            info['status'] = 'terminated_no_future_data_for_reward'
            return observation, reward, terminated, truncated, info

        reward = self._calculate_reward(predicted_ohlcv_norm, actual_ohlcv_raw)
        
        # Advance time by 1 day. The agent makes a prediction for a future *window* at each *daily* step.
        self.current_step += 1 

        terminated = False
        # Termination condition: Can we form a valid observation AND a valid target sequence for the *next* step?
        # 1. Can we form an observation? Need self.current_step >= self.lookback_window
        # 2. Can we get a target sequence? Need self.current_step + self.current_pred_horizon_days <= len(self.current_data)
        
        # If, after incrementing current_step, we can't get a full target sequence for the *current* prediction horizon,
        # then the episode must end.
        if self.current_step + self.current_pred_horizon_days > len(self.current_data):
            terminated = True
            # print(f"Terminating: Not enough data for next target. Current step {self.current_step}, horizon {self.current_pred_horizon_days}, data len {len(self.current_data)}")
        
        # Also, if we can't even form a lookback window for the next observation, terminate.
        # This check (self.current_step < self.lookback_window) should ideally not be hit if reset is correct
        # and we start with current_step >= lookback_window. But as a safeguard:
        if self.current_step < self.lookback_window and not terminated:
            # This implies an issue, possibly with very short data or faulty logic.
            print(f"Warning: current_step ({self.current_step}) < lookback_window ({self.lookback_window}) in step. Terminating.")
            terminated = True
            reward -= 50 # Additional penalty for unexpected state

        truncated = False # Not using truncation based on max episode steps for now
        
        if terminated:
            # If terminated, the observation returned is for the state *before* termination, 
            # but it might not be used by the learning algorithm if done=True.
            # Some algorithms might use the last observation for value estimation.
            # Let's provide the observation from the current (now terminal) state.
            observation = self._get_observation() 
        else:
            observation = self._get_observation()
            
        info = self._get_info()
        if terminated:
            info['status'] = info.get('status', 'terminated_end_of_company_data')
            # print(f"Episode terminated for {self.current_company_symbol} at step {self.current_step}. Reward: {reward:.2f}")

        return observation, reward, terminated, truncated, info

    def render(self):
        # For now, no specific rendering is implemented.
        # Could be used to plot stock prices, trades, portfolio value, etc.
        print(f"Company: {self.current_company_symbol}, Step: {self.current_step}, Horizon: {self.current_pred_horizon_days} days")

    def close(self):
        # No specific resources to clean up in this version.
        pass

    def _get_info(self):
        # Provides auxiliary information about the environment's state.
        # Useful for debugging or logging.
        return {
            'current_company_symbol': self.current_company_symbol,
            'current_step': self.current_step,
            'current_pred_horizon_days': self.current_pred_horizon_days,
            'episode_start_step': self.episode_start_step,
            'total_data_length_for_company': len(self.current_data) if self.current_data is not None else 0
        }

# --- Main Training Script V2 ---
if __name__ == '__main__':
    print("Starting RL Stock Predictor V2 Training...")

    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    data_dict, financial_feature_names_used = load_and_preprocess_data_v2(
        COMPANY_SYMBOLS,
        STOCK_DATA_DIR,
        PROCESSED_FINANCIAL_DATA_DIR, # Corrected from FINANCIAL_DATA_DIR to PROCESSED_FINANCIAL_DATA_DIR
        SENTIMENT_DATA_DIR
    )

    if not data_dict:
        print("No data loaded. Exiting training.")
        exit()
    
    print(f"Data loaded for {len(data_dict)} companies.")
    print(f"Financial features used: {financial_feature_names_used}")

    # 2. Create StockTradingEnvV2 instance
    print("Creating PPO training environment...")
    # Check if any company has enough data before creating env
    valid_companies_for_env = []
    for sym, df_company in data_dict.items():
        # Check against the shortest horizon and lookback
        min_len_needed = HISTORICAL_LOOKBACK_DAYS + PREDICTION_HORIZONS['short']
        if len(df_company) >= min_len_needed:
            valid_companies_for_env.append(sym)
        else:
            print(f"Excluding {sym} from environment: data length {len(df_company)} < min required {min_len_needed}")

    if not valid_companies_for_env:
        print("No companies have sufficient data for training after preprocessing. Exiting.")
        exit()
    
    # Create a filtered data_dict for the environment
    env_data_dict = {s: data_dict[s] for s in valid_companies_for_env}

    env_lambda = lambda: StockTradingEnvV2(env_data_dict, financial_feature_names_used, 
                                           HISTORICAL_LOOKBACK_DAYS, PREDICTION_HORIZONS)
    env = DummyVecEnv([env_lambda])
    
    # Save scalers from the environment *before* wrapping with VecNormalize
    original_env_instance = env.envs[0]
    scalers = {
        'observation_scaler': original_env_instance.observation_scaler,
        'action_ohlcv_scaler': original_env_instance.action_ohlcv_scaler
    }
    os.makedirs(TRAINED_MODEL_DIR_V2, exist_ok=True)
    scaler_save_path = os.path.join(TRAINED_MODEL_DIR_V2, "scalers_v2.pkl")
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scalers, f) # Use pickle for scalers
    print(f"Scalers saved to {scaler_save_path}")

    # 3. Wrap with VecNormalize for SB3 compatibility
    # Important: norm_obs=False because our environment's _get_observation already returns scaled data.
    # We only want VecNormalize to handle reward normalization and potentially observation clipping if desired.
    vec_env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.) 
    print("Environment wrapped with VecNormalize.")

    # 4. Define and train the RL model (e.g., PPO from stable-baselines3)
    model_save_path = os.path.join(TRAINED_MODEL_DIR_V2, "ppo_stock_predictor_v2.zip")
    vec_normalize_stats_path = os.path.join(TRAINED_MODEL_DIR_V2, "vec_normalize_v2.pkl")
    tensorboard_log_dir = "./ppo_stock_tensorboard_v2/"
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    # LOAD_PRETRAINED_MODEL_IF_EXISTS is now a global constant

    # Check if a pre-trained model exists
    if os.path.exists(model_save_path) and LOAD_PRETRAINED_MODEL_IF_EXISTS:
        print(f"Loading pre-trained model from {model_save_path}")
        # When loading a model, also load the VecNormalize stats if they exist
        if os.path.exists(vec_normalize_stats_path):
            print(f"Loading VecNormalize stats from {vec_normalize_stats_path}")
            vec_env = VecNormalize.load(vec_normalize_stats_path, vec_env)
            vec_env.training = True # Set to training mode
            print("VecNormalize stats loaded and environment updated.")
        else:
            print(f"Warning: VecNormalize stats not found at {vec_normalize_stats_path}. Model performance may be affected.")
        model = PPO.load(model_save_path, env=vec_env, tensorboard_log=tensorboard_log_dir) # Use vec_env
        print("Pre-trained model loaded.")
    else:
        print("Creating a new PPO model...")
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
        model = PPO(
            "MlpPolicy", 
            vec_env, # Use vec_env
            verbose=1, 
            tensorboard_log=tensorboard_log_dir,
            policy_kwargs=policy_kwargs,
            learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
            max_grad_norm=0.5
        )
        print("New PPO model created.")

    total_timesteps_to_train = 200000 # Increased for potentially better learning
    save_freq = 50000
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=TRAINED_MODEL_DIR_V2, name_prefix='ppo_stock_v2_checkpoint')
    
    print(f"Starting training for {total_timesteps_to_train} timesteps...")
    try:
        model.learn(
            total_timesteps=total_timesteps_to_train, 
            callback=checkpoint_callback, # Use checkpoint callback
            tb_log_name="PPO_V2_StockRun",
            reset_num_timesteps=not (os.path.exists(model_save_path) and LOAD_PRETRAINED_MODEL_IF_EXISTS)
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        # 5. Save the final trained model and VecNormalize stats
        model.save(model_save_path)
        print(f"Final model saved to {model_save_path}")
        if isinstance(vec_env, VecNormalize):
            vec_env.save(vec_normalize_stats_path)
            print(f"VecNormalize stats saved to {vec_normalize_stats_path}")
        print(f"TensorBoard logs are in {tensorboard_log_dir}")
        print(f"To view TensorBoard, run: tensorboard --logdir={os.path.abspath(tensorboard_log_dir)}")

# --- Prediction Function V2 ---
def predict_for_new_data_v2(company_symbol_to_predict, 
                            model_load_path, 
                            scaler_load_path, 
                            vec_normalize_load_path, # Added path for VecNormalize stats
                            # Optional: Pass new data directly to avoid re-loading within the function
                            stock_data_input_df=None, 
                            financial_data_input_df=None, 
                            sentiment_data_input_df=None,
                            prediction_horizon_key='short', # 'short' or 'long'
                            output_csv_path=None):
    """
    Predicts future stock prices.
    If stock_data_input_df, financial_data_input_df, sentiment_data_input_df are None,
    it will attempt to load the latest data for company_symbol_to_predict.
    """
    print(f"--- Starting Prediction for {company_symbol_to_predict} --- ")    
    if not all(os.path.exists(p) for p in [model_load_path, scaler_load_path, vec_normalize_load_path]):
        print("Error: Model, scaler, or VecNormalize file not found.")
        if not os.path.exists(model_load_path): print(f"Missing: {model_load_path}")
        if not os.path.exists(scaler_load_path): print(f"Missing: {scaler_load_path}")
        if not os.path.exists(vec_normalize_load_path): print(f"Missing: {vec_normalize_load_path}")
        return None

    # 1. Load model, scalers, and VecNormalize stats
    try:
        model = PPO.load(model_load_path)
        with open(scaler_load_path, 'rb') as f:
            scalers = pickle.load(f) # Use pickle
        observation_scaler = scalers['observation_scaler']
        action_ohlcv_scaler = scalers['action_ohlcv_scaler']

        num_total_obs_features = observation_scaler.n_features_in_
        num_ohlcv_sentiment_features = len(OHLCV_COLUMNS) + len(SENTIMENT_FEATURES_NAMES)
        num_financial_features = num_total_obs_features - num_ohlcv_sentiment_features
        # These dummy names must match how the observation_scaler was fitted during training env setup
        dummy_financial_features_for_env = [f'fin_feat_{i}' for i in range(num_financial_features)]

        # Create a dummy environment instance to correctly load VecNormalize stats
        # The parameters for this dummy env should match those used during training for observation space consistency
        dummy_env_instance_for_norm = StockTradingEnvV2(
            data_dict={company_symbol_to_predict: pd.DataFrame()}, # Minimal data_dict
            financial_feature_names=dummy_financial_features_for_env, # Crucial for obs space
            lookback_window=HISTORICAL_LOOKBACK_DAYS, 
            pred_horizons=PREDICTION_HORIZONS
        )
        vec_env_for_norm_load = DummyVecEnv([lambda: dummy_env_instance_for_norm])
        vec_env_for_norm_load = VecNormalize.load(vec_normalize_load_path, vec_env_for_norm_load)
        vec_env_for_norm_load.training = False # Set to inference mode
        vec_env_for_norm_load.norm_reward = False # Not needed for prediction

    except Exception as e:
        print(f"Error loading model/scalers/VecNormalize: {e}"); return None
    print("Model, scalers, and VecNormalize stats loaded.")

    # 2. Load and preprocess LATEST data for the observation window
    # This part needs to fetch the most recent HISTORICAL_LOOKBACK_DAYS data
    # Or use provided dataframes
    print(f"Loading latest data for {company_symbol_to_predict} for lookback (if not provided)... ")
    if stock_data_input_df is None:
        stock_df = load_stock_data_v2(company_symbol_to_predict, STOCK_DATA_DIR)
    else:
        stock_df = stock_data_input_df.copy()
    
    fin_feats_actual = [] # To store actual financial feature names from loaded data
    if financial_data_input_df is None:
        financial_df, fin_feats_actual = load_financial_data_v2(company_symbol_to_predict, PROCESSED_FINANCIAL_DATA_DIR)
    else:
        financial_df = financial_data_input_df.copy()
        # Infer fin_feats_actual if financial_df is provided and not empty
        if not financial_df.empty:
            fin_feats_actual = [col for col in financial_df.columns if col not in ['Date', 'symbol'] and 'symbol' not in col.lower() and 'Unnamed' not in col]

    if sentiment_data_input_df is None:
        sentiment_df = load_sentiment_data_v2(company_symbol_to_predict, SENTIMENT_DATA_DIR)
    else:
        sentiment_df = sentiment_data_input_df.copy()

    if stock_df.empty or len(stock_df) < HISTORICAL_LOOKBACK_DAYS:
        print(f"Not enough stock data for {company_symbol_to_predict} for lookback ({len(stock_df)} days). Prediction aborted.")
        return None

    # Combine into a single DataFrame, similar to all_companies_data[symbol] in training preprocessing
    # Prepare the observation dataframe using the dummy_financial_features_for_env for column structure
    # but populate with actual data from fin_feats_actual where available.
    
    # Start with the latest stock data (last HISTORICAL_LOOKBACK_DAYS rows)
    obs_df_base = stock_df.iloc[-HISTORICAL_LOOKBACK_DAYS:][OHLCV_COLUMNS].copy()
    if obs_df_base.shape[0] < HISTORICAL_LOOKBACK_DAYS:
        print(f"Not enough base stock data rows ({obs_df_base.shape[0]}) for lookback {HISTORICAL_LOOKBACK_DAYS}. Aborting.")
        return None
    current_obs_dates = obs_df_base.index # Dates for alignment

    # Add financial features, aligning with dummy_financial_features_for_env
    # fin_feats_actual contains names from loaded financial_df
    # dummy_financial_features_for_env contains generic names like 'fin_feat_0', 'fin_feat_1', ...
    # We need to map actual features to these dummy placeholders if the scaler was trained on them.
    # For simplicity here, we assume the scaler was trained with features named 'fin_feat_0', etc.
    # and the financial_df columns are the actual features that need to be mapped or selected.
    # This part might need careful adjustment based on how financial_feature_names_used was defined in training.
    # Assuming dummy_financial_features_for_env are the columns the scaler expects.

    aligned_financial_df = pd.DataFrame(index=current_obs_dates)
    if not financial_df.empty:
        # Ensure financial_df has 'Date' index for reindexing
        temp_financial_df = financial_df.copy()
        if 'Date' not in temp_financial_df.index.name:
             if 'Date' in temp_financial_df.columns: temp_financial_df.set_index('Date', inplace=True)
             else: print("Warning: Financial data has no 'Date' index or column for alignment.")
        
        if 'Date' in temp_financial_df.index.name:
            reindexed_fin = temp_financial_df.reindex(current_obs_dates, method='ffill')
            # Map actual financial features to the dummy ones if necessary
            # For now, let's assume the first N features from fin_feats_actual map to dummy_financial_features_for_env
            for i, dummy_col_name in enumerate(dummy_financial_features_for_env):
                if i < len(fin_feats_actual) and fin_feats_actual[i] in reindexed_fin.columns:
                    aligned_financial_df[dummy_col_name] = reindexed_fin[fin_feats_actual[i]]
                else:
                    aligned_financial_df[dummy_col_name] = 0.0 # Default if not enough actual features or mismatch
        else: # Financial data could not be aligned
            for dummy_col_name in dummy_financial_features_for_env:
                aligned_financial_df[dummy_col_name] = 0.0
    else: # No financial data
        for dummy_col_name in dummy_financial_features_for_env:
            aligned_financial_df[dummy_col_name] = 0.0
    obs_df_base = pd.concat([obs_df_base, aligned_financial_df.fillna(0.0)], axis=1)

    # Add sentiment features
    if not sentiment_df.empty:
        temp_sentiment_df = sentiment_df.copy()
        if 'Date' not in temp_sentiment_df.index.name:
            if 'Date' in temp_sentiment_df.columns: temp_sentiment_df.set_index('Date', inplace=True)
            else: print("Warning: Sentiment data has no 'Date' index or column for alignment.")
        
        if 'Date' in temp_sentiment_df.index.name:
            reindexed_sentiment = temp_sentiment_df.reindex(current_obs_dates, method='ffill')
            obs_df_base['SentimentScore'] = reindexed_sentiment.get('SentimentScore', 0.0)
        else:
            obs_df_base['SentimentScore'] = 0.0
    else:
        obs_df_base['SentimentScore'] = 0.0
    obs_df_base['SentimentScore'].fillna(0.0, inplace=True)

    # Ensure final column order and fill any remaining NaNs (e.g., from OHLCV if bfill didn't cover all)
    final_ordered_cols_for_obs = OHLCV_COLUMNS + SENTIMENT_FEATURES_NAMES + dummy_financial_features_for_env
    obs_input_df = obs_df_base.reindex(columns=final_ordered_cols_for_obs, fill_value=0.0).fillna(0.0)

    if obs_input_df.shape[0] != HISTORICAL_LOOKBACK_DAYS:
        print(f"Error: Observation DataFrame row count mismatch. Expected: {HISTORICAL_LOOKBACK_DAYS}, Got: {obs_input_df.shape[0]}")
        return None

    if obs_input_df.shape[0] != HISTORICAL_LOOKBACK_DAYS or obs_input_df.shape[1] != num_total_obs_features:
        print(f"Error: Observation DataFrame shape mismatch. Expected: ({HISTORICAL_LOOKBACK_DAYS}, {num_total_obs_features}), Got: {obs_input_df.shape}")
        return None

    # 3. Construct observation vector
    raw_observation_features = obs_input_df.values
    scaled_observation_features = observation_scaler.transform(raw_observation_features)
    
    pred_horizon_days_value = PREDICTION_HORIZONS.get(prediction_horizon_key)
    if pred_horizon_days_value is None:
        print(f"Invalid prediction_horizon_key: {prediction_horizon_key}. Use 'short' or 'long'."); return None
        
    horizon_indicator = 0.0 if pred_horizon_days_value == PREDICTION_HORIZONS['short'] else 1.0
    single_obs_flat = np.concatenate((scaled_observation_features.flatten(), np.array([horizon_indicator], dtype=np.float32)))
    
    # The observation_scaler expects features in the order: OHLCV, Sentiment, Financial_feats
    # The environment's observation space also includes the horizon_indicator at the end.
    
    final_obs_flat_before_vecnorm = np.concatenate((scaled_observation_features.flatten(), np.array([horizon_indicator], dtype=np.float32)))
    final_obs_reshaped_for_vecnorm = final_obs_flat_before_vecnorm.reshape(1, -1)
    
    # Normalize the observation using the loaded VecNormalize stats
    # vec_env_for_norm_load.normalize_obs expects a batch, so reshape
    normalized_obs_for_model = vec_env_for_norm_load.normalize_obs(final_obs_reshaped_for_vecnorm)

    # 4. Use model.predict()
    action, _states = model.predict(normalized_obs, deterministic=True)
    predicted_ohlcv_norm = action.reshape(pred_horizon_days_value, len(OHLCV_COLUMNS))

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

    print(f"--- Prediction for {company_symbol_to_predict} Complete --- ")
    return predictions_df

# --- Example Usage of Prediction Function (Illustrative) ---
# def run_example_prediction():
#     print("\n--- Running Example Prediction ---")
#     model_p = os.path.join(TRAINED_MODEL_DIR_V2, "ppo_stock_predictor_v2.zip")
#     scalers_p = os.path.join(TRAINED_MODEL_DIR_V2, "scalers_v2.pkl")
#     vec_norm_p = os.path.join(TRAINED_MODEL_DIR_V2, "vec_normalize_v2.pkl") 

#     if not all(os.path.exists(p) for p in [model_p, scalers_p, vec_norm_p]):
#         print("Trained model, scalers, or VecNormalize stats not found. Please train the model first.")
#         if not os.path.exists(model_p): print(f"Missing model: {model_p}")
#         if not os.path.exists(scalers_p): print(f"Missing scalers: {scalers_p}")
#         if not os.path.exists(vec_norm_p): print(f"Missing VecNormalize: {vec_norm_p}")
#         return

#     example_symbol = 'AAPL'
#     if not COMPANY_SYMBOLS or example_symbol not in COMPANY_SYMBOLS:
#         example_symbol = COMPANY_SYMBOLS[0] if COMPANY_SYMBOLS else 'MSFT' # Fallback
#         print(f"Using {example_symbol} for example prediction.")

#     # PREDICTION_OUTPUT_DIR is now a global constant and created at the start
#     # os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True) # Not needed here anymore

#     predicted_df_short = predict_for_new_data_v2(
#         company_symbol_to_predict=example_symbol,
#         model_load_path=model_p,
#         scaler_load_path=scalers_p,
#         vec_normalize_load_path=vec_norm_p, 
#         prediction_horizon_key='short',
#         output_csv_path=os.path.join(PREDICTION_OUTPUT_DIR, f"{example_symbol}_short_term_prediction_v2.csv")
#     )
#     if predicted_df_short is not None: print(f"\nShort-term predictions for {example_symbol}:\n{predicted_df_short}")

#     predicted_df_long = predict_for_new_data_v2(
#         company_symbol_to_predict=example_symbol,
#         model_load_path=model_p,
#         scaler_load_path=scalers_p,
#         vec_normalize_load_path=vec_norm_p, 
#         prediction_horizon_key='long',
#         output_csv_path=os.path.join(PREDICTION_OUTPUT_DIR, f"{example_symbol}_long_term_prediction_v2.csv")
#     )
#     if predicted_df_long is not None: print(f"\nLong-term predictions for {example_symbol}:\n{predicted_df_long}")


# if __name__ == '__main__':
#    # run_example_prediction() # Call after training if desired
    pass