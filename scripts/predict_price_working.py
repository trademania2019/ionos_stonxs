import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import os
import requests
import logging
import time
from joblib import Parallel, delayed


os.environ['PYTHONIOENCODING'] = 'utf-8'

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Extract command-line arguments
symbol = sys.argv[1]
interval = sys.argv[2]
user_id = sys.argv[3]

logging.info(f"Running predict_price_working.py with symbol={symbol} and interval={interval}")
# Define constants and function to fetch data
DATA_DIR = os.path.join('C:\\Flask2', 'data', user_id)
os.makedirs(DATA_DIR, exist_ok=True)
# Define constants and function to fetch data
DURATION_MAPPING = {
    '1min': '1',
    '3min': '3',
    '5min': '5',
    '15min': '15',
    '30min': '30',
    '45min': '45',
    '1H': '1H',
    '1D': '1D',
}
API_TOKEN = os.getenv('API_TOKEN', 'bjhWSElQRk1yRHdVV2xNNERtSVQwR0tGcWI5aElLdEp6MWtzX0tKUnVwYz0')

def get_stock_data(symbol, interval, days=7):
    start_time = time.time()
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    api_interval = DURATION_MAPPING.get(interval)
    if not api_interval:
        raise ValueError(f"Unsupported interval. Supported intervals are: {list(DURATION_MAPPING.keys())}")

    url = f"https://api.marketdata.app/v1/stocks/candles/{api_interval}/{symbol}?from={start_date_str}&to={end_date_str}&token={API_TOKEN}"
    logging.info(f"Requesting URL: {url}")
    response = requests.get(url, headers={'Accept': 'application/json'})

    if response.status_code == 200:
        logging.info("Data fetched successfully")
        logging.info(f"Time to fetch data: {time.time() - start_time} seconds")
        return response.json()
    else:
        logging.error(f"Failed to retrieve data: {response.status_code} {response.text}")
        raise Exception(f"Failed to retrieve data: {response.status_code} {response.text}")

def process_interval(interval):
    days = 60 if interval == '1D' else 7  # Adjust days to fetch based on interval
    # Fetch the data
    try:
        start_time = time.time()
        data = get_stock_data(symbol, interval, days=days)
        logging.info(f"Data fetched for symbol={symbol}, interval={interval}")
        logging.info(f"Time to fetch stock data: {time.time() - start_time} seconds")
    except Exception as e:
        logging.error(f"Error fetching data for interval {interval}: {e}")
        return

    # Function to compute RSI
    def compute_rsi(data, window):
        start_time = time.time()
        logging.info(f"Computing RSI with window: {window}")
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        logging.info(f"Time to compute RSI: {time.time() - start_time} seconds")
        return rsi

    # Function to compute Bollinger Bands
    def compute_bollinger_bands(data, window, num_std_dev):
        start_time = time.time()
        logging.info(f"Computing Bollinger Bands with window: {window}, num_std_dev: {num_std_dev}")
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        bollinger_high = rolling_mean + (rolling_std * num_std_dev)
        bollinger_low = rolling_mean - (rolling_std * num_std_dev)
        logging.info(f"Time to compute Bollinger Bands: {time.time() - start_time} seconds")
        return bollinger_high, bollinger_low

    # Function to compute MACD
    def compute_macd(data, slow, fast, signal):
        start_time = time.time()
        logging.info(f"Computing MACD with slow: {slow}, fast: {fast}, signal: {signal}")
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        logging.info(f"Time to compute MACD: {time.time() - start_time} seconds")
        return macd, macd_signal

    # Convert to DataFrame
    if data['s'] == 'ok':
        start_time = time.time()
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['t'], unit='s'),
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v']
        })
        df.set_index('timestamp', inplace=True)
        if interval == '1D':
            df.index = pd.DatetimeIndex(df.index).to_period('D')
        else:
            df.index = pd.DatetimeIndex(df.index).to_period('min')  # Use 'T' for minute-level frequency

        # Compute technical indicators
        df['RSI'] = compute_rsi(df['close'], window=14)
        df['BB_high'], df['BB_low'] = compute_bollinger_bands(df['close'], window=20, num_std_dev=2)
        df['MACD'], df['MACD_signal'] = compute_macd(df['close'], slow=26, fast=12, signal=9)
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['Delta_VWAP'] = df['close'] - df['VWAP']
        logging.info("Technical indicators computed.")
        logging.info(f"Time to convert to DataFrame and compute indicators: {time.time() - start_time} seconds")
    else:
        logging.error(f"Data not in expected format for interval {interval}")
        return

    # Filter the DataFrame to include only the last two weeks of data
    start_time = time.time()
    two_weeks_ago = df.index[-1].to_timestamp() - timedelta(days=14)
    df = df[df.index.to_timestamp() >= two_weeks_ago]
    logging.info(f"Time to filter DataFrame: {time.time() - start_time} seconds")

    # Drop rows with NaN values
    start_time = time.time()
    df.dropna(inplace=True)
    logging.info(f"Time to drop NaN values: {time.time() - start_time} seconds")

    # Prepare the dataset for regression models
    start_time = time.time()
    features = ['open', 'high', 'low', 'close', 'RSI', 'BB_high', 'BB_low', 'MACD', 'MACD_signal', 'Delta_VWAP']
    X = df[features]
    y = df['close']

    # Train-test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    logging.info(f"Time to prepare dataset and split train-test: {time.time() - start_time} seconds")

    # Scale the data
    start_time = time.time()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logging.info(f"Time to scale data: {time.time() - start_time} seconds")

    # Define sequence length
    sequence_length = 5

    # Define create_sequences function
    def create_sequences(data, sequence_length):
        start_time = time.time()
        logging.info(f"Creating sequences with length: {sequence_length}")
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
        logging.info(f"Time to create sequences: {time.time() - start_time} seconds")
        return np.array(sequences)

    # Parallel processing for training models
    def evaluate_model(model, X_train, y_train, X_test, y_test):
        start_time = time.time()
        logging.info(f"Training model: {model.__class__.__name__}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"Model {model.__class__.__name__} MSE: {mse}")
        logging.info(f"Time to train and evaluate model {model.__class__.__name__}: {time.time() - start_time} seconds")
        return mse, model

    # Function to train ARIMA
    def evaluate_arima(X_train, y_train, X_test, y_test):
        start_time = time.time()
        logging.info(f"Training ARIMA model")
        model = ARIMA(y_train, order=(5, 1, 0))
        model_fit = model.fit(method_kwargs={"disp": 0})
        y_pred = model_fit.forecast(steps=len(y_test))
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"ARIMA model MSE: {mse}")
        logging.info(f"Time to train and evaluate ARIMA model: {time.time() - start_time} seconds")
        return mse, model_fit

    # Function to train GARCH
    def evaluate_garch(X_train, y_train, X_test, y_test):
        start_time = time.time()
        logging.info(f"Training GARCH model")
        model = arch_model(y_train, vol='Garch', p=1, q=1)
        model_fit = model.fit(disp='off')
        
        # Print the forecast structure for debugging
        forecast = model_fit.forecast(horizon=len(y_test))
        logging.info(f"GARCH forecast: {forecast}")
        
        # Access the correct forecast values
        try:
            y_pred = forecast.mean.iloc[-1].values
        except KeyError as e:
            logging.error(f"Key error in GARCH forecast: {e}")
            raise

        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"GARCH model MSE: {mse}")
        logging.info(f"Time to train and evaluate GARCH model: {time.time() - start_time} seconds")
        return mse, model_fit

    # Define LSTM model using PyTorch
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, output_size)  # * 2 for bidirectional

        def forward(self, x):
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # * 2 for bidirectional
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_lstm_model(X_train, y_train, X_test, y_test):
        start_time = time.time()
        X_train_seq = create_sequences(scaler.fit_transform(X_train), sequence_length)
        y_train_seq = y_train[sequence_length:]
        X_test_seq = create_sequences(scaler.transform(X_test), sequence_length)
        y_test_seq = y_test[sequence_length:]

        # Check if there is enough data to train the LSTM model
        if len(X_train_seq) == 0 or len(y_train_seq) == 0 or len(X_test_seq) == 0 or len(y_test_seq) == 0:
            logging.warning("Not enough data to train LSTM model.")
            return float('inf'), None

        model = LSTMModel(input_size=X_train_seq.shape[2], hidden_size=50, num_layers=2, output_size=1).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        X_train_seq_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
        y_train_seq_tensor = torch.tensor(y_train_seq.values, dtype=torch.float32).to(device).view(-1, 1)

        model.train()
        for epoch in range(10):  # Train for 10 epochs
            outputs = model(X_train_seq_tensor)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train_seq_tensor)
            loss.backward()
            optimizer.step()

        X_test_seq_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_seq_tensor).cpu().numpy()

        mse = mean_squared_error(y_test_seq, y_pred)
        logging.info(f"LSTM model MSE: {mse}")
        logging.info(f"Time to train and evaluate LSTM model: {time.time() - start_time} seconds")
        return mse, model

    # Train models in parallel
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_model)(model, X_train_scaled, y_train, X_test_scaled, y_test)
        for model in [
            LinearRegression(),
            DecisionTreeRegressor(),
            RandomForestRegressor(),
            GradientBoostingRegressor(),
        ]
    ) + [
        evaluate_arima(X_train, y_train, X_test, y_test),
        evaluate_garch(X_train, y_train, X_test, y_test),
        create_lstm_model(X_train, y_train, X_test, y_test)
    ]

    # Select the best model
    start_time = time.time()
    model_mse = {result[1].__class__.__name__ if isinstance(result[1], nn.Module) else result[1].__class__.__name__: result[0] for result in results}
    best_model_name = min(model_mse, key=model_mse.get)
    logging.info(f"Best model: {best_model_name} with MSE: {model_mse[best_model_name]}")
    logging.info(f"Time to select best model: {time.time() - start_time} seconds")

    # Use the best model to make predictions
    start_time = time.time()
    best_model = {result[1].__class__.__name__ if isinstance(result[1], nn.Module) else result[1].__class__.__name__: result[1] for result in results}[best_model_name]

    # Predict the next intervals
    if best_model_name == 'LSTMModel':
        # LSTM specific prediction
        last_sequence = scaler.transform(df[features])[-5:]
        predicted_ohlc = []
        for _ in range(10):
            last_sequence_reshaped = np.reshape(last_sequence, (1, sequence_length, len(features)))
            last_sequence_tensor = torch.tensor(last_sequence_reshaped, dtype=torch.float32).to(device)
            with torch.no_grad():
                pred = best_model(last_sequence_tensor).cpu().numpy()
            predicted_ohlc.append(pred[0])
            last_sequence = np.append(last_sequence[1:], [pred[0]], axis=0)
    elif best_model_name in ['ARIMA', 'GARCH']:
        if best_model_name == 'ARIMA':
            predicted_ohlc = best_model.forecast(steps=10)
        else:
            predicted_ohlc = best_model.forecast(horizon=10).mean.iloc[-1].values
    else:
        # Non-LSTM prediction
        last_sequence = scaler.transform(df[features].iloc[-10:])
        predicted_ohlc = best_model.predict(last_sequence)
        predicted_ohlc = np.array(predicted_ohlc).reshape(-1, 1)  # Ensure the shape is consistent for DataFrame creation

    # Inverse transform the predictions if LSTM
    if best_model_name == 'LSTMModel':
        predicted_ohlc = scaler.inverse_transform(predicted_ohlc)

    # Prepare the prediction DataFrame
    if best_model_name == 'LSTMModel':
        predicted_ohlc_df = pd.DataFrame(predicted_ohlc, columns=features[:len(predicted_ohlc[0])])
    else:
        predicted_ohlc_df = pd.DataFrame(predicted_ohlc, columns=['close'])
        predicted_ohlc_df['open'] = predicted_ohlc_df['close']
        predicted_ohlc_df['high'] = predicted_ohlc_df['close']
        predicted_ohlc_df['low'] = predicted_ohlc_df['close']

    predicted_ohlc_df['volume'] = 0

    # Create future dates for the prediction
    if interval == '1D':
        future_dates = pd.date_range(start=df.index[-1].to_timestamp(), periods=11, freq='D')[1:]
    else:
        future_dates = pd.date_range(start=df.index[-1].to_timestamp(), periods=11, freq='15min')[1:]
    predicted_ohlc_df['timestamp'] = future_dates
    predicted_ohlc_df.set_index('timestamp', inplace=True)

    # Combine actual and future data
    combined_df = pd.concat([df, predicted_ohlc_df[['open', 'high', 'low', 'close']]])
    logging.info(f"Time to make predictions and prepare prediction DataFrame: {time.time() - start_time} seconds")

    # Create a table of candlestick data and round to 2 decimal places
    start_time = time.time()
    candlestick_data = pd.DataFrame({
        'Date': list(df.index.to_timestamp()) + list(predicted_ohlc_df.index),
        'Open': list(df['open']) + list(predicted_ohlc_df['open']),
        'High': list(df['high']) + list(predicted_ohlc_df['high']),
        'Low': list(df['low']) + list(predicted_ohlc_df['low']),
        'Close': list(df['close']) + list(predicted_ohlc_df['close']),
        'Volume': list(df['volume']) + list(predicted_ohlc_df['volume'])
    }).round(2)

    # Save the table to a CSV file
    #candlestick_data.to_csv(f'candlestick_data_{interval}.csv', index=False)
    #logging.info(f"Candlestick data saved to candlestick_data_{interval}.csv.")
    #logging.info(f"Time to create and save candlestick data: {time.time() - start_time} seconds")
#     Save the table to a CSV file
    candlestick_data.to_csv(os.path.join(DATA_DIR, f'{user_id}_candlestick_data_{interval}.csv'), index=False)
    logging.info(f"Candlestick data saved to {os.path.join(DATA_DIR, f'{user_id}_candlestick_data_{interval}.csv')}.")


    # Determine the trend by comparing the average close prices over the last 5 actual intervals and the next 5 predicted intervals
    start_time = time.time()
    average_actual_close = round(df['close'].iloc[-5:].mean(), 2)
    expected_high = round(predicted_ohlc_df['high'].max(), 2)
    expected_low = round(predicted_ohlc_df['low'].min(), 2)
    expected_close = round(predicted_ohlc_df['close'].iloc[-1], 2)

    if expected_close > average_actual_close:
        trend = "Trend Up"
    elif expected_close < average_actual_close:
        trend = "Trend Down"
    else:
        trend = "Trend Flat"
    logging.info(f"Time to determine trend: {time.time() - start_time} seconds")

# Save the trend and additional information to a text file
    trend_file_path = os.path.join(DATA_DIR, f'{user_id}_{interval}_trend.txt')
    with open(trend_file_path, 'w') as file:
        file.write(f"Symbol: {symbol}\nInterval: {interval}\n")
        file.write(f"Actual Close price Over last 3 intervals: {average_actual_close}\n")
        file.write(f"Trend Prediction for next 3 intervals: {trend}\n")
        file.write(f"Predicted High: {expected_high}\n")
        file.write(f"Predicted Low: {expected_low}\n")
        file.write(f"Predicted Close: {expected_close}\n")
    logging.info(f"Trend information saved to {trend_file_path}")

    # Save the recommendation
    start_time = time.time()
    recommendation = "Hold"
    if trend == "Trend Up":
        recommendation = "Buy"
    elif trend == "Trend Down":
        recommendation = "Sell"

# Save the recommendation
    recommendation_file_path = os.path.join(DATA_DIR, f'{user_id}_{interval}_recommendation.txt')
    with open(recommendation_file_path, 'w') as file:
        file.write(f"Recommendation: {recommendation}\n")
        file.write(f"Symbol: {symbol}\nInterval: {interval}\n")
        file.write(f"Actual close price over last 3 intervals: {average_actual_close}\n")
        file.write(f"Trend for next 3 intervals: {trend}\n")
        file.write(f"Predicted High: {expected_high}\n")
        file.write(f"Predicted Low: {expected_low}\n")
        file.write(f"Predicted Close: {expected_close}\n")
    logging.info(f"Recommendation saved to {recommendation_file_path}")

    return best_model_name, model_mse

# Process the user-specified interval
best_model_name, model_mse = process_interval(interval)

# Display the best model to the user
print(f"Best model for {symbol} with interval {interval} is {best_model_name} with MSE: {model_mse[best_model_name]}")

# Notify the user where the results are saved
print(f"Candlestick data saved to data/candlestick_data_{interval}.csv.")
print(f"Trend information saved to data/{interval}_trend.txt.")
print(f"Recommendation saved to data/{interval}_recommendation.txt.")
