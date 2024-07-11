# Inside trades_working_monitor.py

import sys
import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Set random seeds for reproducibility
np.random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Extract command-line arguments
symbol = sys.argv[1]
interval = sys.argv[2]
trade_size = 100
user_id = sys.argv[4]

logging.info(f"Running script with symbol={symbol} and interval={interval} and trade size={trade_size}")

# Define constants and function to fetch data
DATA_DIR = os.getenv('DATA_DIR', 'data')
DATA_DIR = os.path.join(DATA_DIR, user_id)
os.makedirs(DATA_DIR, exist_ok=True)

DURATION_MAPPING = {
    '1min': '1',
    '5min': '5',
    '15min': '15',
    '60min': '60',
    '1D': '1D',
}
API_TOKEN = os.getenv('API_TOKEN')

def get_stock_data(symbol, interval):
    if interval == '1D':
        days = 360
    else:
        days = 7

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
        logging.info(f"Data successfully fetched for {symbol} at interval {interval}")
        return response.json()
    else:
        logging.error(f"Failed to retrieve data: {response.status_code} {response.text}")
        return None

# Function to compute SMI
def compute_smi(df, period=14, smooth_k=3, smooth_d=3):
    df['max_high'] = df['high'].rolling(window=period).max()
    df['min_low'] = df['low'].rolling(window=period).min()
    df['midpoint'] = (df['max_high'] + df['min_low']) / 2
    df['diff'] = df['max_high'] - df['min_low']
    df['smi_raw'] = (df['close'] - df['midpoint']) / (df['diff'] / 2) * 100
    df['smi'] = df['smi_raw'].rolling(window=smooth_k).mean()
    df['smi_signal'] = df['smi'].rolling(window=smooth_d).mean()
    return df['smi_signal']

# Function to determine long and short signals
def determine_signals(df, smi_overbought=40, smi_oversold=-40):
    df['Buy_Signal'] = (df['smi'] < smi_oversold)
    df['Sell_Signal'] = (df['smi'] > smi_overbought)
    df['Short_Sell_Signal'] = (df['smi'] > smi_overbought)
    df['Cover_Signal'] = (df['smi'] < smi_oversold)
    return df

# Function to process data for the given interval
def process_interval(symbol, interval):
    try:
        data = get_stock_data(symbol, interval)
        if data is None:
            return None
        logging.info(f"Data fetched for symbol={symbol}, interval={interval}")
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

    # Convert to DataFrame
    if data['s'] == 'ok':
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['t'], unit='s', utc=True),
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v']
        })
        df.set_index('timestamp', inplace=True)
        # Convert timestamp to EST
        df.index = df.index.tz_convert('US/Eastern')
        # Only filter for trading hours if interval is less than 1D
        if interval != '1D':
            df = df.between_time('09:30', '16:00')
    else:
        logging.error("Data not in expected format")
        return None

    logging.info(f"Computing indicators for {interval} interval")

    # Compute technical indicators
    df['smi'] = compute_smi(df)

    df = determine_signals(df)
    return df

# Process the specified interval
df = process_interval(symbol, interval)

if df is not None:
    # Save the processed data
    csv_file_path = os.path.join(DATA_DIR, f"{user_id}_{symbol}_{interval}_data.csv")
    df.to_csv(csv_file_path)
    logging.info(f"Data for {interval} interval saved to {csv_file_path}")

    # Function to evaluate the accuracy of trend predictions and record trades
    def evaluate_performance(df, interval, trade_size, symbol):
        long_trades = []
        short_trades = []
        open_positions = []

        long_entry_price = None
        short_entry_price = None
        long_entry_time = None
        short_entry_time = None

        for i in range(1, len(df)):
            # Long Trades
            if df['Buy_Signal'].iloc[i-1]:
                long_entry_price = round(df['close'].iloc[i-1], 2)
                long_entry_time = df.index[i-1]
            if df['Sell_Signal'].iloc[i-1] and long_entry_price is not None:
                exit_price = round(df['close'].iloc[i], 2)
                exit_time = df.index[i]
                profit = round((exit_price - long_entry_price) * trade_size, 2)
                status = 'Win' if profit > 0 else 'Loss'
                long_trades.append([symbol, interval, long_entry_time, long_entry_price, exit_time, exit_price, profit, status])
                long_entry_price = None  # Reset entry price after trade is closed

            # Short Trades
            if df['Short_Sell_Signal'].iloc[i-1]:
                short_entry_price = round(df['close'].iloc[i-1], 2)
                short_entry_time = df.index[i-1]
            if df['Cover_Signal'].iloc[i-1] and short_entry_price is not None:
                exit_price = round(df['close'].iloc[i], 2)
                exit_time = df.index[i]
                profit = round((short_entry_price - exit_price) * trade_size, 2)
                status = 'Win' if profit > 0 else 'Loss'
                short_trades.append([symbol, interval, short_entry_time, short_entry_price, exit_time, exit_price, profit, status])
                short_entry_price = None  # Reset entry price after trade is closed

        # Handle open positions at the end of the script execution
        if long_entry_price is not None:
            open_positions.append([symbol, interval, long_entry_time, long_entry_price, 'open', 'long'])
        if short_entry_price is not None:
            open_positions.append([symbol, interval, short_entry_time, short_entry_price, 'open', 'short'])

        long_df = pd.DataFrame(long_trades, columns=['symbol', 'Interval', 'Entry_date_time', 'Entry_price', 'Exit_date_time', 'Exit_price', 'Profit', 'Status'])
        short_df = pd.DataFrame(short_trades, columns=['symbol', 'Interval', 'Entry_date_time', 'Entry_price', 'Exit_date_time', 'Exit_price', 'Profit', 'Status'])
        open_df = pd.DataFrame(open_positions, columns=['symbol', 'Interval', 'Entry_date_time', 'Entry_price', 'Status', 'Type'])

        return long_df, short_df, open_df

    long_df, short_df, open_df = evaluate_performance(df, interval, trade_size, symbol)

    # Save trade data
    long_df.to_csv(os.path.join(DATA_DIR, f"{user_id}_{symbol}_long_trades.csv"), index=False)
    short_df.to_csv(os.path.join(DATA_DIR, f"{user_id}_{symbol}_short_trades.csv"), index=False)
    open_df.to_csv(os.path.join(DATA_DIR, f"{user_id}_{symbol}_open_positions.csv"), index=False)

    # Calculate the success rate and trade summary for the specified interval
    def calculate_success_and_profit(df):
        total_trades = len(df)
        wins = (df['Status'] == 'Win').sum()
        losses = (df['Status'] == 'Loss').sum()
        success_rate = wins / total_trades if total_trades > 0 else 0
        total_profit = df['Profit'].sum()
        return total_trades, wins, losses, success_rate, total_profit

    long_summary = calculate_success_and_profit(long_df)
    short_summary = calculate_success_and_profit(short_df)

    # Save summary to CSV file
    summary_file_path = os.path.join(DATA_DIR, f"{user_id}_{symbol}_summary.csv")
    summary_data = [
        [symbol, interval, trade_size, long_summary[0], long_summary[1], long_summary[2], f"{long_summary[3] * 100:.2f}%", f"${long_summary[4]:.2f}", "Long"],
        [symbol, interval, trade_size, short_summary[0], short_summary[1], short_summary[2], f"{short_summary[3] * 100:.2f}%", f"${short_summary[4]:.2f}", "Short"]
    ]

    summary_df = pd.DataFrame(summary_data, columns=['Symbol', 'Interval', 'Trade_size', 'Total_Trades', 'Wins', 'Losses', 'SuccessRate', 'Total_profit', 'Trade_Type'])
    summary_df.to_csv(summary_file_path, index=False)

    # Display open positions
    print("Open Positions:")
    print(open_df)
else:
    logging.error("No data available to process for the given symbol and interval.")
