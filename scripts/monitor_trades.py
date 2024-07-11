import sys
import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
import pytz
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from flask_login import UserMixin

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://stonx_app_user:securepassword@localhost/stonxs'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'stonxsnotification@gmail.com'
app.config['MAIL_PASSWORD'] = 'vzqxqkbhtpgpmwoo'

mail = Mail(app)
db = SQLAlchemy(app)

os.environ['PYTHONIOENCODING'] = 'utf-8'

# Set random seeds for reproducibility
np.random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)

DATA_DIR = os.path.join('C:\\Flask2', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

DURATION_MAPPING = {'1min': '1', '5min': '5', '15min': '15', '60min': '60'}
API_TOKEN = 'bjhWSElQRk1yRHdVV2xNNERtSVQwR0tGcWI5aElLdEp6MWtzX0tKUnVwYz0'

def get_stock_data(symbol, interval, days=1):
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

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_sma(data, window):
    return data.rolling(window=window).mean()

def compute_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def compute_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    bollinger_high = rolling_mean + (rolling_std * num_std_dev)
    bollinger_low = rolling_mean - (rolling_std * num_std_dev)
    return bollinger_high, bollinger_low

def compute_smi(df, period=14, smooth_k=3, smooth_d=3):
    df['max_high'] = df['high'].rolling(window=period).max()
    df['min_low'] = df['low'].rolling(window=period).min()
    df['midpoint'] = (df['max_high'] + df['min_low']) / 2
    df['diff'] = df['max_high'] - df['min_low']
    df['smi_raw'] = (df['close'] - df['midpoint']) / (df['diff'] / 2) * 100
    df['smi'] = df['smi_raw'].rolling(window=smooth_k).mean()
    df['smi_signal'] = df['smi'].rolling(window=smooth_d).mean()
    return df['smi_signal']

def determine_signals(df, smi_overbought=40, smi_oversold=-40):
    df['Buy_Signal'] = ((df['smi'] < smi_oversold) &
                        (df['close'] < df['BB_low']) &
                        (df['RSI'] < 30))

    df['Sell_Signal'] = ((df['smi'] > smi_overbought) &
                         (df['close'] > df['BB_high']) &
                         (df['RSI'] > 70))

    df['Short_Sell_Signal'] = ((df['smi'] > smi_overbought) &
                               (df['close'] > df['BB_high']) &
                               (df['RSI'] > 70))

    df['Cover_Signal'] = ((df['smi'] < smi_oversold) &
                          (df['close'] < df['BB_low']) &
                          (df['RSI'] < 30))
    return df

def process_interval(symbol, interval):
    try:
        data = get_stock_data(symbol, interval)
        if data is None:
            return None
        logging.info(f"Data fetched for symbol={symbol}, interval={interval}")
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

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
        df.index = df.index.tz_convert('US/Eastern')
        df = df.between_time('09:30', '16:00')
    else:
        logging.error("Data not in expected format")
        return None

    logging.info(f"Computing indicators for {interval} interval")
    df['RSI'] = compute_rsi(df['close'])
    df['SMA'] = compute_sma(df['close'], window=14)
    df['EMA'] = compute_ema(df['close'], window=14)
    df['BB_high'], df['BB_low'] = compute_bollinger_bands(df['close'])
    df['smi'] = compute_smi(df)

    df = determine_signals(df)
    return df

def evaluate_performance(df, interval, trade_size, symbol):
    long_trades = []
    short_trades = []
    open_positions = []

    long_entry_price = None
    short_entry_price = None
    long_entry_time = None
    short_entry_time = None

    for i in range(1, len(df)):
        if df['Buy_Signal'].iloc[i-1]:
            long_entry_price = round(df['close'].iloc[i-1], 2)
            long_entry_time = df.index[i-1]
        if df['Sell_Signal'].iloc[i-1] and long_entry_price is not None:
            exit_price = round(df['close'].iloc[i], 2)
            exit_time = df.index[i]
            profit = round((exit_price - long_entry_price) * trade_size, 2)
            status = 'Win' if profit > 0 else 'Loss'
            long_trades.append([symbol, interval, long_entry_time, long_entry_price, exit_time, exit_price, profit, status])
            long_entry_price = None

        if df['Short_Sell_Signal'].iloc[i-1]:
            short_entry_price = round(df['close'].iloc[i-1], 2)
            short_entry_time = df.index[i-1]
        if df['Cover_Signal'].iloc[i-1] and short_entry_price is not None:
            exit_price = round(df['close'].iloc[i], 2)
            exit_time = df.index[i]
            profit = round((short_entry_price - exit_price) * trade_size, 2)
            status = 'Win' if profit > 0 else 'Loss'
            short_trades.append([symbol, interval, short_entry_time, short_entry_price, exit_time, exit_price, profit, status])
            short_entry_price = None

    if long_entry_price is not None:
        open_positions.append([symbol, interval, long_entry_time, long_entry_price, 'open_positions', 'long'])
    if short_entry_price is not None:
        open_positions.append([symbol, interval, short_entry_time, short_entry_price, 'open_positions', 'short'])

    long_df = pd.DataFrame(long_trades, columns=['symbol', 'Interval', 'Entry_date_time', 'Entry_price', 'Exit_date_time', 'Exit_price', 'Profit', 'Status'])
    short_df = pd.DataFrame(short_trades, columns=['symbol', 'Interval', 'Entry_date_time', 'Entry_price', 'Exit_date_time', 'Exit_price', 'Profit', 'Status'])
    open_df = pd.DataFrame(open_positions, columns=['symbol', 'Interval', 'Entry_date_time', 'Entry_price', 'Status', 'Type'])

    return long_df, short_df, open_df

def send_notification_email(user_id, notification_file_path):
    user = User.query.get(user_id)
    if not user:
        logging.error(f"User with ID {user_id} not found")
        return

    msg = Message('Trade Notification', sender='noreply@stonx.com', recipients=[user.email])
    with open(notification_file_path, 'r') as file:
        msg.body = file.read()
    mail.send(msg)

class NotificationJob(db.Model):
    __tablename__ = 'notification_jobs'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    interval = db.Column(db.String(10), nullable=False)
    active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(150), nullable=False)
    last_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

def monitor_trades(job_id, symbol, interval, trade_size, user_id):
    with app.app_context():
        job = NotificationJob.query.get(job_id)
        if not job or not job.active:
            logging.info(f"Job {job_id} for {symbol} at {interval} has been canceled or completed.")
            return

        df = process_interval(symbol, interval)
        if df is not None:
            long_df, short_df, open_df = evaluate_performance(df, interval, trade_size, symbol)

            if not open_df.empty:
                notification_file_path = os.path.join(DATA_DIR, f"{user_id}_notification.csv")
                open_df.to_csv(notification_file_path, index=False)
                send_notification_email(job.user_id, notification_file_path)

            long_df.to_csv(os.path.join(DATA_DIR, f"{user_id}_{symbol}_long_trades.csv"), index=False)
            short_df.to_csv(os.path.join(DATA_DIR, f"{user_id}_{symbol}_short_trades.csv"), index=False)

            def calculate_success_and_profit(df):
                total_trades = len(df)
                wins = (df['Status'] == 'Win').sum()
                losses = (df['Status'] == 'Loss').sum()
                success_rate = wins / total_trades if total_trades > 0 else 0
                total_profit = df['Profit'].sum()
                return total_trades, wins, losses, success_rate, total_profit

            long_summary = calculate_success_and_profit(long_df)
            short_summary = calculate_success_and_profit(short_df)

            summary_data = [
                [symbol, interval, trade_size, long_summary[0], long_summary[1], long_summary[2], f"{long_summary[3] * 100:.2f}%", f"${long_summary[4]:.2f}", "Long"],
                [symbol, interval, trade_size, short_summary[0], short_summary[1], short_summary[2], f"{short_summary[3] * 100:.2f}%", f"${short_summary[4]:.2f}", "Short"]
            ]

            summary_df = pd.DataFrame(summary_data, columns=['Symbol', 'Interval', 'Trade_size', 'Total_Trades', 'Wins', 'Losses', 'SuccessRate', 'Total_profit', 'Trade_Type'])
            summary_file_path = os.path.join(DATA_DIR, f"{user_id}_{symbol}_summary.csv")
            summary_df.to_csv(summary_file_path, index=False)
        else:
            logging.error("No data available to process for the given symbol and interval.")
