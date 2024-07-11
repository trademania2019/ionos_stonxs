# Import necessary modules
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, SelectField, FloatField
from wtforms.validators import DataRequired, Email, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.base import JobLookupError
import subprocess
import logging
import os
import pandas as pd
import time
import requests
from datetime import datetime, timedelta
from pytz import timezone
from bs4 import BeautifulSoup
from logging.config import dictConfig
import flask_profiler
from dotenv import load_dotenv
from flask_migrate import Migrate
from flask import jsonify
import json

os.environ['PYTHONIOENCODING'] = 'utf-8'

# Detailed logging configuration
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

# Initialize Flask app and database
app = Flask(__name__)
# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'postgresql://default_uri')
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
DATA_DIR = os.getenv('DATA_DIR', 'data')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
migrate = Migrate(app, db)

# Flask Profiler configuration
app.config["flask_profiler"] = {
    "enabled": app.config["DEBUG"],
    "storage": {
        "engine": "sqlite",
        "FILE": "flask_profiler.sqlite"
    },
    "basicAuth": {
        "enabled": False
    },
    "ignore": ["^/static/.*"]
}
flask_profiler.init_app(app)

# Initialize Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False  # Ensure only one of TLS or SSL is enabled

mail = Mail(app)

# Initialize Flask-Login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize APScheduler
scheduler = BackgroundScheduler(timezone=timezone('America/New_York'))
scheduler.add_jobstore(SQLAlchemyJobStore(url='sqlite:///jobs.db'), 'default')
scheduler.start()

# Define the base directory for data files
DATA_DIR = os.getenv('DATA_DIR', 'data')

# User model for database
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(150), nullable=False)
    last_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    agreement_accepted = db.Column(db.Boolean, nullable=False, default=False)

class ForumPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(150), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('posts', lazy=True))

class NotificationJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    interval = db.Column(db.String(10), nullable=False)
    active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Form classes
class SymbolForm(FlaskForm):
    symbol = StringField('Stock Symbol', validators=[DataRequired()])
    interval = SelectField('Interval', choices=[('1min', '1 Minute'), ('5min', '5 Minutes'), ('15min', '15 Minutes'), ('60min', '60 Minutes'), ('1D', 'Daily')], validators=[DataRequired()])
    submit = SubmitField('Predict Price')

class PerformanceForm(FlaskForm):
    symbol = StringField('Stock Symbol', validators=[DataRequired()])
    interval = SelectField('Interval', choices=[('1min', '1 Minute'), ('5min', '5 Minutes'), ('15min', '15 Minutes'), ('60min', '60 Minutes'), ('1D', 'Daily')], validators=[DataRequired()])
    trade_size = FloatField('Trade Size', validators=[DataRequired()])
    submit = SubmitField('Show Trades')

class UpdateAccountForm(FlaskForm):
    first_name = StringField('First Name', validators=[DataRequired()])
    last_name = StringField('Last Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('New Password')
    confirm_password = PasswordField('Confirm New Password', validators=[EqualTo('password')])
    submit = SubmitField('Update')

class PasswordRecoveryForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Recover Password')

class PasswordResetForm(FlaskForm):
    password = PasswordField('New Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm New Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')

class UsernameRecoveryForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Recover Username')

class NotificationForm(FlaskForm):
    symbol = StringField('Symbol', validators=[DataRequired()])
    interval = SelectField('Interval', choices=[('1min', '1 Minute'), ('5min', '5 Minutes'), ('15min', '15 Minutes'), ('60min', '60 Minutes'), ('1D', 'Daily')], validators=[DataRequired()])
    submit = SubmitField('Setup Notification')

class ForumPostForm(FlaskForm):
    title = StringField('Title', validators=[DataRequired()])
    content = StringField('Content', validators=[DataRequired()])
    submit = SubmitField('Post')

@app.route('/user-agreement')
def user_agreement():
    return render_template('user_agreement.html')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/forum', methods=['GET', 'POST'])
@login_required
def forum():
    form = ForumPostForm()
    if form.validate_on_submit():
        post = ForumPost(
            title=form.title.data,
            content=form.content.data,
            user_id=current_user.id
        )
        db.session.add(post)
        db.session.commit()
        flash('Post created successfully!', 'success')
        return redirect(url_for('forum'))

    posts = ForumPost.query.order_by(ForumPost.created_at.desc()).all()
    return render_template('forum.html', form=form, posts=posts)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords do not match!")
            return redirect(url_for('register'))

        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash("A user with this email or username already exists.")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(first_name=first_name, last_name=last_name, email=email, username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful. Please log in.")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check your credentials.')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/account', methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        current_user.first_name = form.first_name.data
        current_user.last_name = form.last_name.data
        current_user.email = form.email.data
        current_user.username = form.username.data
        if form.password.data:
            hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
            current_user.password = hashed_password
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.first_name.data = current_user.first_name
        form.last_name.data = current_user.last_name
        form.email.data = current_user.email
        form.username.data = current_user.username
    return render_template('account.html', title='Account', form=form)

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    predict_form = SymbolForm()
    performance_form = PerformanceForm()

    if request.method == 'POST':
        action = request.form.get('action')
        symbol = request.form.get('symbol')
        interval = request.form.get('interval')
        user_id = current_user.id
        script_dir = '.'  # Updated script directory
        python_executable = 'venv/Scripts/python.exe'  # Full path to the Python executable in the virtual environment

        if action == 'predict_price' and predict_form.validate_on_submit():
            start_time = time.time()
            try:
                result = subprocess.run([python_executable, 'predict_price_working.py', symbol, interval, str(user_id)], capture_output=True, text=True, check=True, cwd=script_dir)
                logging.info(f'predict_price_working.py output: {result.stdout}')
                logging.error(f'predict_price_working.py errors: {result.stderr}')
                logging.info('predict_price_working.py executed successfully.')
                return redirect(url_for('result_predict', symbol=symbol, interval=interval, user_id=user_id))
            except subprocess.CalledProcessError as e:
                logging.error(f'Error executing script: {e.stdout}\n{e.stderr}')
                flash('Error executing script. Please try again later.')
            except Exception as e:
                logging.exception('Exception while running script.')
                flash('Unexpected error occurred. Please contact the administrator.')
            finally:
                end_time = time.time()
                logging.info(f"Script execution time: {end_time - start_time} seconds")
            return redirect(url_for('index'))

        elif action == 'show_trades' and performance_form.validate_on_submit():
            trade_size = request.form.get('trade_size')
            start_time = time.time()
            try:
                result = subprocess.run([python_executable, 'trades_working.py', symbol, interval, str(trade_size), str(user_id)], capture_output=True, text=True, check=True, cwd=script_dir)
                logging.info(f'trades_working.py output: {result.stdout}')
                logging.error(f'trades_working.py errors: {result.stderr}')
                logging.info('trades_working.py executed successfully.')
                return redirect(url_for('result_trades', symbol=symbol, interval=interval, trade_size=trade_size, user_id=user_id))
            except subprocess.CalledProcessError as e:
                logging.error(f'Error executing script: {e.stdout}\n{e.stderr}')
                flash('Error executing script. Please try again later.')
            except Exception as e:
                logging.exception('Exception while running script.')
                flash('Unexpected error occurred. Please contact the administrator.')
            finally:
                end_time = time.time()
                logging.info(f"Script execution time: {end_time - start_time} seconds")
            return redirect(url_for('index'))

    return render_template('index.html', predict_form=predict_form, performance_form=performance_form)

@app.route('/result_trades')
@login_required
def result_trades():
    symbol = request.args.get('symbol')
    interval = request.args.get('interval')
    trade_size = request.args.get('trade_size')
    user_id = request.args.get('user_id')

    summary_file_path = os.path.join(DATA_DIR, str(user_id), f"{user_id}_{symbol}_summary.csv")
    if not os.path.exists(summary_file_path):
        flash('Summary file not found.')
        return redirect(url_for('index'))

    summary_df = pd.read_csv(summary_file_path)

    if summary_df['Total_profit'].dtype == 'object':
        summary_df['Total_profit'] = summary_df['Total_profit'].str.replace('$', '').astype(float)
    else:
        summary_df['Total_profit'] = summary_df['Total_profit'].astype(float)

    formatted_summary_lines = summary_df.to_dict('records')

    long_trades_path = os.path.join(DATA_DIR, str(user_id), f"{user_id}_{symbol}_long_trades.csv")
    short_trades_path = os.path.join(DATA_DIR, str(user_id), f"{user_id}_{symbol}_short_trades.csv")
    open_positions_path = os.path.join(DATA_DIR, str(user_id), f"{user_id}_{symbol}_open_positions.csv")

    if not os.path.exists(long_trades_path) or not os.path.exists(short_trades_path):
        flash('Trades file not found.')
        return redirect(url_for('index'))

    long_trades = pd.read_csv(long_trades_path).to_dict('records')
    short_trades = pd.read_csv(short_trades_path).to_dict('records')

    open_positions = []
    if os.path.exists(open_positions_path):
        open_positions = pd.read_csv(open_positions_path).to_dict('records')

    open_positions_summary = {
        'Total_open_positions': len(open_positions),
        'Total_open_profit': sum([pos['Profit'] for pos in open_positions if 'Profit' in pos]),
    }

    return render_template('result_trades.html', 
                           symbol=symbol, 
                           interval=interval,
                           trade_size=trade_size, 
                           user_id=user_id,  # Ensure user_id is passed to the template
                           summary_lines=formatted_summary_lines, 
                           long_trades=long_trades, 
                           short_trades=short_trades, 
                           open_positions=open_positions, 
                           open_positions_summary=open_positions_summary)

@app.route('/predict_trade', methods=['POST'])
@login_required
def predict_trade():
    form = SymbolForm()
    if form.validate_on_submit():
        symbol = form.symbol.data
        interval = form.interval.data
        user_id = current_user.id
        python_executable = 'venv/Scripts/python.exe'
        script_dir = '.'
        
        start_time = time.time()
        try:
            result = subprocess.run([python_executable, 'predict_price_working.py', symbol, interval, str(user_id)], capture_output=True, text=True, check=True, cwd=script_dir)
            logging.info(f'predict_price_working.py output: {result.stdout}')
            logging.error(f'predict_price_working.py errors: {result.stderr}')
            logging.info('predict_price_working.py executed successfully.')
            return redirect(url_for('result_predict', symbol=symbol, interval=interval, user_id=user_id))
        except subprocess.CalledProcessError as e:
            logging.error(f'Error executing script: {e.stdout}\n{e.stderr}')
            flash('Error executing script. Please try again later.')
        except Exception as e:
            logging.exception('Exception while running script.')
            flash('Unexpected error occurred. Please contact the administrator.')
        finally:
            end_time = time.time()
            logging.info(f"Script execution time: {end_time - start_time} seconds")
    else:
        flash('Form validation failed. Please check your inputs.')

    return redirect(url_for('index'))

@app.route('/show_algo_performance', methods=['POST'])
@login_required
def show_algo_performance():
    form = PerformanceForm()
    if form.validate_on_submit():
        symbol = form.symbol.data
        interval = form.interval.data
        trade_size = form.trade_size.data
        user_id = current_user.id
        action = request.form.get('action')
        script_dir = '.'  # Updated script directory
        python_executable = 'venv/Scripts/python.exe'  # Full path to the Python executable in the virtual environment

        start_time = time.time()
        try:
            result = subprocess.run([python_executable, 'trades_working.py', symbol, interval, str(trade_size), str(user_id)], capture_output=True, text=True, check=True, cwd=script_dir)
            logging.info(f'trades_working.py output: {result.stdout}')
            logging.error(f'trades_working.py errors: {result.stderr}')
            logging.info('trades_working.py executed successfully.')
            return redirect(url_for('result_trades', symbol=symbol, interval=interval, trade_size=trade_size, user_id=user_id))
        except subprocess.CalledProcessError as e:
            logging.error(f'Error executing script: {e.stdout}\n{e.stderr}')
            flash('Error executing script. Please try again later.')
            return redirect(url_for('index'))
        except Exception as e:
            logging.exception('Exception while running script.')
            flash('Unexpected error occurred. Please contact the administrator.')
            return redirect(url_for('index'))
        finally:
            end_time = time.time()
            logging.info(f"Script execution time: {end_time - start_time} seconds")
    return redirect(url_for('index'))

@app.route('/result_predict')
@login_required
def result_predict():
    symbol = request.args.get('symbol')
    interval = request.args.get('interval')
    user_id = request.args.get('user_id')

    app.logger.info(f"result_predict called with symbol: {symbol}, interval: {interval}")

    if not symbol or not interval:
        app.logger.error("Missing symbol or interval in request")
        flash('Missing symbol or interval.')
        return redirect(url_for('index'))

    trend_file_path = os.path.join(DATA_DIR, str(user_id), f"{user_id}_{interval}_trend.txt")
    recommendation_file_path = os.path.join(DATA_DIR, str(user_id), f"{user_id}_{interval}_recommendation.txt")

    app.logger.info(f"Checking existence of trend_file: {trend_file_path}")
    app.logger.info(f"Checking existence of recommendation_file: {recommendation_file_path}")

    if not os.path.exists(trend_file_path):
        app.logger.error(f"Trend file not found: {trend_file_path}")
        flash('Trend file not found.')
        return redirect(url_for('index'))

    if not os.path.exists(recommendation_file_path):
        app.logger.error(f"Recommendation file not found: {recommendation_file_path}")
        flash('Recommendation file not found.')
        return redirect(url_for('index'))

    with open(trend_file_path, 'r') as f:
        trend_content = f.read()

    with open(recommendation_file_path, 'r') as f:
        recommendation_content = f.read()

    app.logger.info("All required files found. Rendering result_predict.html")

    return render_template('result_predict.html', 
                           symbol=symbol, 
                           interval=interval, 
                           trend_content=trend_content, 
                           recommendation_content=recommendation_content)

# Define the path to store last notified positions from environment variable
LAST_NOTIFIED_PATH = os.getenv('LAST_NOTIFIED_PATH', 'last_notified_positions.json')

# Function to load the last notified positions from file
def load_last_notified_positions():
    if os.path.exists(LAST_NOTIFIED_PATH):
        with open(LAST_NOTIFIED_PATH, 'r') as file:
            return json.load(file)
    return {}

# Function to save the last notified positions to file
def save_last_notified_positions(last_notified_positions):
    with open(LAST_NOTIFIED_PATH, 'w') as file:
        json.dump(last_notified_positions, file)

# Function to check if the position has changed
def has_position_changed(last_position, current_position):
    return last_position != current_position

def monitor_trades(job_id, symbol, interval, user_id):
    with app.app_context():
        logging.info(f"Starting to monitor trades for job_id: {job_id}, symbol: {symbol}, interval: {interval}, user_id: {user_id}")
        job = NotificationJob.query.get(job_id)
        if not job or not job.active:
            logging.info(f"Job {job_id} for {symbol} at {interval} has been canceled or completed.")
            return

        logging.info(f"Monitoring trades for symbol: {symbol}, interval: {interval}, user_id: {user_id}")

        try:
            python_executable = 'venv/Scripts/python.exe'  # Adjust this path to your virtual environment's Python executable
            result = subprocess.run([python_executable, 'trades_working_monitor.py', symbol, interval, '1', str(user_id)], capture_output=True, text=True, check=True)
            logging.info(f'trades_working_monitor.py output: {result.stdout}')
            logging.error(f'trades_working_monitor.py errors: {result.stderr}')
        except subprocess.CalledProcessError as e:
            logging.error(f'Error executing trades_working_monitor.py: {e.stdout}\n{e.stderr}')
            return

        open_positions_file_path = os.path.join(DATA_DIR, str(user_id), f"{user_id}_{symbol}_open_positions.csv")
        if os.path.exists(open_positions_file_path):
            # Load the last notified positions
            last_notified_positions = load_last_notified_positions()

            # Load current open positions
            current_open_positions = pd.read_csv(open_positions_file_path).to_dict('records')

            # Check if the positions have changed
            user_key = f"{user_id}_{symbol}_{interval}"
            last_position = last_notified_positions.get(user_key, [])

            if has_position_changed(last_position, current_open_positions):
                # Send notification email if there are changes
                send_notification_email(user_id, open_positions_file_path)

                # Update the last notified positions
                last_notified_positions[user_key] = current_open_positions
                save_last_notified_positions(last_notified_positions)
            else:
                logging.info(f"No change in position for {user_key}. No email sent.")
def send_notification_email(user_id, file_path):
    user = User.query.get(user_id)
    if not user:
        logging.error(f'User with id {user_id} not found.')
        return

    with open(file_path, 'r') as file:
        email_content = file.read()

    msg = Message(
        'Trade Notification',
        sender=app.config['MAIL_USERNAME'],
        recipients=[user.email]
    )
    msg.body = email_content

    try:
        mail.send(msg)
        logging.info(f'Notification email sent to {user.email}')
    except Exception as e:
        logging.error(f'Failed to send email: {e}')

@app.route('/notifications', methods=['GET', 'POST'])
@login_required
def notifications():
    form = NotificationForm()
    if form.validate_on_submit():
        symbol = form.symbol.data
        interval = form.interval.data
        user_id = current_user.id

        # Create a new notification job entry in the database
        job = NotificationJob(user_id=user_id, symbol=symbol, interval=interval)
        db.session.add(job)
        db.session.commit()

        # Use the actual database ID as the job ID
        job_id = job.id
        
        # Define the CronTrigger to run the job from Monday to Friday, 9:30 AM to 4:00 PM EST
        trigger = CronTrigger(
            day_of_week='mon-fri',
            hour='9-16',
            minute='*/3',
            timezone='America/New_York'
        )

        # Schedule the job with the defined CronTrigger
        scheduler.add_job(monitor_trades, trigger, args=[job_id, symbol, interval, user_id], id=str(job_id), replace_existing=True)

        flash('Notification setup successfully!', 'success')
        return redirect(url_for('notifications'))

    notifications = NotificationJob.query.filter_by(user_id=current_user.id, active=True).all()
    return render_template('notification.html', form=form, notifications=notifications)

@app.route('/cancel_notification', methods=['POST'])
@login_required
def cancel_notification():
    notification_id = request.form['notification_id']
    job = NotificationJob.query.get(notification_id)
    if job and job.user_id == current_user.id:
        job.active = False
        db.session.commit()
        job_id = str(job.id)
        app.logger.info(f"Attempting to cancel job with ID: {job_id}")
        try:
            scheduler.remove_job(job_id)
            flash('Notification canceled successfully!', 'success')
        except JobLookupError:
            app.logger.error(f"No job found with ID: {job_id}")
            flash('No notification found to cancel.', 'danger')
    else:
        flash('No notification found to cancel.', 'danger')
    return redirect(url_for('notifications'))

@app.route('/data/<filename>')
@login_required
def serve_data_file(filename):
    user_id = current_user.id
    return send_from_directory(os.path.join(DATA_DIR, str(user_id)), filename, as_attachment=True)


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'DNT': '1',  # Do Not Track Request Header
    'Upgrade-Insecure-Requests': '1',
}

def fetch_market_data(url):
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.text
    return None

def parse_market_data(html):
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', {'class': 'W(100%)'})
    if not table:
        return []

    rows = table.find('tbody').find_all('tr')
    data = []
    for row in rows[:5]:  # Limit to top 5 entries
        columns = row.find_all('td')
        data.append({
            'symbol': columns[0].text.strip(),
            'name': columns[1].text.strip(),
            'price': columns[2].text.strip(),
            'change': columns[3].text.strip(),
            '%change': columns[4].text.strip(),
            'volume': columns[5].text.strip(),
            'avg_volume': columns[6].text.strip(),
            'market_cap': columns[7].text.strip(),
            'pe_ratio': columns[8].text.strip(),
        })
    return data

@app.route('/market_watch')
@login_required
def market_watch():
    gainers_url = "https://finance.yahoo.com/screener/predefined/day_gainers/"
    losers_url = "https://finance.yahoo.com/screener/predefined/day_losers/"
    most_active_url = "https://finance.yahoo.com/screener/predefined/most_actives/"
    most_shorted_url = "https://finance.yahoo.com/screener/predefined/most_shorted_stocks/"

    gainers_html = fetch_market_data(gainers_url)
    if gainers_html is None:
        print("Failed to fetch data from gainers URL.")
    top_gainers = parse_market_data(gainers_html) if gainers_html else []

    losers_html = fetch_market_data(losers_url)
    if losers_html is None:
        print("Failed to fetch data from losers URL.")
    top_losers = parse_market_data(losers_html) if losers_html else []

    most_active_html = fetch_market_data(most_active_url)
    if most_active_html is None:
        print("Failed to fetch data from most active URL.")
    most_active = parse_market_data(most_active_html) if most_active_html else []

    most_shorted_html = fetch_market_data(most_shorted_url)
    if most_shorted_html is None:
        print("Failed to fetch data from most shorted URL.")
    most_shorted = parse_market_data(most_shorted_html) if most_shorted_html else []

    return render_template('market_watch.html', top_gainers=top_gainers, top_losers=top_losers, most_active=most_active, most_shorted=most_shorted)

@app.route('/recover_password', methods=['GET', 'POST'])
def recover_password():
    form = PasswordRecoveryForm()
    if form.validate_on_submit():
        email = form.email.data
        user = User.query.filter_by(email=email).first()
        if user:
            token = generate_password_reset_token(user.id)
            send_password_recovery_email(user.email, token)
            flash('A password recovery email has been sent.', 'info')
        else:
            flash('No account found with that email address.', 'warning')
        return redirect(url_for('login'))
    return render_template('recover_password.html', form=form)

@app.route('/recover_username', methods=['GET', 'POST'])
def recover_username():
    form = UsernameRecoveryForm()
    if form.validate_on_submit():
        email = form.email.data
        user = User.query.filter_by(email=email).first()
        if user:
            send_username_recovery_email(user.email, user.username)
            flash('A username recovery email has been sent.', 'info')
        else:
            flash('No account found with that email address.', 'warning')
        return redirect(url_for('login'))
    return render_template('recover_username.html', form=form)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user_id = verify_password_reset_token(token)
    if not user_id:
        flash('Invalid or expired token', 'warning')
        return redirect(url_for('recover_password'))
    
    form = PasswordResetForm()
    if form.validate_on_submit():
        user = User.query.get(user_id)
        old_password_hash = user.password
        new_password = form.password.data
        new_password_hash = generate_password_hash(new_password)
        user.password = new_password_hash
        db.session.commit()
        updated_user = User.query.get(user_id)
        updated_password_hash = updated_user.password
        if new_password_hash == updated_password_hash:
            print("Password has been successfully updated.")
        else:
            print("Password update failed.")
        flash('Your password has been updated!', 'success')
        return redirect(url_for('login'))
    return render_template('reset_password.html', form=form, token=token)

def generate_password_reset_token(user_id):
    serializer = URLSafeTimedSerializer(app.secret_key)
    return serializer.dumps(user_id, salt='password-reset-salt')

def verify_password_reset_token(token, expiration=3600):
    serializer = URLSafeTimedSerializer(app.secret_key)
    try:
        user_id = serializer.loads(token, salt='password-reset-salt', max_age=expiration)
    except Exception:
        return None
    return user_id

def send_password_recovery_email(email, token):
    msg = Message('Password Recovery', sender='noreply@stonxs.com', recipients=[email])
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_password', token=token, _external=True)}

If you did not make this request, simply ignore this email and no changes will be made.
'''
    mail.send(msg)

def send_username_recovery_email(email, username):
    msg = Message('Username Recovery', sender='noreply@stonxs.com', recipients=[email])
    msg.body = f'''Your username is: {username}

If you did not make this request, simply ignore this email.
'''
    mail.send(msg)
@app.route('/how-to-use')
def how_to_use():
    return render_template('how_to_use.html')

@app.route('/how-to-use-pdf')
def how_to_use_pdf():
    return send_from_directory('static', 'stonx_howtousev1.pdf')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    with app.app_context():
        db.create_all()
    app.run(debug=True)
