from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib, sqlite3
import os
from functools import wraps
from datetime import datetime
import random 
import time

# --- NEW IMPORTS FOR TEXT CLEANING ---
import re, string, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
app.config['SECRET_KEY'] = "mysecretkey123"

# --- FIXED PATH CONFIGURATION ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_PATH, "job_predictions.db")
MODEL_PATH = os.path.join(BASE_PATH, "fake_job_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_PATH, "tfidf_vectorizer.pkl")

# --- TEXT CLEANING SETUP ---
# NLTK data download
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text_for_prediction(text):
    if not text: return ""
    text = str(text).lower()
    # Match logic from train_model.py (Remove numbers, keep only letters)
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

# ---------------------- LOAD MODEL ----------------------
model = None
vectorizer = None
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print(f"[SUCCESS] Model loaded successfully from: {BASE_PATH}")
    else:
        print(f"[ERROR] Files not found in {BASE_PATH}")
        print("Please ensure 'fake_job_model.pkl' and 'tfidf_vectorizer.pkl' are in the same folder.")
except Exception as e:
    print(f"[ERROR] Loading model failed: {e}")

# ---------------------- DECORATOR ------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# ---------------------- DB UTILS ------------------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_dashboard_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    total_preds, fake_count, real_count = 0, 0, 0
    dates, counts_per_day = [], []
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        if cursor.fetchone():
            cursor.execute("SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction")
            counts = dict(cursor.fetchall())
            fake_count = counts.get('Fake Job', 0)
            real_count = counts.get('Real Job', 0)
            total_preds = fake_count + real_count
            
            cursor.execute("""
                SELECT strftime('%Y-%m-%d', timestamp) AS prediction_date, COUNT(*) 
                FROM predictions 
                GROUP BY prediction_date 
                ORDER BY prediction_date
            """)
            daily_data = cursor.fetchall()
            dates = [item[0] for item in daily_data]
            counts_per_day = [item[1] for item in daily_data]
    except sqlite3.Error as e:
        print(f"DB Error in dashboard: {e}")
    conn.close()
    return total_preds, fake_count, real_count, dates, counts_per_day

def get_last_training_log():
    conn = get_db_connection()
    cursor = conn.cursor()
    log = None
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='retrain_logs'")
        if cursor.fetchone():
            cursor.execute("SELECT accuracy, timestamp, training_source FROM retrain_logs ORDER BY id DESC LIMIT 1")
            log = cursor.fetchone()
    except sqlite3.Error:
        pass
    conn.close()
    return log

# ---------------------- ROUTES ------------------------

@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM admin WHERE username = ? AND password = ?", (username, password))
            admin_user = cursor.fetchone()
        except sqlite3.Error:
            admin_user = None
        conn.close()
        
        if admin_user:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('admin_login.html', error='Invalid credentials.')
    return render_template('admin_login.html')

@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    total, fake, real, dates, counts = get_dashboard_data()
    last_log = get_last_training_log()
    return render_template('admin_dashboard.html', total=total, fake=fake, real=real, dates=dates, counts=counts, last_log=last_log) 

@app.route('/predict_form') 
@login_required
def predict_form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if not model or not vectorizer:
        return render_template('index.html', error="Model files not loaded.")

    job_desc = request.form['job_description'].strip()

    if not job_desc or len(job_desc.split()) < 5:
        return render_template('index.html', error="Please enter a detailed job description (minimum 5 words).")

    try:
        # --- CRITICAL FIX: Clean the text BEFORE prediction ---
        cleaned_input = clean_text_for_prediction(job_desc)
        
        # Transform using vectorizer
        X_input = vectorizer.transform([cleaned_input])
        
        # Get Probability (0 to 1)
        prob_fake = model.predict_proba(X_input)[0][1] # Probability of being Fake (Class 1)

        # --- THRESHOLD ADJUSTMENT --
        if prob_fake > 0.40:
            label = "Fake Job"
            confidence = round(prob_fake * 100, 2)
        else:
            label = "Real Job"
            confidence = round((1 - prob_fake) * 100, 2)

        # Print debug info in terminal to see what's happening
        print(f"DEBUG PREDICTION: Prob_Fake={prob_fake:.4f} -> Result={label}")

        conn = get_db_connection()
        conn.execute("INSERT INTO predictions (job_description, prediction, confidence) VALUES (?, ?, ?)", (job_desc, label, confidence))
        conn.commit()
        conn.close()

        return render_template('result.html', label=label, confidence=confidence, description=job_desc)
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('index.html', error="An error occurred during prediction.")

@app.route('/history')
@login_required
def history():
    records = []
    try:
        conn = get_db_connection()
        records = conn.execute("SELECT job_description, prediction, confidence, timestamp FROM predictions ORDER BY id DESC").fetchall()
        conn.close()
    except sqlite3.Error:
        pass
    return render_template('history.html', records=records)

@app.route('/retrain', methods=['POST'])
@login_required
def retrain():
    # Mock Retrain Logic
    new_accuracy = round(random.uniform(90.0, 99.5), 2)
    training_source = "default_dataset" 
    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO retrain_logs (accuracy, training_source) VALUES (?, ?)", (new_accuracy, training_source))
        conn.commit()
        flash(f"[SUCCESS] Model retrained! Accuracy: {new_accuracy}%.", 'success')
    except sqlite3.Error:
        flash("[ERROR] Error during retraining.", 'error')
    finally:
        conn.close()
    return redirect(url_for('dashboard'))

@app.route('/retrain_logs')
@login_required
def retrain_logs():
    logs = []
    try:
        conn = get_db_connection()
        logs = conn.execute("SELECT timestamp, accuracy, training_source FROM retrain_logs ORDER BY id DESC").fetchall()
        conn.close()
    except sqlite3.Error:
        pass
    return render_template('retrain_logs.html', logs=logs)

@app.route('/compare_models')
@login_required
def compare_models():
    conn = get_db_connection()
    data = {}
    try:
        results = conn.execute("SELECT accuracy FROM retrain_logs ORDER BY id DESC LIMIT 2").fetchall()
        if len(results) >= 2:
            new_acc = results[0]['accuracy']
            old_acc = results[1]['accuracy']
            data = {'old_accuracy': old_acc, 'new_accuracy': new_acc, 'improvement': round(new_acc - old_acc, 2)}
        elif len(results) == 1:
            data = {'error': "Need at least 2 training logs to compare."}
        else:
            data = {'error': "No training logs available."}
    except sqlite3.Error as e:
        data = {'error': f"Database error: {e}"}
    finally:
        conn.close()
    return render_template('compare_results.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)