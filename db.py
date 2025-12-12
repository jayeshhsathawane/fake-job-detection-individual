import sqlite3
import os

# --- STEP 1: FIX PATH (Dynamic Path) ---
# This ensures the DB is created in the exact same folder as this script
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_PATH, "job_predictions.db")

print(f"Connecting to database at: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# -------------------------------
# 1. Create ADMIN TABLE
# -------------------------------
cur.execute("""
CREATE TABLE IF NOT EXISTS admin (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
);
""")

# -------------------------------
# 2. Create PREDICTIONS TABLE
# -------------------------------
cur.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_description TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
""")

# -------------------------------
# 3. Create RETRAIN LOGS TABLE
# -------------------------------
cur.execute("""
CREATE TABLE IF NOT EXISTS retrain_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    accuracy REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    training_source TEXT NOT NULL
);
""")

# -------------------------------
# 4. Insert Default Admin User
# -------------------------------
try:
    cur.execute("SELECT id FROM admin WHERE username='admin'")
    if cur.fetchone() is None:
        cur.execute("INSERT INTO admin (username, password) VALUES (?, ?)",
                    ("admin", "admin123"))
        print("✔ Admin user created (admin / admin123)")
    else:
        print("ℹ Admin user already exists")
except Exception as e:
    print(f"❌ Error inserting admin user: {e}")

# -------------------------------
# 5. Insert Dummy Data for 'Compare Models' (Optional)
# -------------------------------
# We add this so the "Compare Models" page doesn't show an error on fresh start
try:
    cur.execute("SELECT COUNT(*) FROM retrain_logs")
    count = cur.fetchone()[0]
    if count == 0:
        print("Adding dummy training logs for demonstration...")
        # Old Model
        cur.execute("INSERT INTO retrain_logs (accuracy, training_source) VALUES (?, ?)", (92.5, "initial_setup"))
        # New Model (Simulated)
        cur.execute("INSERT INTO retrain_logs (accuracy, training_source) VALUES (?, ?)", (94.2, "initial_setup_v2"))
        print("✔ Dummy training logs added.")
except Exception as e:
    print(f"Error adding dummy logs: {e}")

# Commit & Close
conn.commit()
conn.close()

print("\n✔ Success! Database 'job_predictions.db' is ready.")