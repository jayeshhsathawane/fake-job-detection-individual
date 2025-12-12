import sqlite3
import os

db_path = r"D:\INTERNSHIP\Fake_job_detection\Tasks\job_predictions.db"

# Make sure folder exists
os.makedirs(os.path.dirname(db_path), exist_ok=True)

conn = sqlite3.connect(db_path)

conn.execute("""
CREATE TABLE IF NOT EXISTS admin (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    password TEXT NOT NULL
);
""")

conn.execute("""
INSERT INTO admin (username, password)
VALUES ('admin', 'admin123');
""")

conn.commit()
conn.close()

print("Admin user created: username='admin', password='admin123'")
