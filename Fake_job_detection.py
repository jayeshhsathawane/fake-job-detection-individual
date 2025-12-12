# =========================================================
#   FAKE JOB DETECTION PIPELINE — COMPLETE PROJECT
# =========================================================

# ----------------------------
# Day 2 — Data Understanding
# ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- PATH CONFIGURATION (Fixed for your folder) ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_PATH, 'fake_job_postings.csv')
MODEL_PATH = os.path.join(BASE_PATH, 'fake_job_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_PATH, 'tfidf_vectorizer.pkl')

print("Loading dataset...")
if not os.path.exists(CSV_PATH):
    print("Error: 'fake_job_postings.csv' not found in this folder.")
    exit()

df = pd.read_csv(CSV_PATH)

print("Dataset loaded successfully.")
print("\nSample Data:\n", df.head())
print("\nDataset Info:")
print(df.info())

# Missing values
print("\nMissing Values per Column:\n", df.isnull().sum())

# Target distribution
print("\nTarget (fraudulent) Distribution:\n", df['fraudulent'].value_counts())

# ----------------------------
# Day 3 — Text Cleaning & Preprocessing
# ----------------------------
import re, string, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print("\nDownloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # --- LOGIC UPDATE: Keeping Numbers and $ Sign ---
    # Fake jobs me aksar salary numbers ($5000) hote hain, isliye inhe nahi hatayenge
    text = re.sub(r'[^a-zA-Z0-9\s$]', '', text) 
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

print("\nCleaning text data... (Please wait)")
df['clean_description'] = df['description'].apply(clean_text)

df['word_count_before'] = df['description'].fillna('').apply(lambda x: len(str(x).split()))
df['word_count_after'] = df['clean_description'].apply(lambda x: len(x.split()))

print("\nAverage word count before:", df['word_count_before'].mean())
print("Average word count after:", df['word_count_after'].mean())
print("Text cleaning completed.")

# ----------------------------
# Day 3.5 — Feature Correlation & Insights
# ----------------------------
from wordcloud import WordCloud

print("\nGenerating Visualizations... (Close windows to continue)")

# Plots
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.countplot(x='has_company_logo', hue='fraudulent', data=df)
plt.title('Company Logo vs Fraudulent')

plt.subplot(1, 3, 2)
sns.countplot(x='telecommuting', hue='fraudulent', data=df)
plt.title('Remote Work vs Fraudulent')

plt.subplot(1, 3, 3)
sns.countplot(x='employment_type', hue='fraudulent', data=df)
plt.title('Employment Type vs Fraudulent')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# WordClouds
real_text = " ".join(df[df['fraudulent'] == 0]['clean_description'])
fake_text = " ".join(df[df['fraudulent'] == 1]['clean_description'])

if len(fake_text) > 0:
    real_wc = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(real_text)
    fake_wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(fake_text)

    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.imshow(real_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Real Job Descriptions (Green)')

    plt.subplot(1,2,2)
    plt.imshow(fake_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Fake Job Descriptions (Red)')
    plt.show()

# ----------------------------
# Day 4 — Feature Extraction
# ----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

# --- LOGIC UPDATE: Better Features ---
print("\nExtracting Features (TF-IDF)...")
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(df['clean_description'])

print("TF-IDF Shape:", X_tfidf.shape)

# ----------------------------
# Day 5 — Model Training (UPDATED)
# ----------------------------
from sklearn.model_selection import train_test_split
# Using Random Forest instead of Logistic Regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y = df['fraudulent']
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining Random Forest Model (Better for Imbalanced Data)...")
# Class weight balanced helps detect fake jobs better
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ----------------------------
# Day 6 — Model Saving
# ----------------------------
import joblib

print(f"\nSaving model and vectorizer to current folder...")
joblib.dump(rf_model, MODEL_PATH)
joblib.dump(tfidf, VECTORIZER_PATH)

print("Success! Model saved.")
