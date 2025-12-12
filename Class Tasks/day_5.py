# Day 5: Logistic Regression Model for Fake Job Detection
 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 
# Load dataset (preprocessed with clean_description)
df = pd.read_csv('fake_job_postings.csv')
df = df.dropna(subset=['description'])
 
# 1️⃣ Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['description'])
y = df['fraudulent']
 
# 2️⃣ Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
# 3️⃣ Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
 
# 4️⃣ Make predictions
y_pred = model.predict(X_test)
 
# 5️⃣ Evaluate performance
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
# 6️⃣ Check example predictions
test_samples = [
    "Work from home! Limited vacancies. Apply now.",
    "We are hiring a data scientist for our Bangalore office."
]
sample_features = vectorizer.transform(test_samples)
print("\nSample Predictions:", model.predict(sample_features))