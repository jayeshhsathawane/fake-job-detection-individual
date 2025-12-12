

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------
# 🔹 Load dataset
# ------------------------------------------------------------
df = pd.read_csv('fake_job_postings.csv')

# Drop rows where the cleaned text is missing
df = df.dropna(subset=['description'])

# Target variable (0 = real, 1 = fake)
y = df['fraudulent']
#  TASK 1 – Baseline Model Evaluation


bow_vectorizer = CountVectorizer(max_features=5000)
X_bow = bow_vectorizer.fit_transform(df['description'])

X_train_bow, X_test_bow, y_train, y_test = train_test_split(
    X_bow, y, test_size=0.2, random_state=42, stratify=y
)

model_bow = LogisticRegression(max_iter=200)
model_bow.fit(X_train_bow, y_train)

y_pred_bow = model_bow.predict(X_test_bow)

print(" Bag-of-Words (BoW) Results")
print("Accuracy:", accuracy_score(y_test, y_pred_bow))
print(classification_report(y_test, y_pred_bow))

# ---------- TF-IDF ----------
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df['description'])

X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

model_tf = LogisticRegression(max_iter=200)
model_tf.fit(X_train_tf, y_train_tf)

y_pred_tf = model_tf.predict(X_test_tf)

print("\n TF-IDF Results")
print("Accuracy:", accuracy_score(y_test_tf, y_pred_tf))
print(classification_report(y_test_tf, y_pred_tf))

#  Which performed better?

print("""
Comparison Summary:
TF-IDF usually performs better than Bag-of-Words because:
• It down-weights common words like “the”, “apply”, “job”.
• It gives higher importance to unique or rare words that reveal fraud clues.
""")

#  TASK 2 – Model Analysis


# Predict probabilities for test set
probs = model_tf.predict_proba(X_test_tf)[:, 1]     # Probability of being fake
df_test = df.iloc[y_test_tf.index].copy()           # Get matching test rows
df_test['predicted_proba'] = probs

# Sort to get top-5 most suspicious job postings
top_fake = df_test[['title', 'description', 'predicted_proba']].sort_values(
    by='predicted_proba', ascending=False
).head(5)

print("Top 5 Most Suspicious Job Posts:\n")
print(top_fake[['title', 'predicted_proba']])

print("""
Observation:
Jobs with high fake probabilities often include phrases such as:
 "Work from home", "Limited vacancies", "Apply now"
 "Training fee", "Visa guarantee", "Money transfer"
These expressions sound unrealistic or demand payment - typical fraud signals.
""")


# TASK 3 – Optional Advanced

for mf in [1000, 5000, 10000]:
    tfidf = TfidfVectorizer(max_features=mf)
    X = tfidf.fit_transform(df['description'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"max_features = {mf:<5} -> Accuracy: {acc:.4f}")

print("""
Result Summary:
Accuracy usually improves when increasing features from 1000 - 5000
but may level off or slightly drop at 10,000 due to added noise.
An optimal setting is often around 5000 features.
""")


