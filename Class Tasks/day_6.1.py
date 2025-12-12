# ========================================
# 🧩 Model Comparison & Feature Importance
# ========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# -------------------------------
# Load dataset and vectorize text
# -------------------------------

df = pd.read_csv("fake_job_postings.csv")
df = df.dropna(subset=["description"])
X_text = df["description"]
y = df["fraudulent"]

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(X_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 1️⃣ Logistic Regression (Baseline)
# -------------------------------

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# -------------------------------
# 2️⃣ Decision Tree (try multiple depths)
# -------------------------------

depths = [10, 20, 30]
dt_results = []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    dt_results.append({
        "Model": f"DecisionTree (max_depth={d})",
        "Accuracy": accuracy_score(y_test, y_pred_dt),
        "Precision": precision_score(y_test, y_pred_dt),
        "Recall": recall_score(y_test, y_pred_dt),
        "F1-score": f1_score(y_test, y_pred_dt)
    })
# -------------------------------
# 3️⃣ Random Forest (try multiple n_estimators)
# -------------------------------

estimators = [50, 100, 200]
rf_results = []

for n in estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_results.append({
        "Model": f"RandomForest (n={n})",
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf),
        "Recall": recall_score(y_test, y_pred_rf),
        "F1-score": f1_score(y_test, y_pred_rf)
    })

# -------------------------------
# Combine Results
# -------------------------------

comparison_data = [
    {
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(y_test, y_pred_lr),
        "Precision": precision_score(y_test, y_pred_lr),
        "Recall": recall_score(y_test, y_pred_lr),
        "F1-score": f1_score(y_test, y_pred_lr)
    }
] + dt_results + rf_results

results_df = pd.DataFrame(comparison_data)
print("\n Model Comparison Results:")
print(results_df)

# -------------------------------
# Plot Comparison
# -------------------------------

plt.figure(figsize=(10,6))
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# -------------------------------
# 🧠 Feature Importance (Random Forest)
# -------------------------------

best_rf = RandomForestClassifier(n_estimators=200, random_state=42)
best_rf.fit(X_train, y_train)

importances = best_rf.feature_importances_
indices = importances.argsort()[-15:][::-1]
top_features = [tfidf.get_feature_names_out()[i] for i in indices]
top_importances = importances[indices]

plt.figure(figsize=(10,5))
sns.barplot(x=top_importances, y=top_features, palette="Reds_r")
plt.title("Top 15 Important Words (Random Forest)")
plt.xlabel("Feature Importance")
plt.ylabel("Word")
plt.tight_layout()
plt.show()

# -------------------------------
# Interpretation
# -------------------------------

print("\n Interpretation:")
print(" The top words often relate to suspicious terms like 'limited', 'urgent', 'fee', or 'visa'.")
print(" These words align with typical fake job language patterns.")
print(" If unrelated or generic words appear in the top list, it suggests noise or dataset imbalance.")
print("\n Random Forest usually outperforms Decision Tree due to averaging multiple trees and reducing overfitting.")
