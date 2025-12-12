# ============================================
#  Day 7: Model Evaluation & Hyperparameter Tuning


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# -------------------------------
# Load and Prepare Data
# -------------------------------
df = pd.read_csv('fake_job_postings.csv')
df = df.dropna(subset=['description'])

X_text = df['description']
y = df['fraudulent']

# Convert text to numerical features
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(X_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Task 1: Cross-Validation Analysis
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_results[name] = {
        "mean": scores.mean(),
        "std": scores.std()
    }
    print(f"{name} - Mean CV Accuracy: {scores.mean():.4f} | Std Dev: {scores.std():.4f}")

# Convert to DataFrame for plotting
cv_df = pd.DataFrame(cv_results).T
cv_df.reset_index(inplace=True)
cv_df.columns = ['Model', 'Mean Accuracy', 'Std Dev']

# Bar chart for CV results
plt.figure(figsize=(8,5))
sns.barplot(x='Model', y='Mean Accuracy', data=cv_df, palette='viridis')
plt.title('5-Fold Cross-Validation Accuracy Comparison')
plt.ylabel('Mean Accuracy')
plt.xlabel('')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Identify stable model
most_stable = cv_df.loc[cv_df['Std Dev'].idxmin(), 'Model']
print(f"\n Most stable model (lowest variance): {most_stable}")

# -------------------------------
# Task 2: ROC-AUC Visualization
# -------------------------------
# Train all models
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

# Plot ROC curves
plt.figure(figsize=(8,6))
for name, model in trained_models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0,1], [0,1], 'k--', label='Random Chance')
plt.title('ROC-AUC Comparison of Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# Task 3: Hyperparameter Tuning (Decision Tree)
# -------------------------------
param_grid = {
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\n Best Decision Tree Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Compare tuned Decision Tree vs Random Forest
best_dt = grid_search.best_estimator_
best_dt.fit(X_train, y_train)
y_pred_dt = best_dt.predict(X_test)
y_pred_rf = trained_models["Random Forest"].predict(X_test)

acc_dt = (y_pred_dt == y_test).mean()
acc_rf = (y_pred_rf == y_test).mean()

print(f"\n Tuned Decision Tree Accuracy: {acc_dt:.4f}")
print(f" Random Forest Accuracy: {acc_rf:.4f}")

# -------------------------------
# Interpretation
# -------------------------------
print("""
 Interpretation:
Cross-validation shows how consistently each model performs.
ROC-AUC curves indicate how well each model separates fake vs real jobs.
After tuning, the Decision Tree often improves accuracy and becomes closer to Random Forest performance.
Random Forest still tends to perform best overall due to its ensemble nature and reduced overfitting.
""")
