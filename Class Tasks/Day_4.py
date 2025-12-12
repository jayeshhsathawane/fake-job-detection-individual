# Day 4: Feature Extraction using BoW and TF-IDF 

# 1️⃣ Import required libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

# 2️⃣ Load the preprocessed dataset (from Day 3)
df = pd.read_csv('fake_job_postings.csv')

# Ensure the 'description' column exists and fill any missing values
texts = df['description'].fillna('').tolist()

# ----------------------------------------------------------
# 🧩 Bag-of-Words (BoW)
# ----------------------------------------------------------
bow_vectorizer = CountVectorizer(max_features=2000)  # limit to top 2000 frequent words
X_bow = bow_vectorizer.fit_transform(texts)

print("BoW shape:", X_bow.shape)
print("Sample feature names (BoW):", bow_vectorizer.get_feature_names_out()[:10])

# ----------------------------------------------------------
# 🧠 TF-IDF (Term Frequency - Inverse Document Frequency)
# ----------------------------------------------------------
tfidf_vectorizer = TfidfVectorizer(max_features=2000)
X_tfidf = tfidf_vectorizer.fit_transform(texts)

print("\nTF-IDF shape:", X_tfidf.shape)
print("Sample feature names (TF-IDF):", tfidf_vectorizer.get_feature_names_out()[:10])

# ----------------------------------------------------------
# 🔍 Compare sparsity and vector values
# ----------------------------------------------------------
print("\nExample BoW vector (first row):")
print(X_bow[0].toarray())

print("\nExample TF-IDF vector (first row):")
print(X_tfidf[0].toarray())

# ==========================================================
# 🧭 TASK 1: Create BoW and TF-IDF for 'company_profile'
# ==========================================================

# Fill missing company profiles
profiles = df['company_profile'].fillna('').tolist()

# Bag-of-Words for company profiles
bow_vectorizer_profile = CountVectorizer(max_features=1000)
X_bow_profile = bow_vectorizer_profile.fit_transform(profiles)

# TF-IDF for company profiles
tfidf_vectorizer_profile = TfidfVectorizer(max_features=1000)
X_tfidf_profile = tfidf_vectorizer_profile.fit_transform(profiles)

print("\n[Task 1] Company Profile - Feature Shapes:")
print("BoW (company_profile):", X_bow_profile.shape)
print("TF-IDF (company_profile):", X_tfidf_profile.shape)

print("\nWhich captures meaning better?")
print("TF-IDF generally captures meaning better because it gives higher weight to unique, informative words and reduces weight of common words like 'the', 'and', 'company'.")

# ==========================================================
# 🧭 TASK 2: Top 20 Most Frequent Words in Job Descriptions (using BoW)
# ==========================================================

# Get word frequencies
word_counts = np.array(X_bow.sum(axis=0)).flatten()
vocab = bow_vectorizer.get_feature_names_out()

# Combine into DataFrame
word_freq_df = pd.DataFrame({'word': vocab, 'count': word_counts})
top20 = word_freq_df.sort_values(by='count', ascending=False).head(20)

print("\n[Task 2] Top 20 Most Frequent Words in Job Descriptions:")
print(top20)

# Optional: Show as bar chart
#try:
   # import matplotlib.pyplot as plt
   # plt.figure(figsize=(10,5))
   # plt.barh(top20['word'][::-1], top20['count'][::-1])
   # plt.xlabel('Frequency')
   # plt.title('Top 20 Most Frequent Words in Job Descriptions')
  #  plt.show()
#except ImportError:
    #print("matplotlib not installed — skipping plot.")
