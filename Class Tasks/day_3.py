# Day 3: Text Cleaning and Preprocessing

# 1️⃣ Import required libraries
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 2️⃣ Download necessary NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # optional but recommended

# 3️⃣ Load dataset
df = pd.read_csv('fake_job_postings.csv')

# 4️⃣ Define text cleaning function
def clean_text(text):
    if pd.isnull(text):  # Handle missing values
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 4. Remove punctuation and numbers
    text = re.sub(r'[%s\d]' % re.escape(string.punctuation), ' ', text)

    # 5. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)

# 5️⃣ Apply cleaning function
df['clean_description'] = df['description'].apply(clean_text)

# 6️⃣ Show before and after cleaning
print("Original Text:\n", df['description'].iloc[1][:300])
print("\nCleaned Text:\n", df['clean_description'].iloc[1][:300])

# 7️⃣ Display sample cleaned data
print("\nExample of Cleaned Data:")
print(df[['description', 'clean_description']].head(3))
