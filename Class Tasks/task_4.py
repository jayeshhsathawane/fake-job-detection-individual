import pandas as pd
import re

# 🗂️ Load dataset
df = pd.read_csv('fake_job_postings.csv')  

text_column = 'description'

#  List of suspicious keywords
suspicious_keywords = [
    "urgent", "work from home", "limited vacancy", "visa",
    "investment", "training fee", "money transfer", "earn", "easy job", "quick money"
]

#  Rule-based flag function
def rule_based_flag(text):
    if pd.isnull(text):
        return 0
    text = text.lower()
    for kw in suspicious_keywords:
        if kw in text:
            return 1
    return 0

#  Apply function correctly
df['suspect_flag'] = df[text_column].apply(rule_based_flag)

# Compare with actual fraudulent column
print("\nRule-based vs Actual Fraudulent:")
print(pd.crosstab(df['suspect_flag'], df['fraudulent'], normalize='all'))

# examples that were flagged as suspicious but not actually fake
suspect_not_fake = df[(df['suspect_flag'] == 1) & (df['fraudulent'] == 0)]
print("\nExamples of jobs flagged as suspect but real:\n")
for i in range(3):
    print(f"Title: {suspect_not_fake.iloc[i]['title']}")
    print(f"Description: {suspect_not_fake.iloc[i]['description'][:200]}...\n")
