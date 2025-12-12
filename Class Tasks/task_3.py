# ----------------------------
#  Task 3: Feature Correlation and Text Insights
# ----------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load dataset (make sure it includes 'clean_description')
df = pd.read_csv('fake_job_postings.csv')


# Otherwise, rerun the cleaning code from Task 1 before continuing.

# Select relevant columns
features = ['has_company_logo', 'telecommuting', 'employment_type', 
            'required_experience', 'fraudulent']
subset = df[features]

#  Grouped analysis
print("\n--- Feature Distribution Grouped by Fraudulent ---")
for col in ['has_company_logo', 'telecommuting', 'employment_type']:
    print(f"\nFeature: {col}")
    print(df.groupby('fraudulent')[col].value_counts(normalize=True))

#  Bar plots comparing proportions
plt.figure(figsize=(15, 5))

# 1 Company Logo presence
plt.subplot(1, 3, 1)
sns.countplot(x='has_company_logo', hue='fraudulent', data=df)
plt.title('Company Logo vs Fraudulent')
plt.xlabel('Has Company Logo (1 = Yes, 0 = No)')
plt.ylabel('Count')

# 2 Remote work availability
plt.subplot(1, 3, 2)
sns.countplot(x='telecommuting', hue='fraudulent', data=df)
plt.title('Remote Work (Telecommuting) vs Fraudulent')
plt.xlabel('Telecommuting (1 = Yes, 0 = No)')
plt.ylabel('Count')

# 3 Employment Type
plt.subplot(1, 3, 3)
sns.countplot(x='employment_type', hue='fraudulent', data=df)
plt.title('Employment Type vs Fraudulent')
plt.xlabel('Employment Type')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# WordClouds for real vs fake job posts
real_text = " ".join(df[df['fraudulent'] == 0]['description'].dropna())
fake_text = " ".join(df[df['fraudulent'] == 1]['description'].dropna())

real_wc = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(real_text)
fake_wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(fake_text)

# Display both word clouds
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
