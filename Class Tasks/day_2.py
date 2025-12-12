# Day 2: Understanding and Loading the Dataset
 
import pandas as pd
 
# Load dataset (after downloading from Kaggle)
df = pd.read_csv('fake_job_postings.csv')
 
# Display first few rows
print("Sample Data:")
print(df.head())
 
# Display basic info
print("\nDataset Info:")
print(df.info())
 
# Check for missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())
 
# Check distribution of target variable
print("\nTarget (fraudulent) Distribution:")
print(df['fraudulent'].value_counts())
 
# Basic statistics
print("\nDataset Summary:")
print(df.describe(include='all'))

 