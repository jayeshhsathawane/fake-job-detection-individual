# Fake Job Detection System  
**Infosys Springboard 6.0 – Internship Project**

A Machine Learning–based web application to identify fake job postings using Natural Language Processing (NLP).  
This project was developed as part of the **Infosys Springboard 6.0 Internship Program**.

---

## 📌 Internship Details

- **Internship Program:** Infosys Springboard 6.0  
- **Project Type:** Individual Project  
- **Domain:** Artificial Intelligence / Machine Learning  
- **Project Title:** Fake Job Detection System  

---

## 🧠 Project Overview

With the rapid growth of online job portals, fake job postings have become a serious issue. This project aims to help job seekers by automatically detecting whether a job posting is **Real** or **Fake** using Machine Learning and NLP techniques.

The system analyzes job descriptions and related information to classify job postings accurately.

---

## 🎯 Objectives

- Detect fake job postings automatically  
- Apply NLP techniques for text processing  
- Train and use a Machine Learning classification model  
- Provide a simple web interface for predictions  

---

## 🗂 Project Structure


fake-job-detection-individual/
│
├── static/                   
│   └── (CSS files, images, and other static assets)
│
├── templates/                
│   └── (HTML templates for the web interface)
│
├── Fake_job_detection.py     
│   └── Train Model file
│
├── admin.py                  
│   └── Main Flask application file
│
├── adim_create.py            
│   └── Script to create/admin user setup
│
├── db.py                     
│   └── Database connection and operations
│
├── fake_job_model.pkl        
│   └── Trained Machine Learning model
│
├── tfidf_vectorizer.pkl      
│   └── TF-IDF vectorizer for text feature extraction
│
├── fake_job_postings.csv     
│   └── Dataset used for training and testing
│
├── job_predictions.db       
│   └── SQLite database to store prediction history
│
├── requirements.txt          
│   └── List of required Python dependencies
│
└── README.md                 
    └── Project documentation


    ---

## ⚙️ Technologies Used

- **Python**
- **Flask**
- **Machine Learning (scikit-learn)**
- **Natural Language Processing (NLP)**
- **Pandas & NumPy**
- **HTML, CSS**
- **SQLite Database**

---

## 🔍 How the System Works

1. Job posting data is collected from the dataset  
2. Text data is cleaned and preprocessed  
3. TF-IDF is used to convert text into numerical features  
4. A trained ML model predicts whether the job is **Fake** or **Real**  
5. The result is displayed via a web interface  

---

## 🚀 Installation & Execution

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/jayeshhsathawane/fake-job-detection-individual.git
cd fake-job-detection-individual

#Install Required Packages
pip install -r requirements.txt

#Run the Application
python admin.py

#Open in Browser
http://127.0.0.1:5000

