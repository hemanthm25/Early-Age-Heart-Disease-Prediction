# Early-Age-Heart-Disease-Prediction
Machine learning–based Early Age Heart Disease Prediction with Streamlit web app and PDF report generation. 

📌 Overview
Heart disease remains one of the leading causes of global mortality, increasingly affecting younger age groups. Early detection plays a vital role in prevention and treatment.This project uses Machine Learning to predict the likelihood of heart disease based on clinical parameters, helping users and healthcare providers identify risk early, quickly, and effectively.The system combines data preprocessing, multiple ML models, performance evaluation, and a Streamlit-based interactive web application to provide real-time predictions.

🎯 Project Highlights
Predicts heart disease using 13 clinical features.
Implements 5 ML models (Random Forest, Decision Tree, Logistic Regression, SVM, KNN).
Interactive Streamlit UI for user-friendly predictions.
Generates downloadable PDF health reports using FPDF.
Includes model comparison, ROC curves, confusion matrices, and feature importance visualizations.

🧠 Problem Statement
Increasing numbers of young individuals are affected by cardiac conditions, yet early screening is limited, costly, and time-consuming.
The challenge is to create a solution that is:
✔ Accessible
✔ Cost-effective
✔ Reliable
✔ Easy to use even for non-experts
This system aims to predict heart disease early based on user-input health indicators.

🎯 Objectives
Main Objective
Develop a machine-learning based system that predicts whether a person is at risk of heart disease.

Specific Objectives
Data preprocessing, cleaning & normalization
Train multiple ML models and compare performance
Deploy a user-friendly prediction interface
Generate downloadable PDF reports
Encourage early awareness and medical consultation

📂 Dataset Description
Source: Kaggle – Heart Disease Dataset
Records: 1026 rows
Features: 13 clinical inputs
Target: 1 (Disease) / 0 (No disease)
Includes: Age, cholesterol, chest pain type, BP, sugar, ECG, maximum heart rate, etc.

🛠️ Tech Stack
Programming
Python
Streamlit
FPDF (PDF generation)
Libraries
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
Joblib

⚙️ System Architecture
Dataset → Preprocessing → Train/Test Split → Model Training → Evaluation → Streamlit App → Prediction & Report

Workflow Diagram
(Reference: Figure 6.1 – Project Workflow)

🧪 Models Used
Model	Accuracy
Random Forest	98%
Decision Tree	98%
SVM	88%
KNN	83%
Logistic Regression	79%
Final Chosen Model: Random Forest (best stability & feature importance)

📊 Evaluation Metrics
Accuracy
Confusion Matrix
ROC-AUC Curve
Risk Score Probability

🖥️ Application Screenshots

Add images from your /images/ folder, for example:

![Main Interface](images/main_interface.png)
![Input Form](images/input_form.png)
![Prediction Result](images/prediction_result.png)
![Model Comparison](images/model_comparison.png)
(Refer to Figures 8.1 - 8.18 for exact visuals.)


🚀 How to Run the Project
1. Clone the Repo
git clone https://github.com/<your-username>/early-age-heart-disease-prediction.git
cd early-age-heart-disease-prediction

2. Install Dependencies
pip install -r requirements.txt

3. Run the Streamlit App
streamlit run streamlit_app.py

4. Upload Data & Start Predictions

Open the local URL Streamlit provides:
http://localhost:8501

📄 PDF Report Generation
The app automatically generates a structured PDF report containing:
Patient details
Prediction result
Risk score
Clinical values entered

🔮 Future Scope
Integrate with live wearable data
Add doctor recommendation engine
Deploy on cloud (AWS/GCP/Render)
Multi-disease prediction

🏁 Conclusion
This project demonstrates the potential of machine learning in delivering early heart disease detection through a simple, intuitive, and efficient system.
It highlights the power of data-driven healthcare and the future of preventive diagnostics.

👥 Team
Hemanth M
Shalini B
Thejas B S
Varshitha S

🧑‍🏫 Guide
Prof. Sathya Sheela D, Assistant Professor
Dept. of CS&D, KSIT Bengaluru


FINAL REPORT
