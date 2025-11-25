import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from fpdf import FPDF
from datetime import datetime
import base64

# Load pre-trained models and scaler
rf_model = joblib.load('random_forest_model.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')
svc_model = joblib.load('svc_model.pkl')
knn_model = joblib.load('kneighbors_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature list with descriptions
FEATURES = {
    'age': "Age in years",
    'sex': "Sex (1 = male, 0 = female)",
    'cp': "Chest Pain Type (0-typical, 1-atypical, 2-non-anginal, 3-asymptomatic)",
    'trestbps': "Resting Blood Pressure (mm Hg)",
    'chol': "Serum Cholesterol (mg/dl)",
    'fbs': "Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)",
    'restecg': "Resting ECG (0-normal, 1-ST-T abnormality, 2-left ventricular hypertrophy)",
    'thalach': "Max Heart Rate Achieved",
    'exang': "Exercise Induced Angina (1 = yes, 0 = no)",
    'oldpeak': "ST depression induced by exercise",
    'slope': "Slope of ST segment (0-up, 1-flat, 2-down)",
    'ca': "Number of major vessels (0-3)",
    'thal': "Thalassemia (1-normal, 2-fixed defect, 3-reversible defect)"
}

st.title("💓 Heart Disease Prediction App")
st.write("This app uses machine learning models to predict the **likelihood of heart disease** based on your input.")

# Input section in sidebar
st.sidebar.header("📝 Input Patient Data")

# Input fields for patient name and hospital
patient_name = st.sidebar.text_input("Enter Patient's Name")
hospital_name = st.sidebar.text_input("Enter Hospital's Name")

user_inputs = {}
for feature, description in FEATURES.items():
    st.sidebar.markdown(f"**{feature}** – {description}")
    if feature in ['sex', 'fbs', 'exang']:
        user_inputs[feature] = st.sidebar.selectbox(f"{description}", options=[0, 1], key=feature)
    elif feature == 'cp':
        user_inputs[feature] = st.sidebar.selectbox(f"{description}", options=[0, 1, 2, 3], key=feature)
    elif feature == 'restecg':
        user_inputs[feature] = st.sidebar.selectbox(f"{description}", options=[0, 1, 2], key=feature)
    elif feature == 'slope':
        user_inputs[feature] = st.sidebar.selectbox(f"{description}", options=[0, 1, 2], key=feature)
    elif feature == 'ca':
        user_inputs[feature] = st.sidebar.selectbox(f"{description}", options=[0, 1, 2, 3], key=feature)
    elif feature == 'thal':
        user_inputs[feature] = st.sidebar.selectbox(f"{description}", options=[1, 2, 3], key=feature)
    elif feature == 'oldpeak':
        user_inputs[feature] = st.sidebar.number_input(f"{description}", min_value=0.0, max_value=10.0, step=0.1, key=feature)
    elif feature == 'age':
        user_inputs[feature] = st.sidebar.number_input(f"{description}", min_value=20, max_value=100, step=1, key=feature)
    elif feature == 'trestbps':
        user_inputs[feature] = st.sidebar.number_input(f"{description}", min_value=80, max_value=200, step=1, key=feature)
    elif feature == 'chol':
        user_inputs[feature] = st.sidebar.number_input(f"{description}", min_value=100, max_value=600, step=1, key=feature)
    elif feature == 'thalach':
        user_inputs[feature] = st.sidebar.number_input(f"{description}", min_value=70, max_value=210, step=1, key=feature)
    else:
        user_inputs[feature] = st.sidebar.number_input(f"{description}", key=feature)

user_data = pd.DataFrame(user_inputs, index=[0])
user_data_scaled = scaler.transform(user_data)

# Model selection
model_choice = st.selectbox("Select the Model for Prediction", ['Random Forest', 'Logistic Regression', 'Decision Tree', 'SVC', 'KNN'])

# Model mapping
model_map = {
    'Random Forest': rf_model,
    'Logistic Regression': lr_model,
    'Decision Tree': dt_model,
    'SVC': svc_model,
    'KNN': knn_model
}
model = model_map[model_choice]

# Initialize prediction variable to handle potential NameError
prediction = None
risk_score = None

# Check if the prediction logic is inside the button click block
if st.button('🔍 Predict'):
    # Ensure prediction is properly set
    prediction = model.predict(user_data_scaled)[0]
    
    if hasattr(model, "predict_proba"):
        risk_score = model.predict_proba(user_data_scaled)[0][1] * 100
        st.subheader(f"💡 Risk Score: {risk_score:.2f}%")
    
    if prediction == 1:
        st.error("🔴 **Prediction: Heart Disease Detected**")
    else:
        st.success("🟢 **Prediction: No Heart Disease Detected**")

    st.subheader("📋 Input Summary")
    user_data_transposed = user_data.T.rename(columns={0: 'Value'})
    st.dataframe(user_data_transposed)


# Generate PDF Report
if patient_name and hospital_name:  # Ensure both fields are filled
    report_date = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Heart Disease Prediction Report", ln=True, align='C')  # Removed emoji
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.cell(200, 10, txt=f"Hospital: {hospital_name}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {report_date}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Model Used: {model_choice}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {'Heart Disease' if prediction == 1 else 'No Disease'}", ln=True)
    if risk_score is not None:
        pdf.cell(200, 10, txt=f"Risk Score: {risk_score:.2f}%", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Patient Input Data:", ln=True)
    for k, v in user_inputs.items():
        pdf.cell(200, 10, txt=f"{FEATURES[k]}: {v}", ln=True)

    pdf_output = f"{patient_name}_Heart_Report.pdf"
    pdf.output(pdf_output)

    with open(pdf_output, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="{pdf_output}">📥 Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)
else:
    st.error("Please enter the patient's name and hospital name before generating the report.")

# Compare all models
st.subheader("📊 Predict Using All Models")
if st.button('🚀 Compare Across All Models'):
    comparison = []
    for name, model in model_map.items():
        pred = model.predict(user_data_scaled)[0]
        risk = model.predict_proba(user_data_scaled)[0][1] * 100 if hasattr(model, 'predict_proba') else None
        comparison.append({
            'Model': name,
            'Prediction': 'Heart Disease' if pred == 1 else 'No Disease',
            'Risk Score (%)': f"{risk:.2f}" if risk is not None else "N/A"
        })
    st.dataframe(pd.DataFrame(comparison))

# Evaluation section
if st.checkbox("📈 Show Model Evaluation Metrics"):
    y_test_dummy = np.random.randint(0, 2, 100)
    dummy_data = pd.DataFrame(np.random.rand(100, len(FEATURES)), columns=FEATURES.keys())
    X_dummy_scaled = scaler.transform(dummy_data)

    y_pred_dummy = model.predict(X_dummy_scaled)
    cm = confusion_matrix(y_test_dummy, y_pred_dummy)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    # ROC Curve
    if hasattr(model, "predict_proba"):
        fpr, tpr, _ = roc_curve(y_test_dummy, model.predict_proba(X_dummy_scaled)[:, 1])
        roc_auc = auc(fpr, tpr)

        st.subheader("ROC Curve")
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="blue")
        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()
        st.pyplot(fig_roc)

# Feature Importance section
if st.checkbox("📊 Show Feature Importance"):
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance (Global)")
        importance_df = pd.DataFrame({
            'Feature': list(FEATURES.keys()),
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        fig_imp, ax_imp = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis", ax=ax_imp)
        ax_imp.set_title("Feature Importance")
        st.pyplot(fig_imp)
    else:
        st.info("Selected model does not support feature importance display.")

# Footer
st.markdown("---")
st.markdown("""
### ℹ️ About  
This tool was built to demonstrate the predictive power of machine learning models in medical diagnosis, especially for **heart disease**.  
Enter your clinical values on the left and select any model to test real-time predictions.
""")
