# --- Imports and Model Loading ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
from PIL import Image, ImageEnhance
import xgboost as xgb
from io import BytesIO
import os
import plotly.graph_objects as go
import tempfile

preprocessor = joblib.load("data/preprocessor.joblib")
model = xgb.Booster()
model.load_model("model/diabetes_model.xgb")
explainer = shap.TreeExplainer(model)
with open("data/feature_names.pkl", "rb") as f:
    feature_names = joblib.load(f)

# --- parse_feature ---
def parse_feature(encoded_feature_name, raw_input):
    base_name = encoded_feature_name.split("__")[-1]
    for col in raw_input.columns:
        if base_name.lower() in col.lower():
            return col, raw_input.iloc[0][col]
    return encoded_feature_name, "Unknown"


# --- get_reference_range ---
def get_reference_range(feature):
    ranges = {
        "bmi": "Normal: 18.5–24.9 kg/m²",
        "fasting_blood_sugar": "Normal: <100 mg/dL, Prediabetes: 100–125, Diabetes: ≥126",
        "hba1c": "Normal: <5.7%, Prediabetes: 5.7–6.4%, Diabetes: ≥6.5%",
        "cholesterol": "Desirable: <200 mg/dL, Borderline: 200–239, High: ≥240",
        "sleep": "Recommended: 7–9 hours",
        "stress": "Ideal: <7 on a 10-point scale",
        "screen_time": "Recommended: ≤4 hours/day",
        "fast_food": "Recommended: ≤1 meal/week",
        "genetic_risk_score": "1 = Low risk, 10 = High risk",
        "age": "Increased risk ≥45 years",
    }
    return ranges.get(feature, "Clinical guidance varies by context.")

# --- get_intervention ---
def get_intervention(feature, value):
    feature = feature.lower().strip()
    if isinstance(value, str):
        value = value.strip().lower()
    if "bmi" in feature:
        if value < 18.5:
            return "Underweight. Consult a dietitian to achieve a healthy weight."
        elif value >= 25 and value < 30:
            return "Overweight. Begin a structured exercise and dietary program."
        elif value >= 30:
            return "Obesity. Medical management or bariatric consultation recommended."

    if "fasting_blood_sugar" in feature:
        if value >= 100 and value < 126:
            return "Impaired fasting glucose. Initiate lifestyle interventions."
        elif value >= 126:
            return "Diabetes-range fasting glucose. Confirm with repeat testing and initiate care."

    if "hba1c" in feature:
        if value >= 5.7 and value < 6.5:
            return "Prediabetes range HbA1c. Start lifestyle modifications urgently."
        elif value >= 6.5:
            return "Diabetes-range HbA1c. Formal diagnosis and management needed."

    if "cholesterol" in feature:
        if value >= 200:
            return "High cholesterol. Recommend lipid panel, dietary changes, and consider statin therapy."

    if "sleep" in feature:
        if value < 6:
            return "Insufficient sleep. Aim for 7-9 hours to support metabolic health."

    if "stress" in feature:
        if value >= 7:
            return "High stress. Engage in mindfulness, therapy, or stress reduction programs."

    if "screen_time" in feature:
        if value > 6:
            return "Excessive screen time. Limit to <4 hours/day and increase physical activity."

    if "fast_food" in feature:
        if value > 3:
            return "High fast food intake. Reduce frequency to less than once per week."

    if "smoking" in feature:
        if value == "Yes":
            return "Smoking increases diabetes and cardiovascular risk. Immediate cessation advised."

    if "alcohol" in feature:
        if value == "Yes":
            return "Moderate or eliminate alcohol to reduce metabolic risk."

    if "family_history_diabetes" in feature:
        if value == "Yes":
            return "Family history present. Begin preventive lifestyle changes immediately."

    if "genetic_risk_score" in feature:
        if value >= 7:
            return "High genetic risk. Focus on strict lifestyle management to offset risk."

    if "physical_activity_level" in feature:
        if value == "Sedentary":
            return "Sedentary lifestyle. Target 150+ minutes of moderate exercise weekly."

    if "dietary_habits" in feature:
        if value == "Unhealthy":
            return "Adopt a Mediterranean or DASH-style eating plan rich in fruits and vegetables."

    if "prediabetes" in feature:
        if value == "Yes":
            return "Confirmed prediabetes. Initiate diabetes prevention program immediately."

    if "parent_diabetes_type" in feature:
        if value in ["Type 1", "Type 2"]:
            return "Parental diabetes history noted. Increase frequency of metabolic screening."

    if "region" in feature:
        return "Adjust interventions based on local dietary and lifestyle practices."

    if "family_income" in feature:
        if value < 300000:
            return "Consider socioeconomic factors in healthcare accessibility."

    if "age" in feature:
        if value > 45:
            return "Age >45 years. Screen annually for diabetes."

    if feature in [
        "bmi", "fasting_blood_sugar", "hba1c", "cholesterol", "sleep",
        "stress", "screen_time", "fast_food", "genetic_risk_score", "age"
    ]:
        return "Monitor and manage this parameter per clinical guidelines."

    return "No guideline-based recommendation available for this parameter."

# --- sanitize_text ---
def sanitize_text(text, max_length=1000):
    sanitized = (text.replace("≥", ">=")
                    .replace("≤", "<=")
                    .replace("–", "-")
                    .replace("—", "-")
                    .replace("’", "'").replace("‘", "'")
                    .replace("•", "*").replace("\n", " ").replace("\r", " "))
    return sanitized[:max_length] + "..." if len(sanitized) > max_length else sanitized

# --- prepare_transparent_logo ---
def prepare_transparent_logo(path, output_path, alpha=0.1):
    img = Image.open(path).convert("RGBA")
    alpha_layer = img.split()[3]
    alpha_layer = ImageEnhance.Brightness(alpha_layer).enhance(alpha)
    img.putalpha(alpha_layer)
    img.save(output_path)

prepare_transparent_logo("data/logo.png", "data/logo_faint.png", alpha=0.08)


# --- PDF Class ---
class PDF(FPDF):
    def __init__(self, name, age):
        super().__init__()
        self.patient_name = name
        self.patient_age = age

    def header(self):
        self.set_font("Arial", 'B', 16)
        self.cell(0, 10, "PrediX: Diabetes Risk Report", ln=True, align="C")
        self.ln(5)
        self.set_font("Arial", '', 11)
        today = datetime.today().strftime("%b %d, %Y")
        self.cell(70, 8, f"Patient: {self.patient_name}", border=0, ln=0, align="L")
        self.cell(50, 8, f"Age: {self.patient_age}", border=0, ln=0, align="C")
        self.cell(70, 8, f"Date: {today}", border=0, ln=1, align="R")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(8)
        if os.path.exists("data/logo_faint.png"):
            self.image("data/logo_faint.png", x=35, y=80, w=140)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", 'I', 9)
        self.cell(0, 10, f"Page {self.page_no()}    © 2025 PrediX Clinical AI", 0, 0, 'C')

# --- Streamlit UI + Logic ---
st.title("PrediX: Diabetes Risk Prediction and Clinical Report")
st.subheader("AI-powered medical-grade output with personalized care guidance")

name = st.text_input("Patient Name", "")
age = st.number_input("Age (years)", 10, 100, step=1, value=25)

with st.form("input_form"):
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    region = st.selectbox("Region", ["North", "South", "East", "West", "Northeast", "Central"])

    income = st.number_input("Family Income (INR/year)", 100000, 2500000, step=10000)
    st.caption("\U0001F50E *Helps assess socioeconomic context impacting healthcare accessibility.*")

    genetic = st.slider("Genetic Risk Score (1-10)", 1, 10)
    st.caption("\U0001F50E *Indicates inherited risk of diabetes based on family history. A score of 10 suggests both parents or multiple close relatives have diabetes.*")

    bmi = st.number_input("BMI (kg/m²)", 16.0, 40.0)
    st.caption("\U0001F50E *Body Mass Index assesses body fat. Normal range: 18.5–24.9 kg/m².*")

    physical = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "Active"])
    st.caption("\U0001F50E *Describes routine activity level. Sedentary lifestyle increases metabolic risk.*")

    diet = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy"])
    st.caption("\U0001F50E *Reflects daily eating pattern. Unhealthy diets high in sugars/fats elevate diabetes risk.*")

    fast_food = st.slider("Fast Food Intake (meals/week)", 1, 10)
    st.caption("\U0001F50E *Frequent fast food consumption (>1 meal/week) is linked to insulin resistance.*")

    smoking = st.selectbox("Smoking", ["Yes", "No"])
    st.caption("\U0001F50E *Smoking increases cardiovascular and metabolic risks significantly.*")

    alcohol = st.selectbox("Alcohol Consumption", ["Yes", "No"])
    st.caption("\U0001F50E *Excessive alcohol intake impacts liver function and glucose metabolism.*")

    fbs = st.number_input("Fasting Blood Sugar (mg/dL)", 70.0, 180.0)
    st.caption("\U0001F50E *Fasting glucose levels indicate insulin efficiency. Normal: <100 mg/dL, Diabetes: ≥126.*")

    hba1c = st.number_input("HbA1c (%)", 4.0, 10.0)
    st.caption("\U0001F50E *Reflects 3-month average glucose levels. Normal: <5.7%, Diabetes: ≥6.5%.*")

    cholesterol = st.number_input("Cholesterol (mg/dL)", 120.0, 300.0)
    st.caption("\U0001F50E *Total blood cholesterol. Desirable: <200 mg/dL. High cholesterol increases diabetes complications.*")

    prediab = st.selectbox("Prediabetes Diagnosis", ["Yes", "No"])
    st.caption("\U0001F50E *Indicates a prior medical diagnosis of elevated but non-diabetic blood sugar levels.*")

    diabetes_type = st.selectbox("Parental Diabetes Type", ["None", "Type 1", "Type 2"])
    st.caption("\U0001F50E *Family history is a strong genetic predictor. Type 2 is more heritable.*")

    family_hist = st.selectbox("Family History of Diabetes", ["Yes", "No"])
    st.caption("\U0001F50E *If parents or siblings have diabetes, your risk increases substantially.*")

    sleep = st.number_input("Sleep Hours (per night)", 4.0, 10.0)
    st.caption("\U0001F50E *Less than 6 hours/night is associated with metabolic dysregulation.*")

    stress = st.slider("Stress Level (1-10)", 1, 10)
    st.caption("\U0001F50E *Chronic stress alters cortisol and insulin, affecting glucose control.*")

    screen = st.slider("Screen Time (hrs/day)", 1, 12)
    st.caption("\U0001F50E *Sedentary screen time >4 hrs/day is linked to reduced physical activity and obesity.*")

    submit = st.form_submit_button("Generate Report")
if submit:
    # Step 1: Validate patient name
    if not name.strip():
        st.error("Please enter the patient's name.")
        st.stop()

    # Step 2: Create a DataFrame for the raw input data
    raw_input = pd.DataFrame([{
        "Age": age, "Gender": gender, "Region": region, "Family_Income": income,
        "Genetic_Risk_Score": genetic, "BMI": bmi, "Physical_Activity_Level": physical,
        "Dietary_Habits": diet, "Fast_Food_Intake": fast_food, "Smoking": smoking,
        "Alcohol_Consumption": alcohol, "Fasting_Blood_Sugar": fbs, "HbA1c": hba1c,
        "Cholesterol_Level": cholesterol, "Prediabetes": prediab,
        "Parent_Diabetes_Type": diabetes_type, "Family_History_Diabetes": family_hist,
        "Sleep_Hours": sleep, "Stress_Level": stress, "Screen_Time": screen
    }])

    try:
        # Step 3: Preprocess the raw input and make predictions
        X_proc = preprocessor.transform(raw_input)
        X_df = pd.DataFrame(X_proc, columns=feature_names)  # Ensure correct feature names for SHAP
        dinput = xgb.DMatrix(X_proc)
        prediction = model.predict(dinput)[0]
        shap_values = explainer(X_df)  # Use DataFrame with correct feature names
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # Step 4: Generate SHAP plots
    # SHAP Beeswarm Plot
    beeswarm_buf = BytesIO()
    plt.figure(figsize=(10, 5))
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig(beeswarm_buf, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    beeswarm_buf.seek(0)

    # SHAP Waterfall Plot
    waterfall_buf = BytesIO()
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(waterfall_buf, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    waterfall_buf.seek(0)


    # Step 5: Create PDF report
    pdf = PDF(name, age)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, sanitize_text(f"Estimated time to progression to Type 2 Diabetes: {prediction:.1f} years"))
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 13)
    pdf.cell(0, 10, "Top Risk Factors & Recommendations", ln=True)
    pdf.ln(3)
    pdf.set_font("Arial", '', 11)

    # Step 6: Add SHAP values to DataFrame
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values.values[0]
    }).assign(Impact=lambda d: d["SHAP Value"].abs()).sort_values(by="Impact", ascending=False)

    # Step 7: Add recommendations based on SHAP values
    for idx, row in shap_df.iterrows():
        f_raw, val = parse_feature(row["Feature"], raw_input)
        if val == "Unknown":
            continue
        label = f_raw.replace("_", " ").title()
        intervention = get_intervention(f_raw.lower(), val)

        # No color logic anymore
        pdf.set_x(15)
        pdf.multi_cell(180, 8, sanitize_text(f"{label} (Input: {val}) - Impact: {row['SHAP Value']:.2f}"))
        pdf.set_font("Arial", 'BI', 10)
        pdf.set_x(15)
        pdf.multi_cell(180, 7, sanitize_text(intervention))
        pdf.ln(2)
        pdf.set_font("Arial", '', 11)

    # Step 8: Add graphs to PDF report
    def add_graph_page_from_buffer(buf, title, description):
        # Save buffer to temp .png file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(buf.read())
            tmp_file_path = tmp_file.name

        # Add page and image to PDF
        pdf.add_page()
        pdf.set_font("Arial", 'B', 13)
        pdf.cell(0, 10, title, ln=True, align="C")
        pdf.set_font("Arial", '', 11)
        pdf.ln(10)
        pdf.image(tmp_file_path, x=30, w=150)
        pdf.ln(8)
        pdf.multi_cell(0, 8, sanitize_text(description))

        # Remove the temp file after adding
        os.remove(tmp_file_path)

    # Add Beeswarm and Waterfall plots
    add_graph_page_from_buffer(beeswarm_buf, "SHAP Summary Plot", "This plot shows the impact of each feature on the predicted diabetes risk. Each point represents a patient's input for a feature—colored by value: blue = low, red = high. Features pushing the prediction higher appear on the right, and those lowering it on the left.")
    add_graph_page_from_buffer(waterfall_buf, "SHAP Waterfall Plot", "This chart explains how your individual features contributed to the final risk prediction. Red bars increase risk, blue bars reduce it. It starts at the model's baseline and ends at your final prediction. Each label shows a feature from your inputs.")

    # Step 9: Output the PDF report
    pdf_output = BytesIO()
    pdf.output(pdf_output, 'F')
    pdf_output.seek(0)

    st.download_button("Download Your Report", pdf_output, file_name="Diabetes_Risk_Report.pdf", mime="application/pdf")

    # Step 10: Display risk interpretation
    st.subheader("Risk Interpretation")
    if prediction < 3:
        st.error(f"🔴 High Risk: {prediction:.1f} years to progression")
    elif 3 <= prediction <= 6:
        st.warning(f"🟠 Moderate Risk: {prediction:.1f} years to progression")
    else:
        st.success(f"🟢 Low Risk: {prediction:.1f} years to progression")

    # Step 11: Add Gauge chart for visual representation of risk
    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={"text": "Estimated Years of Progression to Type 2 Diabetes"},
        gauge={
            'axis': {'range': [0, 10]},
            'steps': [
                {'range': [0, 3], 'color': "red"},
                {'range': [3, 6], 'color': "orange"},
                {'range': [6, 10], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))
    st.plotly_chart(fig)
