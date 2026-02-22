import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import time
from io import BytesIO

import plotly.graph_objects as go
import plotly.express as px

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ============================
# Import Models
# ============================
from classical_ml.svm_model import run_svm
from classical_ml.logistic_model import run_logistic
from classical_ml.xgb_model import run_xgb
from quantum_ml.vqc_model import run_vqc
from quantum_ml.qsvm_model import run_qsvm
from utils.data_loader import load_data

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="Early Autism Risk Screening Using Quantum Machine Learning",
    layout="centered"
)

# ============================
# Title
# ============================
st.markdown(
    """
    <h1 style='text-align:center'>
    Early Autism Risk Screening Using Quantum Machine Learning
    </h1>
    <p style='text-align:center'>
    Behavioural screening for toddlers (12–60 months)
    </p>
    """,
    unsafe_allow_html=True
)

# ============================
# Sidebar
# ============================
st.sidebar.title("Model Selection")

model_choice = st.sidebar.radio(
    "Select Model",
    [
        "SVM",
        "Logistic Regression",
        "XGBoost",
        "Quantum VQC",
        "Quantum SVM (QSVM)"
    ]
)

# ============================
# Questionnaire
# ============================
st.subheader("Behavioural Questionnaire")
st.caption("Scale: 1 = Never | 10 = Always")

questions = [
    "Difficulty maintaining eye contact",
    "Limited response to name",
    "Low interest in social interaction",
    "Sensitivity to sounds or lights",
    "Difficulty understanding emotions",
    "Distress with routine changes",
    "Repetitive behaviours",
    "Delayed communication",
    "Overwhelm in crowds",
    "Prefers to play alone"
]

responses = []
for i, q in enumerate(questions):
    responses.append(st.slider(f"Q{i+1}. {q}", 1, 10, 5))

responses = np.array(responses)

# ============================
# Risk Gauge
# ============================
def show_risk_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Autism Risk Score (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"},
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# ============================
# Prediction
# ============================
if st.button("Generate Screening Report"):

    start = time.time()

    user_input = (responses / 10).reshape(1, -1)

    acc = None
    confidence = None

    # ----------------------------
    # Classical Models
    # ----------------------------
    if model_choice == "SVM":
        pred, acc, confidence = run_svm(user_input)

    elif model_choice == "Logistic Regression":
        pred, acc, confidence = run_logistic(user_input)

    elif model_choice == "XGBoost":
        pred, confidence, acc = run_xgb(user_input)

    # ----------------------------
    # Quantum Models
    # ----------------------------
    elif model_choice == "Quantum VQC":
        df = load_data()
        acc = run_vqc(df)   # assumes run_vqc internally handles splitting

    elif model_choice == "Quantum SVM (QSVM)":
        df = load_data()
        acc = run_qsvm(df)  # QSVM internally handles splitting

    end = time.time()

    # ----------------------------
    # Risk Calculation
    # ----------------------------
    risk_score = int(np.sum(responses))

    if risk_score <= 25:
        label = "LOW RISK"
    elif risk_score <= 50:
        label = "MODERATE RISK"
    elif risk_score <= 75:
        label = "HIGH RISK"
    else:
        label = "VERY HIGH RISK"

    st.success(label)
    show_risk_gauge(risk_score)

    # ----------------------------
    # Model Output
    # ----------------------------
    st.markdown("### Model Output")
    st.write(f"Risk Score: {risk_score} / 100")
    st.write(f"Model Accuracy: {acc}")
    st.write(f"Execution Time: {end - start:.2f} seconds")

    # ----------------------------
    # Behaviour Graph
    # ----------------------------
    fig = px.bar(
        x=[f"Q{i+1}" for i in range(10)],
        y=responses,
        labels={"x": "Question", "y": "Response (1–10)"},
        title="Behavioural Response Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Report Generation
    # ----------------------------
    report = {
        "Model": model_choice,
        "Risk Score": risk_score,
        "Accuracy": acc,
        "Generated On": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # PDF
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    y_pos = 800
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "Autism Risk Screening Report")
    y_pos -= 40
    c.setFont("Helvetica", 10)

    for k, v in report.items():
        c.drawString(50, y_pos, f"{k}: {v}")
        y_pos -= 18

    c.save()
    pdf_buffer.seek(0)

    st.download_button(
        "Download PDF Report",
        pdf_buffer,
        file_name="autism_screening_report.pdf",
        mime="application/pdf"
    )

    # Excel
    df_report = pd.DataFrame(list(report.items()), columns=["Field", "Value"])
    excel_buffer = BytesIO()
    df_report.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)

    st.download_button(
        "Download Excel Report",
        excel_buffer,
        file_name="autism_screening_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.warning(
        "This tool is for early screening only and does NOT replace medical diagnosis."
    )

# ============================
# Team Section
# ============================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center">
        <h3>Project Developed By</h3>
        <p><b>Asthra Addagoda</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")