import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# 🔄 Load Model, Scaler, Feature Names
# ===============================
model = joblib.load("model/random_forest_model.pkl")
scaler = joblib.load("model/standard_scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")  # list of column names used during training

# ===============================
# 🎯 Prediksi
# ===============================
def predict_dropout(input_df):
    input_df = input_df.copy()

    # Tambahkan kolom yang hilang dengan nilai default 0
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Urutkan kolom sesuai saat training
    input_df = input_df[feature_names]

    # Scaling
    scaled_input = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    return prediction, probability

# ===============================
# 🖼️ UI Streamlit
# ===============================
st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="centered")
st.title("🎓 Prediksi Dropout Mahasiswa")
st.markdown("Masukkan data mahasiswa untuk memprediksi risiko dropout.")

# 🎛️ Form Input
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Umur Saat Masuk (Age_at_enrollment)", 15, 60, 18)
    admission_grade = st.number_input("Admission Grade", 0.0, 200.0, 120.0)
    curricular_enrolled = st.number_input("CU 1st Sem Enrolled", 0, 50, 6)

with col2:
    tuition_paid = st.selectbox("Uang Kuliah Lancar? (Tuition_fees_up_to_date)", [0, 1])
    scholarship = st.selectbox("Penerima Beasiswa? (Scholarship_holder)", [0, 1])
    debtor = st.selectbox("Ada Tunggakan? (Debtor)", [0, 1])

# Susun input menjadi DataFrame
user_input = {
    "Age_at_enrollment": [age],
    "Admission_grade": [admission_grade],
    "Curricular_units_1st_sem_enrolled": [curricular_enrolled],
    "Tuition_fees_up_to_date": [tuition_paid],
    "Scholarship_holder": [scholarship],
    "Debtor": [debtor]
}

input_df = pd.DataFrame(user_input)

# 🔍 Prediksi
if st.button("🔎 Prediksi Dropout"):
    pred, prob = predict_dropout(input_df)

    if pred == 1:
        st.error(f"⚠️ Siswa berisiko dropout! Probabilitas: {prob:.2%}")
    else:
        st.success(f"✅ Siswa **tidak berisiko** dropout. Probabilitas: {prob:.2%}")
