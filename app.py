import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# 🔄 Load Model dan Scaler
# ===============================
model = joblib.load("model/random_forest_model.pkl")
scaler = joblib.load("model/standard_scaler.pkl")

# ===============================
# 🎯 Fungsi Prediksi
# ===============================
def predict_dropout(input_df):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]
    return prediction, probability

# ===============================
# 🖼️ UI - Streamlit App
# ===============================
st.set_page_config(page_title="Prediksi Dropout Siswa", layout="centered")

st.title("🎓 Aplikasi Prediksi Dropout Mahasiswa")
st.markdown("Masukkan data siswa di bawah ini untuk memprediksi apakah siswa berisiko mengalami **dropout**.")

# Contoh input sederhana (ubah sesuai dataset Anda)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Umur Saat Masuk (Age_at_enrollment)", min_value=15, max_value=60, value=18)
    admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=200.0, value=120.0)
    curricular_units_enrolled = st.number_input("CU 1st Sem Enrolled", min_value=0, max_value=50, value=6)

with col2:
    tuition_up_to_date = st.selectbox("Uang Kuliah Lancar?", [0, 1])
    scholarship_holder = st.selectbox("Penerima Beasiswa?", [0, 1])
    debtor = st.selectbox("Ada Tunggakan?", [0, 1])

# Buat DataFrame dari input
input_dict = {
    "Age_at_enrollment": [age],
    "Admission_grade": [admission_grade],
    "Curricular_units_1st_sem_enrolled": [curricular_units_enrolled],
    "Tuition_fees_up_to_date": [tuition_up_to_date],
    "Scholarship_holder": [scholarship_holder],
    "Debtor": [debtor],
}

input_df = pd.DataFrame(input_dict)

# ===============================
# 🔍 Prediksi
# ===============================
if st.button("🔎 Prediksi Dropout"):
    pred, prob = predict_dropout(input_df)

    if pred == 1:
        st.error(f"⚠️ Siswa berisiko dropout dengan probabilitas {prob:.2%}")
    else:
        st.success(f"✅ Siswa **tidak berisiko** dropout (Probabilitas dropout: {prob:.2%})")
