import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# 🔄 Load Model, Scaler, Feature Names
# ===============================
model = joblib.load("model/random_forest_model.pkl")
scaler = joblib.load("model/standard_scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")

# ===============================
# 🎯 Fungsi Prediksi
# ===============================
def predict_dropout(input_df):
    input_df = input_df.copy()

    # Tambahkan kolom yang tidak ada dengan nilai default 0
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Susun ulang urutan kolom sesuai saat training
    input_df = input_df[feature_names]

    # Scaling
    scaled_input = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    return prediction, probability

# ===============================
# 🖼️ Tampilan Streamlit
# ===============================
st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="centered")
st.title("🎓 Prediksi Dropout Mahasiswa")
st.markdown("""
Aplikasi ini menggunakan model machine learning untuk memprediksi apakah seorang mahasiswa berisiko **dropout** (berhenti studi).  
Silakan isi informasi berikut berdasarkan data mahasiswa yang bersangkutan.  
*Kolom lain yang dibutuhkan oleh model akan diisi otomatis dengan nilai default (0).*
""")

# 🎛️ Form Input
st.subheader("📝 Form Input Data Mahasiswa")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Umur Saat Masuk (Age_at_enrollment)",
        min_value=15, max_value=60, value=18,
        help="Umur mahasiswa saat pertama kali mendaftar kuliah. Rentang umum: 17–30 tahun."
    )

    admission_grade = st.number_input(
        "Admission Grade",
        min_value=0.0, max_value=200.0, value=120.0,
        help="Nilai ujian masuk atau skor awal pendaftaran. Rentang: 0–200."
    )

    curricular_enrolled = st.number_input(
        "CU 1st Sem Enrolled",
        min_value=0, max_value=50, value=6,
        help="Jumlah mata kuliah yang diambil pada semester 1. Umumnya 4–10 mata kuliah."
    )

with col2:
    tuition_paid = st.selectbox(
        "Uang Kuliah Lancar? (Tuition_fees_up_to_date)",
        options=[1, 0],
        format_func=lambda x: "Ya" if x == 1 else "Tidak",
        help="Apakah mahasiswa membayar uang kuliah tepat waktu?"
    )

    scholarship = st.selectbox(
        "Penerima Beasiswa? (Scholarship_holder)",
        options=[1, 0],
        format_func=lambda x: "Ya" if x == 1 else "Tidak",
        help="Apakah mahasiswa ini menerima beasiswa akademik atau finansial?"
    )

    debtor = st.selectbox(
        "Ada Tunggakan? (Debtor)",
        options=[1, 0],
        format_func=lambda x: "Ya" if x == 1 else "Tidak",
        help="Apakah mahasiswa memiliki tunggakan administrasi atau keuangan?"
    )

# Susun input ke DataFrame
user_input = {
    "Age_at_enrollment": [age],
    "Admission_grade": [admission_grade],
    "Curricular_units_1st_sem_enrolled": [curricular_enrolled],
    "Tuition_fees_up_to_date": [tuition_paid],
    "Scholarship_holder": [scholarship],
    "Debtor": [debtor]
}

input_df = pd.DataFrame(user_input)

# ===============================
# 🔍 Prediksi + Rekomendasi
# ===============================
if st.button("🔎 Prediksi Dropout"):
    pred, prob = predict_dropout(input_df)

    if pred == 1:
        st.error(f"⚠️ Mahasiswa diprediksi **berisiko dropout**.\n\nProbabilitas: {prob:.2%}")
        st.markdown("### 📌 Rekomendasi Tindakan:")
        st.markdown("""
        - 💬 **Berikan bimbingan akademik dan konseling secara personal.**
        - 💸 **Tinjau ulang kondisi finansial mahasiswa** (cek status pembayaran & beasiswa).
        - 📊 **Evaluasi beban studi**: pertimbangkan pengurangan SKS bila perlu.
        - 🧾 Libatkan wali/orangtua jika diperlukan untuk pendampingan.
        """)
    else:
        st.success(f"✅ Mahasiswa **tidak berisiko** dropout.\n\nProbabilitas: {prob:.2%}")
        st.markdown("### 📌 Rekomendasi:")
        st.markdown("""
        - 👍 Pertahankan performa akademik saat ini.
        - 👁️ Tetap lakukan pemantauan rutin dan deteksi dini untuk semester berikutnya.
        """)
