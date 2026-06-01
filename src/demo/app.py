# [M5] Demo Streamlit — chay: streamlit run src/demo/app.py
# KHUNG: dien logic load model + infer khi M4 co checkpoint.
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Carotid Multimodal Demo", layout="wide")
st.title("🫀 Chẩn đoán Xơ vữa Động mạch cảnh — Demo Đa phương thức")
st.caption("Master-UIT-MedSignal · mô hình giả lập lâm sàng (không dùng cho chẩn đoán thật)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Chỉ số lâm sàng")
    age = st.number_input("Age", 18, 100, 60)
    sex = st.selectbox("Sex", ["Male", "Female"])
    lpa = st.number_input("Lp(a) mg/dL", 0.0, 300.0, 30.0)
    apob = st.number_input("ApoB mg/dL", 0.0, 300.0, 100.0)
    ldl = st.number_input("LDL-C mg/dL", 0.0, 400.0, 130.0)
    tg = st.number_input("Triglyceride mg/dL", 0.0, 600.0, 120.0)
    tc = st.number_input("Total Cholesterol mg/dL", 0.0, 500.0, 190.0)
    nonhdl = st.number_input("Non-HDL mg/dL", 0.0, 400.0, 150.0)
    imt = st.number_input("IMT mm", 0.0, 3.0, 0.7)

with col2:
    st.subheader("Ảnh siêu âm")
    imt_file = st.file_uploader("Ảnh IMT (bắt buộc)", type=["png"])
    cca_files = st.file_uploader("Ảnh CCA (0 hoặc 4)", type=["png"],
                                 accept_multiple_files=True)

if st.button("Dự đoán"):
    # M5 TODO:
    #   1. Load checkpoint MultimodalFusion + scaler (do M4 luu).
    #   2. Tien xu ly input giong CarotidDataset (scale tabular, doc anh).
    #   3. Infer -> hien thi: P(plaque), echogenicity, risk score + Grad-CAM overlay.
    st.warning("KHUNG demo — cắm model đã train (checkpoint từ M4) vào đây.")
    st.info("Sẽ hiển thị: xác suất mảng xơ vữa, độ phản hồi âm, điểm nguy cơ + heatmap Grad-CAM.")
