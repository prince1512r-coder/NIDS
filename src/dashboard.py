import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Network Intrusion Detection System",
    layout="wide"
)

st.title("🛡️ Network Intrusion Detection System")

st.markdown("""
Upload a **UNSW-NB15 CSV file** to analyze network traffic.
""")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Uploaded Successfully!")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])