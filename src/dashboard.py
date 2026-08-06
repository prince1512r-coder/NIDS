import streamlit as st
import pandas as pd
import joblib
from predictor import predict_dataset

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Network Intrusion Detection System",
    layout="wide"
)

st.title("🛡️ Network Intrusion Detection System")

st.markdown(
    """
Upload a **Network Traffic CSV File** for intrusion detection.
"""
)

# --------------------------------------------------
# Load Models
# --------------------------------------------------

MODEL_PATH = r"C:\GITHUB\NIDS\models\binary_model.pkl"
MULTI_MODEL_PATH = r"C:\GITHUB\NIDS\models\multiclass_model.pkl"
ENCODER_PATH = r"C:\GITHUB\NIDS\models\encoders.pkl"

try:

    binary_model = joblib.load(MODEL_PATH)
    multiclass_model = joblib.load(MULTI_MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)

    st.success("✅ Models Loaded Successfully")

except Exception as e:

    st.error("❌ Unable to load trained models.")
    st.code(str(e))

# --------------------------------------------------
# Upload CSV
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose CSV File",
    type=["csv"]
)

# --------------------------------------------------
# Required Columns
# --------------------------------------------------

required_columns = [
    "id",
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports"
]

# --------------------------------------------------
# Read Uploaded CSV
# --------------------------------------------------

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset Uploaded Successfully")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    # --------------------------------------------------
    # Validate Dataset
    # --------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    st.subheader("Dataset Validation")

    if len(missing_columns) == 0:

        st.success("✅ Dataset is ready for prediction.")

        # --------------------------------------------------
        # Predict Button
        # --------------------------------------------------

        if st.button("🚀 Predict Network Traffic"):

            with st.spinner("Analyzing Network Traffic..."):

                try:

                    results = predict_dataset(df)

                    st.success("✅ Prediction Completed Successfully!")

                    # ------------------------------------------
                    # Results Table
                    # ------------------------------------------

                    st.subheader("Prediction Results")

                    st.dataframe(results)

                    # ------------------------------------------
                    # Summary
                    # ------------------------------------------

                    total_records = len(results)

                    normal_records = (
                        results["Binary Prediction"] == "Normal"
                    ).sum()

                    attack_records = (
                        results["Binary Prediction"] == "Attack"
                    ).sum()

                    attack_percentage = (
                        attack_records / total_records
                    ) * 100

                    st.subheader("Summary")

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric(
                        "Total Records",
                        total_records
                    )

                    c2.metric(
                        "Normal",
                        normal_records
                    )

                    c3.metric(
                        "Attack",
                        attack_records
                    )

                    c4.metric(
                        "Attack %",
                        f"{attack_percentage:.2f}%"
                    )

                    # ------------------------------------------
                    # Attack Distribution
                    # ------------------------------------------

                    st.subheader(
                        "Attack Category Distribution"
                    )

                    st.bar_chart(
                        results["Attack Category"].value_counts()
                    )

                    # ------------------------------------------
                    # Download Button
                    # ------------------------------------------

                    csv = results.to_csv(index=False)

                    st.download_button(
                        label="⬇ Download Prediction Results",
                        data=csv,
                        file_name="prediction_results.csv",
                        mime="text/csv"
                    )

                except Exception as e:

                 import traceback

                 st.error("Prediction Failed")

                 st.code(traceback.format_exc())

    else:

        st.error("❌ Dataset is not ready for prediction.")

        st.write("Missing Required Columns:")

        st.write(missing_columns)