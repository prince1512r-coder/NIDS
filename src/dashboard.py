import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from predictor import predict_dataset
from datetime import datetime


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    /* Status cards */
    .status-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        text-align: center;
        min-height: 100px;
    }

    .status-title {
        font-size: 15px;
        opacity: 0.75;
    }

    .status-value {
        font-size: 20px;
        font-weight: 600;
    }

    /* Section headings */
    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🛡️ Network Intrusion Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning-Based Network Security Analysis using UNSW-NB15'
    '</div>',
    unsafe_allow_html=True
)


with st.expander("ℹ️ About This Project"):

    st.markdown(
        """
        This application analyzes network traffic using trained machine
        learning models to detect malicious activities and classify
        different cyber attack categories.

        **Project Information**

        - **Dataset:** UNSW-NB15
        - **Binary Classification:** Normal vs Attack
        - **Multi-Class Classification:** Attack Category Detection
        - **Machine Learning Model:** Decision Tree Classifier
        - **Framework:** Streamlit + Scikit-Learn
        """
    )


st.divider()


# ==================================================
# LOAD MODELS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "binary_model.pkl"

MULTI_MODEL_PATH = MODEL_DIR / "multiclass_model.pkl"

ENCODER_PATH = MODEL_DIR / "encoders.pkl"


models_loaded = False

try:

    binary_model = joblib.load(
        MODEL_PATH
    )

    multiclass_model = joblib.load(
        MULTI_MODEL_PATH
    )

    encoders = joblib.load(
        ENCODER_PATH
    )

    models_loaded = True

except Exception as e:

    st.error(
        "❌ Unable to load trained models."
    )

    st.code(
        str(e)
    )


# ==================================================
# SYSTEM STATUS
# ==================================================

st.markdown(
    '<div class="section-title">🖥️ System Status</div>',
    unsafe_allow_html=True
)


status1, status2, status3 = st.columns(3)


with status1:

    if models_loaded:

        st.success(
            "🟢 Binary Model Loaded"
        )

    else:

        st.error(
            "🔴 Binary Model Error"
        )


with status2:

    if models_loaded:

        st.success(
            "🟢 Multi-Class Model Loaded"
        )

    else:

        st.error(
            "🔴 Multi-Class Model Error"
        )


with status3:

    if models_loaded:

        st.success(
            "🟢 Label Encoders Loaded"
        )

    else:

        st.error(
            "🔴 Encoder Error"
        )


st.divider()


# ==================================================
# UPLOAD DATASET
# ==================================================

st.markdown(
    '<div class="section-title">'
    '📂 Upload Network Traffic Dataset'
    '</div>',
    unsafe_allow_html=True
)


st.info(
    """
    Upload a **CSV file** containing network traffic data.

    **Recommended dataset:** UNSW-NB15

    The system will automatically:

    ✔ Validate the dataset

    ✔ Preprocess network traffic

    ✔ Perform Binary Classification

    ✔ Perform Multi-Class Classification

    ✔ Generate security statistics

    ✔ Generate attack visualizations
    """
)


uploaded_file = st.file_uploader(
    "Choose CSV File",
    type=["csv"]
)


st.divider()


# ==================================================
# REQUIRED COLUMNS
# ==================================================

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


# ==================================================
# READ UPLOADED DATASET
# ==================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(
            uploaded_file
        )

        st.success(
            "✅ Dataset Uploaded Successfully"
        )


        # ==================================================
        # DATASET PREVIEW
        # ==================================================

        st.markdown(
            '<div class="section-title">'
            '📋 Dataset Preview'
            '</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            df.head(),
            use_container_width=True
        )


        # ==================================================
        # DATASET INFORMATION
        # ==================================================

        st.markdown(
            '<div class="section-title">'
            '📊 Dataset Information'
            '</div>',
            unsafe_allow_html=True
        )

        info1, info2, info3 = st.columns(3)


        with info1:

            st.metric(
                "Total Rows",
                df.shape[0]
            )


        with info2:

            st.metric(
                "Total Columns",
                df.shape[1]
            )


        with info3:

            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )


        # ==================================================
        # DATASET VALIDATION
        # ==================================================

        st.markdown(
            '<div class="section-title">'
            '🔍 Dataset Validation'
            '</div>',
            unsafe_allow_html=True
        )


        missing_columns = [

            column

            for column in required_columns

            if column not in df.columns

        ]


        if len(missing_columns) == 0:

            st.success(
                "✅ Dataset is ready for intrusion detection."
            )


            # ==================================================
            # PREDICTION BUTTON
            # ==================================================

            predict_button = st.button(
                "🚀 Start Intrusion Detection",
                use_container_width=True
            )


            if predict_button:

                with st.spinner(
                    "🔄 Analyzing Network Traffic..."
                ):

                    try:

                        results = predict_dataset(
                            df
                        )

                        st.success(
                            "✅ Prediction Completed Successfully!"
                        )

                                                # ==================================================
                        # PREDICTION RESULTS
                        # ==================================================

                        st.markdown(
                            '<div class="section-title">'
                            '🎯 Prediction Results'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        st.dataframe(
                            results,
                            use_container_width=True
                        )


                        # ==================================================
                        # CALCULATE SECURITY STATISTICS
                        # ==================================================

                        total_records = len(
                            results
                        )

                        normal_records = (
                            results[
                                "Binary Prediction"
                            ] == "Normal"
                        ).sum()

                        attack_records = (
                            results[
                                "Binary Prediction"
                            ] == "Attack"
                        ).sum()

                        if total_records > 0:

                            attack_percentage = (
                                attack_records
                                / total_records
                            ) * 100

                        else:

                            attack_percentage = 0


                        # ==================================================
                        # MOST COMMON ATTACK
                        # ==================================================

                        attack_distribution = (
                            results[
                                "Attack Category"
                            ]
                            .value_counts()
                        )

                        if len(attack_distribution) > 0:

                            most_common_attack = (
                                attack_distribution
                                .idxmax()
                            )

                        else:

                            most_common_attack = "None"


                        # ==================================================
                        # NETWORK STATUS
                        # ==================================================

                        st.markdown(
                            '<div class="section-title">'
                            '🛡️ Network Security Status'
                            '</div>',
                            unsafe_allow_html=True
                        )


                        if attack_percentage < 20:

                            st.success(
                                f"""
                                ### 🟢 NETWORK STATUS: SAFE

                                **Threat Level:** LOW

                                **Attack Percentage:** \
                                {attack_percentage:.2f}%

                                **Most Common Category:** \
                                {most_common_attack}
                                """
                            )


                        elif attack_percentage < 50:

                            st.warning(
                                f"""
                                ### 🟡 NETWORK STATUS: MEDIUM RISK

                                **Threat Level:** MEDIUM

                                **Attack Percentage:** \
                                {attack_percentage:.2f}%

                                **Most Common Category:** \
                                {most_common_attack}
                                """
                            )


                        else:

                            st.error(
                                f"""
                                ### 🔴 NETWORK STATUS: HIGH RISK

                                **Threat Level:** HIGH

                                **Attack Percentage:** \
                                {attack_percentage:.2f}%

                                **Most Common Category:** \
                                {most_common_attack}
                                """
                            )


                        st.divider()


                        # ==================================================
                        # TRAFFIC SUMMARY
                        # ==================================================

                        st.markdown(
                            '<div class="section-title">'
                            '📊 Traffic Summary'
                            '</div>',
                            unsafe_allow_html=True
                        )


                        summary1, summary2, summary3, summary4 = (
                            st.columns(4)
                        )


                        with summary1:

                            st.metric(
                                "Total Records",
                                total_records
                            )


                        with summary2:

                            st.metric(
                                "🟢 Normal Traffic",
                                normal_records
                            )


                        with summary3:

                            st.metric(
                                "🔴 Malicious Traffic",
                                attack_records
                            )


                        with summary4:

                            st.metric(
                                "⚠️ Attack Rate",
                                f"{attack_percentage:.2f}%"
                            )


                        st.divider()


                        # ==================================================
                        # NORMAL VS ATTACK DATA
                        # ==================================================

                        st.markdown(
                            '<div class="section-title">'
                            '📊 Traffic Security Distribution'
                            '</div>',
                            unsafe_allow_html=True
                        )


                        traffic_distribution = pd.DataFrame(
                            {
                                "Traffic Type": [
                                    "Normal",
                                    "Attack"
                                ],
                                "Records": [
                                    normal_records,
                                    attack_records
                                ]
                            }
                        )


                        st.bar_chart(
                            traffic_distribution.set_index(
                                "Traffic Type"
                            )
                        )


                        st.divider()
                        
                         # ==================================================
                        # SCAN INFORMATION
                        # ==================================================

                        st.markdown(
                            '<div class="section-title">'
                            '🕐 Scan Information'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        scan_time = datetime.now().strftime(
                            "%d %B %Y, %I:%M:%S %p"
                        )

                        scan1, scan2 = st.columns(2)

                        with scan1:

                            st.metric(
                                "📅 Scan Date",
                                datetime.now().strftime(
                                    "%d %B %Y"
                                )
                            )

                        with scan2:

                            st.metric(
                                "🕐 Scan Time",
                                datetime.now().strftime(
                                    "%I:%M:%S %p"
                                )
                            )

                        st.caption(
                            f"Analysis completed at {scan_time}"
                        )

                        st.divider()


                        # ==================================================
                        # TOP ATTACK CATEGORIES
                        # ==================================================

                        st.markdown(
                            '<div class="section-title">'
                            '🏆 Top Attack Categories'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        # Remove Normal because this section
                        # is specifically about attacks.

                        attack_only = results[
                            results["Binary Prediction"] == "Attack"
                        ]

                        if len(attack_only) > 0:

                            top_attacks = (
                                attack_only[
                                    "Attack Category"
                                ]
                                .value_counts()
                                .head(5)
                            )

                            top_attack_df = (
                                top_attacks
                                .reset_index()
                            )

                            top_attack_df.columns = [
                                "Attack Category",
                                "Number of Records"
                            ]

                            st.dataframe(
                                top_attack_df,
                                use_container_width=True,
                                hide_index=True
                            )

                        else:

                            st.success(
                                "🟢 No attack categories detected."
                            )


                        st.divider()


                        # ==================================================
                        # ATTACK CATEGORY DISTRIBUTION
                        # ==================================================

                        st.markdown(
                            '<div class="section-title">'
                            '📈 Attack Category Distribution'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        attack_distribution = (
                            results[
                                "Attack Category"
                            ]
                            .value_counts()
                        )

                        st.bar_chart(
                            attack_distribution
                        )


                        st.divider()


                        # ==================================================
                        # DOWNLOAD RESULTS
                        # ==================================================

                        st.markdown(
                            '<div class="section-title">'
                            '📥 Export Results'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        csv_data = results.to_csv(
                            index=False
                        )

                        st.download_button(
                            label=(
                                "⬇️ Download Prediction Results"
                            ),
                            data=csv_data,
                            file_name=(
                                "nids_prediction_results.csv"
                            ),
                            mime="text/csv",
                            use_container_width=True
                        )


                        st.divider()


                        # ==================================================
                        # FINAL ANALYSIS MESSAGE
                        # ==================================================

                        if attack_records == 0:

                            st.success(
                                """
                                🟢 **No malicious network traffic was
                                detected in the analyzed dataset.**
                                """
                            )

                        elif attack_percentage < 20:

                            st.info(
                                """
                                ℹ️ **A small proportion of malicious
                                traffic was detected. Continue monitoring
                                the network for suspicious activity.**
                                """
                            )

                        elif attack_percentage < 50:

                            st.warning(
                                """
                                ⚠️ **A significant amount of malicious
                                traffic was detected. Further investigation
                                is recommended.**
                                """
                            )

                        else:

                            st.error(
                                """
                                🚨 **A high proportion of malicious traffic
                                was detected. Immediate investigation of
                                the network traffic is recommended.**
                                """
                            )


                            # ==================================================
                            # FOOTER
                            # ==================================================
                            
                            st.markdown(
                                """
                                <br>
                               <hr>
                            
                                <div style="text-align:center; opacity:0.65;">
                            
                                🛡️ <b>Machine Learning-Based Network Intrusion Detection System</b>

                                <br>
                            
                                UNSW-NB15 Dataset • Python • Streamlit • Scikit-Learn
                            
                                <br><br>
                            
                                Network Security Analysis Dashboard
                            
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    except Exception as e:

                        import traceback

                        st.error(
                            "❌ Prediction Failed"
                        )

                        st.code(
                            traceback.format_exc()
                        )


        else:

            st.error(
                "❌ Dataset is not ready for prediction."
            )

            st.write(
                "Missing Required Columns:"
            )

            st.write(
                missing_columns
            )

    except Exception as e:

        st.error(
            "❌ Unable to read the uploaded CSV file."
        )

        st.code(
            str(e)
        )