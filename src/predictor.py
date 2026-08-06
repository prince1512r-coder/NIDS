import joblib
import pandas as pd

print("===== NEW predictor.py LOADED =====")

# -----------------------------
# Load Saved Models
# -----------------------------

binary_model = joblib.load(
    r"C:\GITHUB\NIDS\models\binary_model.pkl"
)

multiclass_model = joblib.load(
    r"C:\GITHUB\NIDS\models\multiclass_model.pkl"
)

encoders = joblib.load(
    r"C:\GITHUB\NIDS\models\encoders.pkl"
)

# -----------------------------
# Required Features
# -----------------------------

FEATURE_COLUMNS = [
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

# -----------------------------
# Preprocess Uploaded Dataset
# -----------------------------

def preprocess_data(df):

    df = df.copy()

    # Remove ID column
    if "id" in df.columns:
        df = df.drop("id", axis=1)

    # Remove target columns if present
    for column in ["label", "attack_cat"]:
        if column in df.columns:
            df = df.drop(column, axis=1)

    # Encode categorical columns
    categorical_columns = [
        "proto",
        "service",
        "state"
    ]

    for column in categorical_columns:

        if column in df.columns:

            encoder = encoders[column]

            class_map = {
                value: index
                for index, value in enumerate(encoder.classes_)
            }

            df[column] = (
                df[column]
                .astype(str)
                .map(class_map)
                .fillna(-1)
                .astype(int)
            )

    # Check required columns
    missing_columns = [
        col
        for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if len(missing_columns) > 0:

        raise ValueError(
            "Missing Required Columns:\n"
            + ", ".join(missing_columns)
        )

    # Arrange columns
    df = df[FEATURE_COLUMNS]

    return df


# -----------------------------
# Binary Prediction
# -----------------------------

def predict_binary(df):

    return binary_model.predict(df)


# -----------------------------
# Multi-Class Prediction
# -----------------------------

def predict_multiclass(df):

    return multiclass_model.predict(df)


# -----------------------------
# Complete Prediction Pipeline
# -----------------------------

def predict_dataset(df):

    processed_df = preprocess_data(df)

    binary_predictions = predict_binary(processed_df)

    multiclass_predictions = predict_multiclass(processed_df)

    attack_encoder = encoders["attack_cat"]

    attack_names = attack_encoder.inverse_transform(
        multiclass_predictions
    )

    results = df.copy()

    # -----------------------------
    # Binary Prediction (1st Column)
    # -----------------------------

    results.insert(
        0,
        "Binary Prediction",
        [
            "Attack" if prediction == 1 else "Normal"
            for prediction in binary_predictions
        ]
    )

    # -----------------------------
    # Attack Category (2nd Column)
    # -----------------------------

    results.insert(
        1,
        "Attack Category",
        attack_names
    )

    # Normal traffic should always show Normal
    results.loc[
        results["Binary Prediction"] == "Normal",
        "Attack Category"
    ] = "Normal"

    return results