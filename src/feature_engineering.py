import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

os.makedirs(r"C:\GITHUB\NIDS\data\processed", exist_ok=True)
os.makedirs(r"C:\GITHUB\NIDS\models", exist_ok=True)

train_df = pd.read_csv(
    r"C:\GITHUB\NIDS\data\processed\cleaned_training_dataset.csv"
)

print("Dataset Loaded Successfully")

print("\nOriginal Dataset Shape")
print(train_df.shape)

train_df = train_df.drop("id", axis=1)

print("\n'id' column removed successfully.")

categorical_columns = ["proto", "service", "state", "attack_cat"]

encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    train_df[column] = encoder.fit_transform(train_df[column])

    encoders[column] = encoder

print("\nCategorical columns encoded successfully.")

print("\nFirst Five Rows of Engineered Dataset")
print(train_df.head())

print("\nEngineered Dataset Shape")
print(train_df.shape)

train_df.to_csv(
    r"C:\GITHUB\NIDS\data\processed\engineered_training_dataset.csv",
    index=False
)

joblib.dump(
    encoders,
    r"C:\GITHUB\NIDS\models\encoders.pkl"
)

print("\nEncoders saved successfully.")

print("\nEngineered dataset saved successfully.")