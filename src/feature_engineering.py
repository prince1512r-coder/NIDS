import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv(r"C:\NIDS_\data\processed\cleaned_training_dataset.csv")

print("Dataset Loaded Successfully")

print("\nOriginal Dataset Shape")
print(train_df.shape)

train_df = train_df.drop("id", axis=1)

print("\n'id' column removed successfully.")

categorical_columns = ["proto", "service", "state", "attack_cat"]

encoder = LabelEncoder()

for column in categorical_columns:
    train_df[column] = encoder.fit_transform(train_df[column])

print("\nCategorical columns encoded successfully.")

print("\nFirst Five Rows of Engineered Dataset")
print(train_df.head())

print("\nEngineered Dataset Shape")
print(train_df.shape)

train_df.to_csv(
    r"C:\NIDS_\data\processed\engineered_training_dataset.csv",
    index=False
)

print("\nEngineered dataset saved successfully.")