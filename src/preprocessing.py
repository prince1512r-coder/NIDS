import pandas as pd

print("Program Started")

train_df = pd.read_csv(r"C:\NIDS_\data\raw\UNSW_NB15_training-set.csv")
test_df = pd.read_csv(r"C:\NIDS_\data\raw\UNSW_NB15_testing-set.csv")

print("\nTraining Dataset Loaded Successfully")
print(train_df.head())

print("\nTesting Dataset Loaded Successfully")
print(test_df.head())

print("\nTraining Dataset Information")
print(train_df.info())

print("\nTesting Dataset Information")
print(test_df.info())

print("\nTraining Dataset Shape:", train_df.shape)
print("Testing Dataset Shape:", test_df.shape)

print("\nTraining Dataset Description")
print(train_df.describe())

print("\nTesting Dataset Description")
print(test_df.describe())

print("\nMissing Values in Training Dataset")
print(train_df.isnull().sum())

print("\nMissing Values in Testing Dataset")
print(test_df.isnull().sum())

print("\nDuplicate Rows in Training Dataset:", train_df.duplicated().sum())
print("Duplicate Rows in Testing Dataset:", test_df.duplicated().sum())

train_df.to_csv(r"C:\NIDS_\data\processed\cleaned_training_dataset.csv", index=False)
test_df.to_csv(r"C:\NIDS_\data\processed\cleaned_testing_dataset.csv", index=False)

print("\nProcessed datasets saved successfully.")