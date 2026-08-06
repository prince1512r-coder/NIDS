import pandas as pd

df = pd.read_csv(
    r"C:\GITHUB\NIDS\data\processed\engineered_training_dataset.csv"
)

print("=" * 60)
print("NETWORK INTRUSION DETECTION SYSTEM (NIDS)")
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print("\nDataset Information")
print("--------------------")
print("Dataset Name : UNSW-NB15")
print("Total Records :", len(df))
print("Total Columns :", len(df.columns))
print("Input Features :", len(df.columns) - 2)

print("\nAttack Categories")
print("--------------------")
print(df["attack_cat"].value_counts().sort_index())

print("\nBinary Classification")
print("--------------------")
print("Algorithm : Decision Tree Classifier")
print("Accuracy  : 96.57 %")

print("\nMulti-Class Classification")
print("--------------------")
print("Algorithm : Decision Tree Classifier")
print("Accuracy  : 85.43 %")

print("\nTop 10 Important Features")
print("--------------------")

feature_importance = [
    ("sttl", 0.337965),
    ("ct_dst_src_ltm", 0.162295),
    ("tcprtt", 0.145339),
    ("sbytes", 0.132874),
    ("ct_srv_dst", 0.033517),
    ("dbytes", 0.033311),
    ("rate", 0.017256),
    ("smean", 0.014042),
    ("ct_srv_src", 0.013268),
    ("sinpkt", 0.012635)
]

for i, (feature, score) in enumerate(feature_importance, start=1):
    print(f"{i}. {feature:<20} {score:.6f}")

print("\nProject Modules Completed")
print("-------------------------")
print("Week 1 : Dataset Study and Preprocessing")
print("Week 2 : Exploratory Data Analysis and Feature Engineering")
print("Week 3 : Binary Classification")
print("Week 4 : Multi-Class Classification")
print("Week 5 : Feature Selection and Explainability")
print("Week 6 : Final Integration and Documentation")

print("\nProject Status")
print("--------------------")
print("Project Completed Successfully")

print("=" * 60)