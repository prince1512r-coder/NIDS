import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv(r"C:\NIDS_\data\processed\cleaned_training_dataset.csv")

print("Dataset Loaded Successfully")

print("\nDataset Shape")
print(train_df.shape)

print("\nColumn Names")
print(train_df.columns)

print("\nAttack Category Distribution")
print(train_df["attack_cat"].value_counts())

attack_counts = train_df["attack_cat"].value_counts()

plt.figure(figsize=(10,6))
attack_counts.plot(kind="bar")
plt.title("Attack Category Distribution")
plt.xlabel("Attack Category")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r"C:\NIDS_\outputs\plots\attack_category_distribution.png")
plt.show()

print("\nLabel Distribution")
print(train_df["label"].value_counts())

label_counts = train_df["label"].value_counts()

plt.figure(figsize=(5,5))
label_counts.plot(kind="bar")
plt.title("Normal vs Attack")
plt.xlabel("Label")
plt.ylabel("Number of Records")
plt.xticks([0,1],["Normal","Attack"],rotation=0)
plt.tight_layout()
plt.savefig(r"C:\NIDS_\outputs\plots\normal_vs_attack.png")
plt.show()

features = ["dur","sbytes","dbytes"]

for feature in features:

    plt.figure(figsize=(8,5))
    train_df[feature].hist(bins=30)

    plt.title(feature + " Distribution")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        r"C:\NIDS_\outputs\plots\\" +
        feature +
        "_distribution.png"
    )

    plt.show()

selected_features = train_df[
    [
        "dur",
        "sbytes",
        "dbytes",
        "spkts",
        "dpkts",
        "rate"
    ]
]

correlation = selected_features.corr()

print("\nCorrelation Matrix")
print(correlation)

plt.figure(figsize=(8,6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    r"C:\NIDS_\outputs\plots\correlation_heatmap.png"
)

plt.show()

print("\nAll graphs generated successfully.")