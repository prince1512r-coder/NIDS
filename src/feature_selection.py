import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

os.makedirs(
    r"C:\GITHUB\NIDS\outputs\results",
    exist_ok=True
)

df = pd.read_csv(
    r"C:\GITHUB\NIDS\data\processed\engineered_training_dataset.csv"
)

print("Dataset Loaded Successfully")

X = df.drop(["label", "attack_cat"], axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features\n")

print(feature_importance.head(10))

plt.figure(figsize=(10,6))

plt.bar(
    feature_importance["Feature"].head(10),
    feature_importance["Importance"].head(10)
)

plt.xticks(rotation=45)

plt.title("Top 10 Feature Importance")

plt.tight_layout()

plt.savefig(
    r"C:\GITHUB\NIDS\outputs\results\feature_importance.png"
)

plt.show()

print("\nFeature Importance Graph Saved Successfully")

print("\nWeek 5 Completed Successfully")