import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

os.makedirs(
    r"C:\GITHUB\NIDS\models",
    exist_ok=True
)

os.makedirs(
    r"C:\GITHUB\NIDS\outputs\results",
    exist_ok=True
)

df = pd.read_csv(
    r"C:\GITHUB\NIDS\data\processed\engineered_training_dataset.csv"
)

print("Dataset Loaded Successfully")

print("\nDataset Shape")
print(df.shape)

X = df.drop(["label", "attack_cat"], axis=1)
y = df["label"]

print("\nFeature Matrix Shape")
print(X.shape)

print("\nTarget Variable Shape")
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape")
print(X_train.shape)

print("\nTesting Data Shape")
print(X_test.shape)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

joblib.dump(
    model,
    r"C:\GITHUB\NIDS\models\binary_model.pkl"
)

print("\nBinary model saved successfully.")

print("\nModel Trained Successfully")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy")
print(round(accuracy * 100, 2), "%")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.title("Binary Classification Confusion Matrix")

plt.savefig(
    r"C:\GITHUB\NIDS\outputs\results\confusion_matrix.png"
)

plt.show()

print("\nWeek 3 Completed Successfully")