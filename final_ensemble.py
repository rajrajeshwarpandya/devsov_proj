import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

rf_models = joblib.load("rf_ensemble_list.joblib")
gbc_model = joblib.load("best_gbc_model.joblib")
stacking_model = joblib.load("stacking_model_no_keras.pkl")

rf_probas = np.mean([model.predict_proba(X_test) for model in rf_models], axis=0)
gbc_probas = gbc_model.predict_proba(X_test)
stacking_probas = stacking_model.predict_proba(X_test)


weights = {
    "rf": 0.15,
    "gbc": 0.25,
    "stack": 0.60
}

combined_probas = (
    weights["rf"] * rf_probas +
    weights["gbc"] * gbc_probas +
    weights["stack"] * stacking_probas
)


y_pred = np.argmax(combined_probas, axis=1)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4, zero_division=0)


print(f"ensemble accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)

labels = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
    "Overweight_Level_I",
    "Overweight_Level_II"
]

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig("final_ensemble_confusion_matrix_named.png")
plt.show()
