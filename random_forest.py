import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
#this is rainforest esemble which uses soft voting
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

param_grid = [
    {'n_estimators': 150, 'max_depth': 12, 'random_state': 0},
    {'n_estimators': 100, 'max_features': 'sqrt', 'random_state': 1},
    {'n_estimators': 120, 'min_samples_leaf': 4, 'random_state': 2},
    {'n_estimators': 180, 'max_depth': 10, 'random_state': 3},
    {'n_estimators': 200, 'max_features': 'log2', 'random_state': 4}
]

rf_models = []
for params in param_grid:
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    rf_models.append(model)

joblib.dump(rf_models, "rf_ensemble_list.joblib")

probas = np.array([model.predict_proba(X_test) for model in rf_models])
avg_proba = np.mean(probas, axis=0)
y_pred = np.argmax(avg_proba, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("confusion matrix")
plt.tight_layout()
plt.savefig("rf_ensemble_confusion_matrix.png")
plt.show()
