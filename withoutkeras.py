import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier


def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()


xgb_model = XGBClassifier(
    tree_method="hist", device="cuda", n_estimators=300,
    max_depth=5, learning_rate=0.08, subsample=0.85,
    colsample_bytree=0.8, random_state=42, eval_metric="mlogloss"
)

lgb_model = LGBMClassifier(
    device='gpu', n_estimators=300, num_leaves=64,
    max_depth=6, learning_rate=0.08, subsample=0.85,
    colsample_bytree=0.8, verbosity=-1, random_state=42
)

cat_model = CatBoostClassifier(
    task_type='GPU', devices='0', iterations=300,
    learning_rate=0.08, depth=5, l2_leaf_reg=3,
    verbose=0, random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=150, max_depth=10,
    random_state=42, n_jobs=-1
)


final_estimator = MLPClassifier(
    hidden_layer_sizes=(100,), activation='relu',
    solver='adam', learning_rate='adaptive',
    max_iter=500, early_stopping=True, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

stack_model = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model),
        ('rf', rf_model)
    ],
    final_estimator=final_estimator,
    passthrough=True,
    n_jobs=1,
    cv=cv
)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('stacking', stack_model)
])

print("training(no keras)...")
pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
#n
print("accuracy:", accuracy_score(y_test, y_pred))
print("\n classification report:\n", classification_report(y_test, y_pred, digits=4, zero_division=0))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.tight_layout()
plt.savefig("confusion_matrix_no_keras.png")
plt.show()


joblib.dump(pipeline, "stacking_model_no_keras.pkl")
print("Saved 'stacking_model_no_keras.pkl'")
