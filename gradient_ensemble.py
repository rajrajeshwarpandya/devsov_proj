import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib 


X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

param_grid = {
    'n_estimators': [100, 200,300],
    'learning_rate': [0.1,0.05,0.01],
    'max_depth': [3,4,5]
}


gbc = GradientBoostingClassifier(random_state=42)
grid = GridSearchCV(gbc, param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("best parameters:", grid.best_params_)
print("best CV accuracy:", grid.best_score_)


joblib.dump(grid.best_estimator_, "best_gbc_model.joblib")
print("model saved as 'best_gbc_model.joblib")

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\ntest accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("confusion matrix")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()
import joblib
model = joblib.load("best_gbc_model.joblib")
y_new_pred = model.predict(X_test)
