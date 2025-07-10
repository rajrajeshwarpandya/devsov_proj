
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv("dataset.csv")

df.drop(columns=['id'], inplace=True)

categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('NObeyesdad')

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


target_encoder = LabelEncoder()
df['NObeyesdad'] = target_encoder.fit_transform(df['NObeyesdad'])

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_cols.remove('NObeyesdad')
scaler = StandardScaler()
#df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


X = df.drop(columns=['NObeyesdad'])
y = df['NObeyesdad']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y)


X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
df.to_csv("cleaned_dataset.csv", index=False)

print("data cleaned")
