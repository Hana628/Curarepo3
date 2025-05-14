import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# Drop duplicates and handle missing values
df = df.drop_duplicates()
df = df.dropna()

# Handle 'Blood_Pressure' column if it exists (Convert '139/91' to separate Systolic/Diastolic columns)
if 'Blood_Pressure' in df.columns:
    df[['Systolic', 'Diastolic']] = df['Blood_Pressure'].str.split('/', expand=True).astype(float)
    df.drop(columns=['Blood_Pressure'], inplace=True)

# Encode categorical variables
le_dict = {}
categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Features and target
X = df.drop(['Sleep Disorder'], axis=1)
y = df['Sleep Disorder']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")

# Save the model
with open('lifestylemodel.pkl', 'wb') as f:
    pickle.dump(model, f)

# Optional: Save label encoders if needed
# with open('label_encoders.pkl', 'wb') as f:
#     pickle.dump(le_dict, f)
