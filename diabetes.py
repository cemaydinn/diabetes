import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('pima-indians-diabetes.csv')  # Replace with the actual file path

# Replace zero values with NaN for specific columns
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_with_zeros] = data[columns_with_zeros].replace(0, np.nan)

# Fill missing values with the median of each column
data.fillna(data.median(), inplace=True)

# Example: Create a new feature for age group
data['AgeGroup'] = pd.cut(data['Age'], bins=[20, 30, 40, 50, 60, 70, 80], labels=False)

# Example: Create a new feature for BMI category
data['BMICategory'] = pd.cut(data['BMI'], bins=[0, 18.5, 25, 30, 40, 50], labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Extremely Obese'])

# Encoding the 'BMICategory' using Label Encoding
label_encoder = LabelEncoder()
data['BMICategory'] = label_encoder.fit_transform(data['BMICategory'].astype(str))

# Standardize numerical features
scaler = StandardScaler()
numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Split the data into training and testing sets
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

