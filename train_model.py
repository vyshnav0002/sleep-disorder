import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# Load the dataset
file_path = 'data/data.csv'
try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file at {file_path} does not exist.")
    raise

# Preprocess the data
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['BMI Category'] = data['BMI Category'].map({'Normal': 0, 'Overweight': 1, 'Obese': 2})

data[['Systolic BP', 'Diastolic BP']] = data['Blood Pressure'].str.split('/', expand=True)
data['Systolic BP'] = data['Systolic BP'].astype(int)
data['Diastolic BP'] = data['Diastolic BP'].astype(int)

data = data.drop(columns=['Person ID', 'Occupation', 'Blood Pressure'])

le = LabelEncoder()
data['Sleep Disorder'] = le.fit_transform(data['Sleep Disorder'])

X = data.drop(columns=['Sleep Disorder'])
y = data['Sleep Disorder']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training completed.")

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, 'sleep_disorder_model.pkl')

try:
    joblib.dump(model, model_path)
    print(f"Model saved successfully at {model_path}.")
except Exception as e:
    print(f"Error saving the model: {e}")
    joblib.dump(le, '../models/label_encoder.pkl')


