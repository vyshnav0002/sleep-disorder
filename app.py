from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('models/sleep_disorder_model.pkl')
le = joblib.load('models/label_encoder.pkl')
feature_columns = ['Age', 'Gender', 'Sleep Duration', 'Quality of Sleep', 
                   'Physical Activity Level', 'Stress Level', 'BMI Category', 
                   'Systolic BP', 'Diastolic BP', 'Heart Rate', 'Daily Steps']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extracting input features from the form
            age = int(request.form['age'])
            gender = request.form['gender']
            sleep_duration = float(request.form['sleep_duration'])
            quality_of_sleep = int(request.form['quality_of_sleep'])
            physical_activity_level = int(request.form['physical_activity_level'])
            stress_level = int(request.form['stress_level'])
            bmi_category = request.form['bmi_category']
            blood_pressure = request.form['blood_pressure']
            heart_rate = int(request.form['heart_rate'])
            daily_steps = int(request.form['daily_steps'])

            # Convert and preprocess input features
            systolic_bp, diastolic_bp = map(int, blood_pressure.split('/'))
            gender = 1 if gender == 'Male' else 0
            bmi_category = {'Normal': 0, 'Overweight': 1, 'Obese': 2}[bmi_category]

            # Create a DataFrame for the input data, ensuring the correct feature order
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Sleep Duration': [sleep_duration],
                'Quality of Sleep': [quality_of_sleep],
                'Physical Activity Level': [physical_activity_level],
                'Stress Level': [stress_level],
                'BMI Category': [bmi_category],
                'Systolic BP': [systolic_bp],
                'Diastolic BP': [diastolic_bp],
                'Heart Rate': [heart_rate],
                'Daily Steps': [daily_steps],
            }, columns=feature_columns)

            # Make prediction using the loaded model
            prediction = model.predict(input_data)[0]
            prediction_label = le.inverse_transform([prediction])[0]

            return render_template('index.html', prediction_text=f'Predicted Sleep Disorder: {prediction_label}')
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__== 'main':
    app.run(debug=True)