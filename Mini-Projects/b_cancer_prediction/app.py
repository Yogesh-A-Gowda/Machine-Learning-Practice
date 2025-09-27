from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # List of expected features in correct order
        feature_names = ['id',
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
            'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
            'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
            'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst',
            'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst',
            'fractal_dimension_worst'
        ]

        # Extract in order
        features = [float(data[fname]) for fname in feature_names]
        final_features = np.array(features).reshape(1, -1)

        # Scale using pre-fitted scaler
        final_features_scaled = scaler.transform(final_features)

        # Predict
        prediction = model.predict(final_features_scaled)[0]
        result = "Malignant" if prediction == 1 else "Benign"

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
if __name__ == "__main__":
    app.run(debug=True)