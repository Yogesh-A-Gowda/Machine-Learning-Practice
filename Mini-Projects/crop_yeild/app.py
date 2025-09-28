from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle


app = Flask(__name__)
model = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
        data = request.json
        Area = data['area']
        Item = data['item']
        Year = int(data['year'])
        average_rain_fall_mm_per_year = float(data['rainfall'])
        pesticides_tonnes = float(data['pesticides'])
        avg_temp = float(data['temp'])
        input_data_as_numpy_array = np.array([[Area,Item,Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp]])
        std_data = preprocessor.transform(input_data_as_numpy_array)
        prediction = model.predict(std_data).reshape(1,-1)
        return {'prediction': prediction.item()}


if __name__ == '__main__' :
    app.run(debug=True)
