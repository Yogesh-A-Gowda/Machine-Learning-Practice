import pickle
import numpy as np
import pandas as pd

# Load model and preprocessor
model = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

print("Model type:", type(model))
print("Preprocessor type:", type(preprocessor))

# Test with REAL data structure (pandas DataFrame)
test_df = pd.DataFrame([{
    'Area': 'India',
    'Item': 'Wheat',
    'Year': 2000,
    'average_rain_fall_mm_per_year': 1000.0,
    'pesticides_tonnes': 50.0,
    'avg_temp': 25.0
}])

print("\nInput DataFrame:")
print(test_df)

try:
    # Transform using preprocessor
    X_transformed = preprocessor.transform(test_df)
    print("\nTransformed shape:", X_transformed.shape)
    
    # Predict
    prediction = model.predict(X_transformed)
    print("✅ Prediction:", prediction[0])
    
except Exception as e:
    print("❌ Error:", e)