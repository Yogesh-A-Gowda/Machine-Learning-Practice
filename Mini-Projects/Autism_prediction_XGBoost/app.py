import streamlit as st
import pickle
import pandas as pd

st.title("Autism Spectrum Disorder Screening App")


# Load the trained model and encoders
with open("best_model.pkl", "rb") as f:
  model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
  encoders = pickle.load(f)


st.header("Enter the following information:")

# Create input fields for each feature
a_scores = {}
for i in range(1, 11):
    a_scores[f"A{i}_Score"] = st.selectbox(f"A{i}_Score", [0, 1])

age = st.number_input("Age", min_value=0, max_value=100, value=25)
gender = st.selectbox("Gender", ["f", "m"])
ethnicity = st.selectbox("Ethnicity", encoders['ethnicity'].classes_)
jaundice = st.selectbox("Jaundice", ["no", "yes"])
austim = st.selectbox("Austim", ["no", "yes"])
country_of_res = st.selectbox("Country of Residence", encoders['contry_of_res'].classes_)
used_app_before = st.selectbox("Used App Before", ["no", "yes"])
result = st.number_input("Result Score", value=8.0)
relation = st.selectbox("Relation", encoders['relation'].classes_)



if st.button("Predict"):
    # Create a DataFrame from the input data
    input_data = {
        **a_scores,
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "jaundice": jaundice,
        "austim": austim,
        "contry_of_res": country_of_res,
        "used_app_before": used_app_before,
        "result": result,
        "relation": relation
    }
    input_df = pd.DataFrame([input_data])

    # Preprocess the input data using the loaded encoders
    for column, encoder in encoders.items():
        input_df[column] = encoder.transform(input_df[column])

    # Make prediction
    prediction = model.predict(input_df)

    # Display the result
    if prediction[0] == 1:
        st.write("Based on the information provided, the prediction is that the individual has Autism Spectrum Disorder (ASD).")
    else:
        st.write("Based on the information provided, the prediction is that the individual does not have Autism Spectrum Disorder (ASD).")
