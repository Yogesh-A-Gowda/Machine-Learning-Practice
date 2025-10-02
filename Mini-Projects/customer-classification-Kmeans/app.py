import numpy as np
import pickle
import streamlit as st

# Load the trained KMeans model
with open('kmeans_model.pkl', 'rb') as model_file:
    kmeans_model = pickle.load(model_file)

def customer(age, avg_spending, visits_per_week, promotion_interest):
    new_customer = np.array([[age, avg_spending, visits_per_week, promotion_interest]])
    predicted_cluster = kmeans_model.predict(new_customer)

    if predicted_cluster == 0:
        return "Cluster 0: Young, low spenders, infrequent visitors, low promotion interest."
    elif predicted_cluster == 1:
        return "Cluster 1: Middle-aged, high spenders, frequent visitors, high promotion interest."
    elif predicted_cluster == 2:
        return "Cluster 2: Older, moderate spenders, moderate visitors, moderate promotion interest."
    elif predicted_cluster == 3:
        return "Cluster 3: Young adults, high spenders, frequent visitors, high promotion interest."
    else:
        return "Unknown cluster."
    pass


st.title("Customer Segmentation using KMeans Clustering")
st.write("Enter customer details to classify them into a cluster.")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
avg_spending = st.number_input("Average Spending ($)", min_value=0, value=
1000)
visits_per_week = st.number_input("Visits per Week", min_value=0, max_value=50, value=5)
promotion_interest = st.number_input("Promotion Interest (1-10)", min_value=1, max_value=10, value=5)
if st.button("Classify Customer"):
    result = customer(age, avg_spending, visits_per_week, promotion_interest)
    st.write(result)