import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model and column names (ensure these files are saved during training)
model = joblib.load(open("model_project.pkl", "rb"))
column_names = joblib.load(open("model_columns.pkl", "rb"))

# Load the dataset (for displaying it and generating options in Streamlit)
df = pd.read_csv("resaleHDBprice.csv")  # Replace with your dataset file

# Title of the Streamlit app
st.title("HDB Resale Price Prediction")

# Sidebar for user input
st.sidebar.header("Input Features")

# Input fields for user interaction
flat_type = st.sidebar.selectbox(
    "Flat Type",
    options=df["flat_type"].unique()
)

flat_model = st.sidebar.selectbox(
    "Flat Model",
    options=df["flat_model"].unique()
)

floor_area_sqm = st.sidebar.slider(
    "Floor Area (sqm)", 
    min_value=20, 
    max_value=500, 
    value=80
)

house_age = st.sidebar.slider(
    "House Age (Years)",
    min_value=0,
    max_value=99,
    value=30
)

transport_type = st.sidebar.selectbox(
    "Transport Type",
    options=df["transport_type"].unique()
)

price_per_sqft = st.sidebar.slider(
    "Price Per Sqft (SGD)", 
    min_value= 0, 
    max_value=2000, 
    value=1000
)

# Create a dictionary of user inputs
user_data = {
    "flat_type": flat_type,
    "flat_model": flat_model,
    "floor_area_sqm": floor_area_sqm,
    "house_age": house_age,
    "transport_type": transport_type,
    "price_per_sqft": price_per_sqft
}

# Convert the user input to DataFrame
input_df = pd.DataFrame([user_data])

# Show the user input
st.subheader("User Input")
st.write(input_df)

# Convert categorical features to one-hot encoding
input_df_encoded = pd.get_dummies(input_df, columns=["transport_type", "flat_type", "flat_model"])

# Reindex the input DataFrame to align with the model's expected columns
input_df_encoded = input_df_encoded.reindex(columns=column_names, fill_value=0)

# Make predictions using the trained model
prediction = model.predict(input_df_encoded)

# Display the predicted resale price
st.subheader("Predicted Resale Price")
st.write(f"SGD {prediction[0]:,.2f}")

