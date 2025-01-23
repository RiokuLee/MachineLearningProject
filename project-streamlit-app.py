import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model and column names (ensure these files are saved during training)
model = joblib.load(open("model_project.pkl", "rb"))
column_names = joblib.load(open("model_columns.pkl", "rb"))

# Load the preprocessed data
df = joblib.load('preprocessed_data.pkl')

# Title of the Streamlit app
st.title("HDB Resale Price Prediction")

# Define region to transport mapping
region_to_transport = {
    "NORTH-EAST REGION": {
        "MRT": ["Sengkang", "Punggol", "Hougang", "Kovan", "Serangoon"],
        "LRT": ["Sengkang", "Compassvale", "Rumbia", "Bakau", "Kangkar", "Ranggung", "Cheng Lim", "Farmway", "Kupang", "Thanggam", "Fernvale", "Layar", "Tongkang", "Renjong","Punggol", "Cove", "Meridian", "Coral Edge", "Riviera", "Kadaloor", "Oasis", "Damai", "Sam Kee", "Teck Lee", "Punggol Point", "Samudera", "Nibong", "Sumang", "Soo Teck"]
    },
    "NORTH REGION": {
        "MRT": ["Yishun", "Khatib", "Sembawang", "Admiralty", "Woodlands"],
        "LRT": []
    },
    "EAST REGION": {
        "MRT": ["Tampines", "Bedok", "Pasir Ris", "Expo", "Simei", "Tanah Merah"],
        "LRT": []
    },
    "WEST REGION": {
        "MRT": ["Jurong East", "Bukit Batok", "Clementi", "Choa Chu Kang", "Bukit Panjang", "Boon Lay", "Lakeside", "Joo Koon", "Pioneer"],
        "LRT": ["Choa Chu Kang", "South View", "Keat Hong", "Teck Whye", "Phoenix", "Bukit Panjang", "Petir", "Pending", "Bangkit", "Fajar", "Segar", "Jelapang", "Senja", "Ten Mile Junction"]
    },
    "CENTRAL REGION": {
        "MRT": ["Dhoby Ghaut", "City Hall", "Orchard", "Raffles Place", "Bugis", "Clarke Quay", "Chinatown", "Little India", "Novena", "Somerset", "Bras Basah", "Marina Bay", "Tanjong Pagar", "Outram Park", "Telok Blangah"],
        "LRT": []
    }
}

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
    value=44
)

house_age = st.sidebar.slider(
    "House Age (Years)",
    min_value=0,
    max_value=99,
    value=38
)

# Select Region Ura first to determine available MRT/LRT stations
region_ura = st.sidebar.selectbox(
    "Region Ura",
    options=df["region_ura"].unique()
)

# Based on selected Region Ura, dynamically update available MRT/LRT stations
available_mrt_stations = region_to_transport[region_ura]["MRT"]
available_lrt_stations = region_to_transport[region_ura]["LRT"]

# Select transport type: MRT or LRT
transport_type = st.sidebar.selectbox(
    "Transport Type",
    options=["MRT", "LRT"]
)

# Show available MRT/LRT stations based on transport type
if transport_type == "MRT":
    closest_mrt_station = st.sidebar.selectbox(
        "MRT Station",
        options=available_mrt_stations
    )
    closest_lrt_station = None  # Hide LRT station option
elif transport_type == "LRT" and available_lrt_stations:
    closest_lrt_station = st.sidebar.selectbox(
        "LRT Station",
        options=available_lrt_stations
    )
    closest_mrt_station = None  # Hide MRT station option

price_per_sqft = st.sidebar.slider(
    "Price Per Sqft (SGD)", 
    min_value= 0, 
    max_value=2000, 
    value=489
)

# Create a dictionary of user inputs
user_data = {
    "flat_type": flat_type,
    "flat_model": flat_model,
    "floor_area_sqm": floor_area_sqm,
    "house_age": house_age,
    "transport_type": transport_type,
    "closest_mrt_station": closest_mrt_station,
    "closest_lrt_station": closest_lrt_station,
    "price_per_sqft": price_per_sqft,
    "region_ura": region_ura,
}

# Convert the user input to DataFrame
input_df = pd.DataFrame([user_data])

# Show the user input
st.subheader("User Input")
st.write(input_df)

# Convert categorical features to one-hot encoding
input_df_encoded = pd.get_dummies(input_df, columns=["region_ura", "closest_mrt_station", "closest_lrt_station", "transport_type", "flat_type", "flat_model"])

# Reindex the input DataFrame to align with the model's expected columns
input_df_encoded = input_df_encoded.reindex(columns=column_names, fill_value=0)

# Make predictions using the trained model
prediction = model.predict(input_df_encoded)

# Display the predicted resale price
st.subheader("Predicted Resale Price")
st.write(f"SGD {prediction[0]:,.2f}")
