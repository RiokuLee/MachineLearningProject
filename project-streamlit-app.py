import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://img.tepcdn.com/img-style/simplecrop_article/83444555.jpg');
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
    }

    /* Add text shadow to regular text content */
    .css-1d391kg {  /* For regular text content */
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
    }

    /* Add text shadow to titles and subheaders */
    .css-1u6dpbm {  /* For titles/subheaders */
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
    }

    /* Add text shadow to data table content */
    .stDataFrame {  /* For tables */
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
    }

    /* Optional: Add shadow to headers */
    h1, h2, h3 {
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True
)
# Load the pre-trained model and column names (ensure these files are saved during training)
model = joblib.load(open("model_gdr.pkl", "rb"))
column_names = joblib.load(open("model_columns.pkl", "rb"))

# Load the preprocessed data
df = joblib.load('preprocessed_data.pkl')

# Title of the Streamlit app
st.title("HDB Resale Price Prediction")

# Define region to transport mapping
region_to_transport = {
    "NORTH-EAST REGION": {
        "MRT": [
            "Ang Mo Kio", "Bartley", "Bright Hill", "Buangkok", "Hougang", "Kovan", "Lentor", 
            "Lorong Chuan", "Mayflower", "Punggol", "Sengkang", "Serangoon", "Yio Chu Kang"
        ],
        "LRT": [
            "Bakau", "Cheng Lim", "Compassvale", "Coral Edge", "Cove", "Damai", "Farmway", 
            "Fernvale", "Kadaloor", "Kangkar", "Kupang", "Layar", "Meridian", "Nibong", "Oasis", 
            "Punggol", "Ranggung", "Riviera", "Rumbia", "Sengkang", "Soo Teck", "Sumang", 
            "Thanggam", "Tongkang"
        ]
    },
    "NORTH REGION": {
        "MRT": [
            "Admiralty", "Canberra", "Khatib", "Marsiling", "Sembawang", "Woodlands", 
            "Woodlands North", "Woodlands South", "Yishun"
        ],
        "LRT": []
    },
    "EAST REGION": {
        "MRT": [
            "Bedok", "Bedok North", "Bedok Reservoir", "Changi Airport", "Kaki Bukit", "Kembangan", 
            "Pasir Ris", "Simei", "Tampines", "Tampines East", "Tampines West", "Tanah Merah", "Ubi", 
            "Upper Changi"
        ],
        "LRT": []
    },
    "WEST REGION": {
        "MRT": [
            "Boon Lay", "Bukit Batok", "Bukit Gombak", "Bukit Panjang", "Chinese Garden", "Chao Chu Kang", 
            "Clementi", "Dover", "Jurong East", "Lakeside", "Pioneer", "Yew Tee"
        ],
        "LRT": [
            "Boon Lay", "Bukit Batok", "Bukit Gombak", "Bukit Panjang", "Chinese Garden", "Chao Chu Kang", 
            "Clementi", "Dover", "Jurong East", "Lakeside", "Pioneer", "Yew Tee"
        ]
    },
    "CENTRAL REGION": {
        "MRT": [
            "Aljunied", "Ang Mo Kio", "Bartley", "Beauty World", "Bencoolen", "Bendemeer", "Bishan", 
            "Boon Keng", "Braddell", "Bras Basah", "Bright Hill", "Bugis", "Buona Vista", "Caldecott", 
            "Chinatown", "Commonwealth", "Dakota", "Dover", "Eunos", "Farrer Park", "Farrer Road", 
            "Geylang Bahru", "Great World", "Harbourfront", "Havelock", "Holland Village", "Jalan Besar", 
            "Kallang", "Kembangan", "Labrador Park", "Lavender", "Little India", "MacPherson", "Marymount", 
            "Matter", "Maxwell", "Mountbatten", "Nicoll Highway", "Novena", "One-North", "Outram Park", 
            "Paya Lebar", "Potong Pasir", "Queenstown", "Redhill", "Rochor", "Tai Seng", "Tanjong Pagar", 
            "Telok Blangah", "Tiong Bahru", "Toa Payoh", "Ubi", "Upper Thomson", "Woodleigh"
        ],
        "LRT": []
    }
}

# Sidebar for user input
st.sidebar.header("Input Features")

# Input fields for user interaction (same as your current code)
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

# Extract feature importances from the model
feature_importances = model.feature_importances_

# Create a DataFrame to store the feature importances
feature_importance_df = pd.DataFrame({
    'Feature': column_names,
    'Importance': feature_importances
})

simplified_importance = {
    'Region Ura': feature_importance_df[feature_importance_df['Feature'].str.contains('region_ura')]['Importance'].sum(),
    'Transport Type': feature_importance_df[feature_importance_df['Feature'].str.contains('transport_type')]['Importance'].sum(),
    'MRT&LRT Station': feature_importance_df[feature_importance_df['Feature'].str.contains('closest_mrt_station')]['Importance'].sum(),
    'Flat Type': feature_importance_df[feature_importance_df['Feature'].str.contains('flat_type')]['Importance'].sum(),
    'Flat Model': feature_importance_df[feature_importance_df['Feature'].str.contains('flat_model')]['Importance'].sum(),
    'Floor Area (sqm)': feature_importance_df[feature_importance_df['Feature'] == 'floor_area_sqm']['Importance'].values[0],
    'House Age': feature_importance_df[feature_importance_df['Feature'] == 'house_age']['Importance'].values[0],
    'Price Per Sqft': feature_importance_df[feature_importance_df['Feature'] == 'price_per_sqft']['Importance'].values[0]
}

# Convert to a DataFrame for plotting
simplified_importance_df = pd.DataFrame(list(simplified_importance.items()), columns=['Feature', 'Importance'])

# Plot the feature importance
st.subheader("Feature Importance (Simplified)")
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=simplified_importance_df, palette='viridis')
plt.title("Feature Importance")
st.pyplot(plt)

# Display the feature importance table
st.subheader("Feature Importance Table")
st.write(simplified_importance_df)