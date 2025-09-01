import streamlit as st
import pickle
import numpy as np
import base64

# Page config
st.set_page_config(page_title="Crop Recommendation System", layout="wide")

# Hide Streamlit header and menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .main-content {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem auto;
        max-width: 800px;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

add_bg_from_local('istockphoto-465910852-612x612.jpg')

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open('trained_model.sav', 'rb'))

model = load_model()

# Crop mapping
crop_dict = {
    0: 'Rice', 1: 'Maize', 2: 'Chickpea', 3: 'Kidney Beans', 4: 'Pigeon Peas',
    5: 'Moth Beans', 6: 'Mung Bean', 7: 'Black Gram', 8: 'Lentil', 9: 'Pomegranate',
    10: 'Banana', 11: 'Mango', 12: 'Grapes', 13: 'Watermelon', 14: 'Muskmelon',
    15: 'Apple', 16: 'Orange', 17: 'Papaya', 18: 'Coconut', 19: 'Cotton',
    20: 'Jute', 21: 'Coffee'
}

# Main app
st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.title("🌾 Crop Recommendation System")
st.write("Enter the soil and climate parameters to get crop recommendations:")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0)
    K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

if st.button("🔍 Predict Crop", type="primary"):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)
    crop_name = crop_dict[prediction[0]]
    
    st.success(f"Recommended Crop: **{crop_name}**")
    st.balloons()

st.markdown('</div>', unsafe_allow_html=True)