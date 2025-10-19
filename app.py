import streamlit as st
import joblib
import pandas as pd

# Define the model filename
MODEL_FILE = "linear_regression_model.pkl"

def load_model():
    """Load the trained model."""
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except FileNotFoundError:
        st.error(f"Model file '{MODEL_FILE}' not found. Please ensure it is uploaded.")
        return None

# Load the model
model = load_model()

# --- Streamlit App Interface ---
st.title("💰 Sales Prediction App")
st.markdown("Estimate sales based on advertising budget across platforms.")

if model:
    # Get user input for features
    st.header("Advertising Budget (in thousands)")
    youtube_budget = st.slider("YouTube Budget", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    tiktok_budget = st.slider("TikTok Budget", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    instagram_budget = st.slider("Instagram Budget", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    
    # Prepare data for prediction
    input_data = pd.DataFrame({
        "youtube": [youtube_budget],
        "tiktok": [tiktok_budget],
        "instagram": [instagram_budget]
    })
    
    if st.button("Predict Sales"):
        # Make predictions
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Sales: ${prediction:,.2f} Thousand")
        st.info("The prediction is an estimated value based on the trained Linear Regression model.")
