import streamlit as st
import pickle
import pandas as pd

MODEL_FILE = "linear_regression_model.pkl"

def load_model():
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found!")
        return None

model = load_model()

st.title("💰 Sales Prediction App")
st.markdown("Estimate sales based on advertising budget")

if model:
    st.header("Advertising Budget (in thousands)")
    youtube = st.slider("YouTube", 0.0, 100.0, 50.0, 0.1)
    tiktok = st.slider("TikTok", 0.0, 100.0, 50.0, 0.1)
    instagram = st.slider("Instagram", 0.0, 100.0, 50.0, 0.1)
    
    if st.button("Predict Sales"):
        input_data = pd.DataFrame({
            "youtube": [youtube],
            "tiktok": [tiktok],
            "instagram": [instagram]
        })
        pred = model.predict(input_data)[0]
        st.success(f"Estimated Sales: ${pred:,.2f}K")
```

**requirements.txt:**
```
streamlit
pandas
scikit-learn
