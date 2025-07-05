# Import all the necessary libraries
import pandas as pd
import numpy as np
import joblib
import pickle
import streamlit as st
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Water Quality Predictor", page_icon="🌊", layout="wide",initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .input-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
        color: #222 !important;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1f77b4;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .number-input {
        background: white;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .text-input {
        background: white;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .tips-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        color: #222 !important;
    }
    
    .interpretation-box {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        color: #222 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and structure
@st.cache_resource
def load_model():
    model = joblib.load("polution_model.pkl")
    model_cols = joblib.load("model_column.pkl")
    return model, model_cols

model, model_cols = load_model()

# Header
st.markdown("""
<div class="main-header">
    <h1>🌊 Water Quality Predictor</h1>
    <p>Advanced AI-powered water pollutant prediction system</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for additional info
with st.sidebar:
    st.markdown("### 📊 About")
    st.markdown("""
    This application uses machine learning to predict water pollutant levels based on:
    - **Year**: Temporal analysis (2000-2100)
    - **Station ID**: Geographic location identifier
    
    **Predicted Pollutants:**
    - O₂ (Dissolved Oxygen)
    - NO₃ (Nitrate)
    - NO₂ (Nitrite)
    - SO₄ (Sulfate)
    - PO₄ (Phosphate)
    - CL (Chloride)
    """)
    
    st.markdown("### 📈 Model Info")
    st.info("Trained on historical water quality data with high accuracy predictions")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="input-section">
        <h3>🔍 Input Parameters</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Input fields with better styling
    col_a, col_b = st.columns(2)
    
    with col_a:
        year_input = st.number_input("📅 Year",min_value=2000,max_value=2100,value=2025,help="Select the year for prediction (2000-2100)")
    
    with col_b:
        station_id = st.text_input( "📍 Station ID", value='2', help="Enter the monitoring station identifier")

with col2:
    st.markdown("""
    <div class="tips-box">
        <h4>💡 Tips</h4>
        <ul style="margin: 0; padding-left: 1.2rem; color: #222;">
            <li>Use years 2000-2021 for historical accuracy</li>
            <li>Station IDs are location-specific</li>
            <li>Results show pollutant concentrations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Prediction button
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_button = st.button('🚀 Predict Water Quality', use_container_width=True)

# Prediction logic
if predict_button:
    if not station_id:
        st.error('⚠️ Please enter a valid Station ID')
    else:
        try:
            # Prepare the input
            input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
            input_encoded = pd.get_dummies(input_df, columns=['id'])

            # Align with model cols
            for col in model_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[model_cols]

            # Predict
            predicted_pollutants = model.predict(input_encoded)[0]
            pollutants = ['O₂', 'NO₃', 'NO₂', 'SO₄', 'PO₄', 'CL']
            pollutant_names = ['Dissolved Oxygen', 'Nitrate', 'Nitrite', 'Sulfate', 'Phosphate', 'Chloride']
            
            # Display results
            st.markdown("""
            <div class="prediction-card">
                <h2>📊 Prediction Results</h2>
                <p>Water quality predictions for Station <strong>{}</strong> in <strong>{}</strong></p>
            </div>
            """.format(station_id, year_input), unsafe_allow_html=True)
            
            # Create metrics display
            col1, col2, col3 = st.columns(3)
            
            for i, (pollutant, name, value) in enumerate(zip(pollutants, pollutant_names, predicted_pollutants)):
                with col1 if i < 2 else col2 if i < 4 else col3:
                    st.metric(label=f"{pollutant} ({name})", value=f"{value:.2f}", delta=None)
            
            # Additional info
            st.markdown("---")
            st.markdown("""
            <div class="interpretation-box">
                <h4>📋 Interpretation Guide</h4>
                <ul style="color: #222;">
                    <li><strong>O₂ (Dissolved Oxygen):</strong> Higher values indicate better water quality</li>
                    <li><strong>NO₃, NO₂ (Nitrogen compounds):</strong> Indicators of nutrient pollution</li>
                    <li><strong>SO₄ (Sulfate):</strong> Natural or industrial contamination</li>
                    <li><strong>PO₄ (Phosphate):</strong> Agricultural runoff indicator</li>
                    <li><strong>CL (Chloride):</strong> Salinity and industrial waste indicator</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.info("Please check your input values and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌊 Water Quality Prediction System | Powered by Machine Learning</p>
</div>
""".format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)