# -*- coding: utf-8 -*-
"""app
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# ----------------------------------------------------------------------------
# 1. DEFINE THE CUSTOM TRANSFORMER CLASS
# ----------------------------------------------------------------------------
# Streamlit needs the *exact* class definition
# to be in the same file as where you load the model.

class Per90Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything, so we just return self
        return self

    def transform(self, X, y=None):
        # Make a copy to avoid changing the original data
        X_copy = X.copy()

        # Calculate 90s (minutes / 90)
        # We must be careful about division by zero if Min is 0
        X_copy['90s'] = X_copy['Min'] / 90

        # Create 'per 90' features
        # We use a loop to handle potential division by zero
        for col in ['xG', 'Sh', 'SoT', 'Ast']:
            new_col_name = f"{col}_per_90"
            # Use np.where to safely divide
            X_copy[new_col_name] = np.where(
                X_copy['90s'] > 0, 
                X_copy[col] / X_copy['90s'], 
                0
            )

        # Replace any inf/-inf values (just in case, though np.where should handle it)
        X_copy.replace([np.inf, -np.inf], 0, inplace=True)

        # Select and return only the features our model was trained on
        final_features = ['Age', 'MP', 'xG_per_90', 'Sh_per_90', 'SoT_per_90', 'Ast_per_90']
        
        # Ensure the output is in the correct order and type
        return X_copy[final_features].astype(float)

# ----------------------------------------------------------------------------
# 2. LOAD THE TRAINED MODEL
# ----------------------------------------------------------------------------
# This file 'football_goal_predictor.joblib' MUST be in your GitHub repo
try:
    pipeline = joblib.load('football_goal_predictor.joblib')
except FileNotFoundError:
    st.error("Model file ('football_goal_predictor.joblib') not found. Please upload this file to your GitHub repo.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading the model: {e}")
    st.image("https://i.imgur.com/x06nXYM.png", caption="This error is likely caused by a version mismatch. See instructions.")
    st.stop()

# ----------------------------------------------------------------------------
# 3. SET UP THE STREAMLIT USER INTERFACE
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Football Goal Predictor", layout="wide")
st.title('⚽ Football Player Season Goal Predictor')
st.markdown("Enter a player's *full season* stats to predict how many goals they will score.")

st.sidebar.header("Player Input Features")
st.sidebar.markdown("Enter the player's seasonal data below.")

# --- Create the input form in the sidebar ---
with st.sidebar:
    age = st.number_input('Age', min_value=15, max_value=45, value=25, help="Player's age at the start of the season.")
    mp = st.number_input('Matches Played (MP)', min_value=1, max_value=60, value=20, help="Total matches played.")
    min_played = st.number_input('Total Minutes Played (Min)', min_value=90, max_value=6000, value=1800, help="Total minutes played all season. Must be at least 90.")
    
    st.divider() # Visual separator
    
    xg = st.number_input('Expected Goals (xG)', min_value=0.0, value=5.0, format="%.2f", help="Total Expected Goals for the season.")
    sh = st.number_input('Total Shots (Sh)', min_value=0, value=30, help="Total shots taken.")
    sot = st.number_input('Shots on Target (SoT)', min_value=0, value=15, help="Total shots on target.")
    ast = st.number_input('Total Assists (Ast)', min_value=0, value=5, help="Total assists for the season.")

    st.divider()

    # The prediction button
    predict_button = st.button('Predict Goals', type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# 4. PREDICTION LOGIC
# ----------------------------------------------------------------------------
if predict_button:
    # This list MUST match the columns your pipeline expects to receive
    # These are the *raw* features before the transformer runs
    feature_names = ['Age', 'MP', 'Min', 'xG', 'Sh', 'SoT', 'Ast']

    # Create a dictionary with the user's input
    user_data = {
        'Age': age,
        'MP': mp,
        'Min': min_played,
        'xG': xg,
        'Sh': sh,
        'SoT': sot,
        'Ast': ast
    }

    # Create a single-row DataFrame
    # This is the *exact* format your pipeline's transformer expects
    try:
        input_df = pd.DataFrame([user_data], columns=feature_names)

        st.subheader("Raw Input Data (Before Transformation):")
        st.dataframe(input_df)

        # Call the pipeline's predict function
        prediction = pipeline.predict(input_df)

        # Get the first (and only) prediction
        predicted_goals = prediction[0]

        st.markdown("---")
        st.subheader("📈 Prediction Result")
        st.success(f"**The model predicts this player will score {predicted_goals:.1f} goals this season.**")

        # Show a rounded number for a cleaner display
        st.metric(label="Predicted Goals (Rounded)", value=f"{round(predicted_goals)}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.exception(e)
