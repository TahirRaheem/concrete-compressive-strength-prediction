# app.py

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load Dataset
df = pd.read_csv("daataset.csv")

# Preprocess Data
X = df[['Cement (component 1)(kg in a m^3 mixture)', 
        'Blast Furnace Slag (component 2)(kg in a m^3 mixture)', 
        'Fly Ash (component 3)(kg in a m^3 mixture)', 
        'Water  (component 4)(kg in a m^3 mixture)', 
        'Superplasticizer (component 5)(kg in a m^3 mixture)', 
        'Coarse Aggregate  (component 6)(kg in a m^3 mixture)', 
        'Fine Aggregate (component 7)(kg in a m^3 mixture)', 
        'Age (day)']]
y = df['Concrete compressive strength(MPa, megapascals) ']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting and Random Forest Models
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Save models to disk
joblib.dump(gbr, 'gbr_model.pkl')
joblib.dump(rf, 'rf_model.pkl')

# Load models
gbr = joblib.load('gbr_model.pkl')
rf = joblib.load('rf_model.pkl')

# Streamlit App
st.title("Concrete Compressive Strength Predictor")

st.write("Enter the values of each component to predict the concrete compressive strength.")

# Input fields
cement = st.number_input("Cement (kg in a m^3 mixture):", min_value=0.0, step=0.1)
slag = st.number_input("Blast Furnace Slag (kg in a m^3 mixture):", min_value=0.0, step=0.1)
fly_ash = st.number_input("Fly Ash (kg in a m^3 mixture):", min_value=0.0, step=0.1)
water = st.number_input("Water (kg in a m^3 mixture):", min_value=0.0, step=0.1)
superplasticizer = st.number_input("Superplasticizer (kg in a m^3 mixture):", min_value=0.0, step=0.1)
coarse_agg = st.number_input("Coarse Aggregate (kg in a m^3 mixture):", min_value=0.0, step=0.1)
fine_agg = st.number_input("Fine Aggregate (kg in a m^3 mixture):", min_value=0.0, step=0.1)
age = st.number_input("Age (days):", min_value=1, step=1)

# Prediction button
if st.button("Predict"):
    # Create DataFrame for input
    input_data = pd.DataFrame([[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]], 
                              columns=X.columns)
    
    # Make predictions
    gbr_strength = gbr.predict(input_data)[0]
    rf_strength = rf.predict(input_data)[0]
    
    # Display Results
    st.write(f"Predicted Concrete Compressive Strength (MPa) - Gradient Boosting: {gbr_strength:.2f}")
    st.write(f"Predicted Concrete Compressive Strength (MPa) - Random Forest: {rf_strength:.2f}")
