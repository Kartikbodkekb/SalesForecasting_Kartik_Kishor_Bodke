import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    """Load and preprocess the required datasets for the dashboard."""
    try:
        train = pd.read_csv("data/train.csv")
        # Try to parse with standard formats, falling back to mixed if necessary
        train["Order Date"] = pd.to_datetime(train["Order Date"], format="%d/%m/%Y", errors="coerce")
        # For any rows that failed to parse (NaT), try another format if needed, though %d/%m/%Y should work based on sample
        if train["Order Date"].isna().any():
            train["Order Date"] = pd.to_datetime(train["Order Date"], errors="coerce")
            
        category_forecast = pd.read_csv("data/category_forecast.csv")
        category_forecast.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
        category_forecast["Date"] = pd.to_datetime(category_forecast["Date"])

        region_forecast = pd.read_csv("data/region_forecast.csv")
        region_forecast.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
        region_forecast["Date"] = pd.to_datetime(region_forecast["Date"])

        anomaly = pd.read_csv("data/weekly_anomaly_detection.csv")
        anomaly["Order Date"] = pd.to_datetime(anomaly["Order Date"])

        demand = pd.read_csv("data/demand_segmentation.csv")
        
        # We might also have model_comparison.csv
        try:
            model_comp = pd.read_csv("data/model_comparison.csv")
        except FileNotFoundError:
            model_comp = pd.DataFrame()

        return train, category_forecast, region_forecast, anomaly, demand, model_comp
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
