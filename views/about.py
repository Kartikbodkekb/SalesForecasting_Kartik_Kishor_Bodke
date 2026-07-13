import streamlit as st

def render_about():
    st.title("📖 About Project")
    
    st.markdown("""
    ## End-to-End Sales Forecasting & Demand Intelligence System
    
    ### 🎯 Project Overview
    This project is an advanced analytics solution designed to analyze historical sales data, forecast future demand, detect anomalies, and segment products. It empowers businesses to make data-driven decisions regarding inventory management, marketing strategies, and operational efficiency.
    
    ### 🚀 Objectives
    - **Data Processing**: Clean and preprocess raw transaction data.
    - **Exploratory Data Analysis (EDA)**: Uncover hidden patterns and insights.
    - **Time Series Forecasting**: Predict future sales at macro (overall), regional, and category levels.
    - **Anomaly Detection**: Automatically flag unusual sales weeks using Isolation Forest and Z-Score techniques.
    - **Demand Segmentation**: Cluster sub-categories into distinct demand profiles using PCA and K-Means.
    
    ### 🗂️ Dataset Description
    The system uses retail sales transaction data spanning multiple years, encompassing order details, customer demographics, product hierarchies (Category, Sub-Category), and financial metrics (Sales, Profit, Quantities).
    
    ### 💻 Technologies & Models Used
    - **Language**: Python
    - **Data Manipulation**: Pandas, NumPy
    - **Machine Learning**: Scikit-Learn (Isolation Forest, K-Means, PCA), XGBoost
    - **Time Series**: Statsmodels (SARIMA), Prophet
    - **Visualization**: Plotly, Streamlit
    
    ### 🔮 Future Scope
    - **Real-time Data Integration**: Connecting directly to ERP/CRM systems or SQL databases.
    - **Advanced Deep Learning Models**: Implementing LSTMs or Transformers for complex forecasting.
    - **Price Optimization**: Adding dynamic pricing recommendations based on demand elasticity.
    - **Automated Alerts**: Email or Slack notifications for real-time anomaly detection.
    
    ---
    *Developed as a comprehensive project demonstrating End-to-End Machine Learning and Analytics workflows.*
    """)
