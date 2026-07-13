import streamlit as st

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Demand Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .css-18e3th9 {
        padding-top: 2rem;
    }
    .css-1d391kg {
        padding-top: 2rem;
    }
    /* Style for metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Import the data loader and view components
from utils.data_loader import load_data
from views.dashboard import render_dashboard
from views.sales_analysis import render_sales_analysis
from views.forecasting import render_forecasting
from views.anomaly_detection import render_anomaly_detection
from views.demand_segmentation import render_demand_segmentation
from views.about import render_about

# --------------------------------------------------
# Load Data
# --------------------------------------------------
train, category_forecast, region_forecast, anomaly, demand, model_comp = load_data()

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Sales Analysis",
        "Forecasting",
        "Anomaly Detection",
        "Demand Segmentation",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("AI Driven Sales Forecasting & Demand Analytics System.")
st.sidebar.caption("v1.0 | Kartik Bodke")

# ==================================================
# PAGE ROUTING
# ==================================================
if page == "Dashboard":
    render_dashboard(train, category_forecast, region_forecast, anomaly, demand, model_comp)
elif page == "Sales Analysis":
    render_sales_analysis(train)
elif page == "Forecasting":
    render_forecasting(category_forecast, region_forecast, model_comp)
elif page == "Anomaly Detection":
    render_anomaly_detection(anomaly)
elif page == "Demand Segmentation":
    render_demand_segmentation(demand)
elif page == "About Project":
    render_about()