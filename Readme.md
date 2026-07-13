# End-to-End Sales Forecasting & Demand Intelligence System

![Dashboard Preview](https://via.placeholder.com/1200x600?text=Sales+Forecasting+Dashboard)

## 📌 Project Overview
This repository contains a comprehensive **End-to-End Sales Forecasting and Demand Intelligence System**. The project transitions from raw transactional data to actionable business insights using Machine Learning, Time Series Analysis, and an interactive Streamlit web application.

## 🚀 Key Features
1. **Interactive Dashboard**: Modern UI with key performance indicators (KPIs), dataset summaries, and business insights.
2. **Sales Analysis**: Slice and dice historical sales data using multi-select filters, visualizing top products, categories, and regional performance.
3. **Forecasting**: Advanced time series forecasting using **SARIMA**, **Prophet**, and **XGBoost** to predict macro, regional, and category-level demand.
4. **Anomaly Detection**: Intelligent flagging of abnormal sales patterns (spikes and drops) using **Isolation Forest** and **Z-Score** methodologies.
5. **Demand Segmentation**: AI-driven product clustering using **PCA** and **K-Means** to categorize inventory into High, Medium, and Low demand tiers.

## 📂 Project Structure
```text
SalesForecasting/
│
├── app.py                     # Main Streamlit application entry point
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
│
├── views/                     # Modular Streamlit page views
│   ├── dashboard.py
│   ├── sales_analysis.py
│   ├── forecasting.py
│   ├── anomaly_detection.py
│   ├── demand_segmentation.py
│   └── about.py
│
├── utils/                     # Helper functions
│   └── data_loader.py         # Data ingestion and caching
│
├── data/                      # CSV datasets
│   ├── train.csv
│   ├── category_forecast.csv
│   ├── region_forecast.csv
│   ├── weekly_anomaly_detection.csv
│   └── demand_segmentation.csv
│
├── charts/                    # Static saved visual charts (if any)
└── report/                    # Business reports and summaries
    └── executive_summary.md
```

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/sales-forecasting-dashboard.git
   cd sales-forecasting-dashboard
   ```

2. **Create a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## 🌐 Deployment (Streamlit Community Cloud)
To deploy this app for free on Streamlit Community Cloud:
1. Push this repository to GitHub.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click "New App".
4. Select your repository, branch, and set the main file path as `app.py`.
5. Click "Deploy". No extra configuration is needed as long as `requirements.txt` is in the root.

## 👨‍💻 Developer
**Kartik Bodke**  
*Data Science Intern @ XYlofy AI*
