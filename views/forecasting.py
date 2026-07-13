import streamlit as st
import plotly.express as px
import pandas as pd

def render_forecasting(category_forecast, region_forecast, model_comp):
    st.title("🔮 Forecasting")
    st.markdown("Future sales predictions and model performance evaluation.")

    tab1, tab2, tab3 = st.tabs(["Category Forecast", "Region Forecast", "Model Comparison"])

    with tab1:
        st.subheader("Category Forecast")
        if not category_forecast.empty:
            cat_melted = category_forecast.melt(id_vars=["Date"], var_name="Category", value_name="Forecasted Sales")
            fig_cat = px.line(cat_melted, x="Date", y="Forecasted Sales", color="Category", template="plotly_white")
            st.plotly_chart(fig_cat, use_container_width=True)
            st.dataframe(category_forecast, use_container_width=True)
            csv_cat = category_forecast.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Category Forecast", data=csv_cat, file_name="category_forecast.csv", mime="text/csv", key="cat_dl")
        else:
            st.warning("Category forecast data not available.")

    with tab2:
        st.subheader("Region Forecast")
        if not region_forecast.empty:
            reg_melted = region_forecast.melt(id_vars=["Date"], var_name="Region", value_name="Forecasted Sales")
            fig_reg = px.line(reg_melted, x="Date", y="Forecasted Sales", color="Region", template="plotly_white")
            st.plotly_chart(fig_reg, use_container_width=True)
            st.dataframe(region_forecast, use_container_width=True)
            csv_reg = region_forecast.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Region Forecast", data=csv_reg, file_name="region_forecast.csv", mime="text/csv", key="reg_dl")
        else:
            st.warning("Region forecast data not available.")

    with tab3:
        st.subheader("Model Comparison (SARIMA vs Prophet vs XGBoost)")
        if not model_comp.empty:
            st.dataframe(model_comp, use_container_width=True)
            csv_model = model_comp.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Model Comparison", data=csv_model, file_name="model_comparison.csv", mime="text/csv", key="model_dl")
            
            # Melt for grouped bar chart
            metrics_melted = model_comp.melt(id_vars=["Model"], value_vars=["MAE", "RMSE", "MAPE"], var_name="Metric", value_name="Value")
            
            fig_metrics = px.bar(
                metrics_melted, 
                x="Model", 
                y="Value", 
                color="Metric", 
                barmode="group",
                template="plotly_white",
                title="Error Metrics Comparison"
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        else:
            st.info("Model comparison data is not currently available.")
