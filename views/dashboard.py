import streamlit as st
import pandas as pd
import plotly.express as px

def render_dashboard(train, category_forecast, region_forecast, anomaly, demand, model_comp):
    st.title("📈 Executive Dashboard")
    st.markdown("A comprehensive view of sales performance, profitability, and demand trends.")

    if train.empty:
        st.warning("Data not available.")
        return

    # ---- KPI Cards ----
    total_sales = train["Sales"].sum()
    total_profit = train["Profit"].sum() if "Profit" in train.columns else 0
    total_orders = train["Order ID"].nunique() if "Order ID" in train.columns else len(train)
    total_customers = train["Customer ID"].nunique() if "Customer ID" in train.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("💰 Total Sales")
        st.metric(label="Revenue", value=f"${total_sales:,.0f}")
    with c2:
        st.success("📈 Total Profit")
        st.metric(label="Profit", value=f"${total_profit:,.0f}")
    with c3:
        st.warning("📦 Total Orders")
        st.metric(label="Orders", value=f"{total_orders:,}")
    with c4:
        st.error("👥 Total Customers")
        st.metric(label="Customers", value=f"{total_customers:,}")

    st.markdown("---")

    # ---- Monthly Sales Trend ----
    st.subheader("Monthly Sales Trend")
    monthly_sales = (
        train.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
        .sum()
        .reset_index()
    )
    fig_monthly = px.area(
        monthly_sales, 
        x="Order Date", 
        y="Sales", 
        title="Historical Monthly Sales",
        template="plotly_white",
        color_discrete_sequence=["#3b82f6"]
    )
    fig_monthly.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_monthly, use_container_width=True)

    st.markdown("---")

    # ---- Dataset Summary & Insights ----
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Summary")
        summary_df = pd.DataFrame({
            "Metric": ["Total Records", "Columns", "Unique Regions", "Unique Categories", "Unique Sub-Categories"],
            "Value": [
                train.shape[0], 
                train.shape[1], 
                train["Region"].nunique() if "Region" in train.columns else 0, 
                train["Category"].nunique() if "Category" in train.columns else 0, 
                train["Sub-Category"].nunique() if "Sub-Category" in train.columns else 0
            ]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Quick Business Insights")
        top_category = train.groupby("Category")["Sales"].sum().idxmax() if "Category" in train.columns else "N/A"
        top_region = train.groupby("Region")["Sales"].sum().idxmax() if "Region" in train.columns else "N/A"
        st.markdown(f"""
        - 🌟 **Top Performing Category**: {top_category}
        - 🌎 **Top Performing Region**: {top_region}
        - 📅 **Data Span**: {train['Order Date'].min().strftime('%Y-%m-%d')} to {train['Order Date'].max().strftime('%Y-%m-%d')}
        - 📊 **Average Order Value**: ${(total_sales/total_orders):,.2f}
        """)
