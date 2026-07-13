import streamlit as st
import pandas as pd
import plotly.express as px

def render_sales_analysis(train):
    st.title("📊 Sales Analysis")
    st.markdown("Interactive exploration of historical sales data.")

    if train.empty:
        st.warning("Data not available.")
        return

    # ---- Filters ----
    st.sidebar.markdown("### Filters")
    years = sorted(train["Order Date"].dt.year.dropna().unique())
    selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years)

    regions = sorted(train["Region"].dropna().unique()) if "Region" in train.columns else []
    selected_regions = st.sidebar.multiselect("Select Region(s)", regions, default=regions)

    categories = sorted(train["Category"].dropna().unique()) if "Category" in train.columns else []
    selected_categories = st.sidebar.multiselect("Select Category(s)", categories, default=categories)

    # Filter data
    filtered_df = train[
        (train["Order Date"].dt.year.isin(selected_years)) &
        (train["Region"].isin(selected_regions) if regions else True) &
        (train["Category"].isin(selected_categories) if categories else True)
    ]

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return

    # ---- Monthly Trend (Filtered) ----
    st.subheader("Sales Trend (Filtered)")
    monthly_trend = filtered_df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum().reset_index()
    fig_trend = px.line(monthly_trend, x="Order Date", y="Sales", markers=True, template="plotly_white")
    st.plotly_chart(fig_trend, use_container_width=True)

    # ---- Category & Region Sales ----
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sales by Category")
        if "Category" in filtered_df.columns:
            cat_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index().sort_values(by="Sales", ascending=False)
            fig_cat = px.bar(cat_sales, x="Category", y="Sales", color="Category", template="plotly_white")
            st.plotly_chart(fig_cat, use_container_width=True)
            
    with col2:
        st.subheader("Sales by Region")
        if "Region" in filtered_df.columns:
            reg_sales = filtered_df.groupby("Region")["Sales"].sum().reset_index()
            fig_reg = px.pie(reg_sales, values="Sales", names="Region", hole=0.4, template="plotly_white")
            st.plotly_chart(fig_reg, use_container_width=True)

    # ---- Top Products ----
    st.subheader("Top Products")
    if "Product Name" in filtered_df.columns:
        top_products = filtered_df.groupby("Product Name")["Sales"].sum().reset_index().sort_values(by="Sales", ascending=False).head(10)
        fig_prod = px.bar(top_products, x="Sales", y="Product Name", orientation="h", template="plotly_white", color="Sales", color_continuous_scale="Blues")
        fig_prod.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_prod, use_container_width=True)

    # ---- Download Button ----
    st.markdown("---")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Sales Data",
        data=csv,
        file_name="filtered_sales_data.csv",
        mime="text/csv",
    )
