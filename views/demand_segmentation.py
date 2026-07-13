import streamlit as st
import plotly.express as px

def render_demand_segmentation(demand):
    st.title("🎯 Demand Segmentation")
    st.markdown("Product categorization based on demand patterns using K-Means Clustering.")

    if demand.empty:
        st.warning("Demand segmentation data not available.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("PCA Cluster Plot")
        # Assuming PCA components exist in the dataset (e.g., PCA1, PCA2).
        # If not, we plot based on Total_Sales and Frequency or similar metrics.
        x_col = "PCA1" if "PCA1" in demand.columns else "Total_Sales"
        y_col = "PCA2" if "PCA2" in demand.columns else (
            "Frequency" if "Frequency" in demand.columns else demand.columns[2]
        )
        
        fig = px.scatter(
            demand,
            x=x_col,
            y=y_col,
            color="Demand Segment",
            hover_data=["Sub-Category"],
            template="plotly_white",
            title="Product Segments"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Segment Distribution")
        segment_counts = demand["Demand Segment"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Count"]
        
        fig_pie = px.pie(
            segment_counts,
            names="Segment",
            values="Count",
            hole=0.4,
            template="plotly_white"
        )
        fig_pie.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Cluster Summary & Recommendations")
    st.dataframe(demand, use_container_width=True)
    
    csv = demand.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Segmentation Data", data=csv, file_name="demand_segmentation.csv", mime="text/csv")

    st.markdown("""
    ### Inventory Strategy
    * **High Demand**: Keep high stock levels, negotiate better rates with suppliers, ensure no stock-outs.
    * **Medium Demand**: Optimize reorder points, monitor closely for shifts to high or low demand.
    * **Low Demand**: Reduce inventory, consider bundling with high-demand items, or phase out if unprofitable.
    """)
