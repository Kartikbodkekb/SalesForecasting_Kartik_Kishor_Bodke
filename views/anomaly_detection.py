import streamlit as st
import plotly.express as px

def render_anomaly_detection(anomaly):
    st.title("🚨 Anomaly Detection")
    st.markdown("Identify abnormal sales spikes or drops using Isolation Forest & Z-Score methods.")

    if anomaly.empty:
        st.warning("Anomaly data not available.")
        return

    st.subheader("Weekly Sales Anomalies")
    
    # Plot anomalies
    fig = px.scatter(
        anomaly,
        x="Order Date",
        y="Sales",
        color="Isolation",
        color_discrete_map={"Normal": "#3b82f6", "Anomaly": "#ef4444"},
        template="plotly_white",
        title="Sales Anomaly Scatter Plot"
    )
    # Add a line connecting the dots to see the trend
    fig.add_traces(px.line(anomaly, x="Order Date", y="Sales").data)
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Business Explanation
    Anomalies represent weeks where sales significantly deviated from the expected trend.
    - **Spikes (Positive Anomalies)**: Could indicate successful marketing campaigns, seasonal peaks, or bulk orders.
    - **Drops (Negative Anomalies)**: Could indicate supply chain issues, website downtime, or external negative factors.
    """)

    st.subheader("Anomaly Data Table")
    st.dataframe(anomaly, use_container_width=True)
    
    csv = anomaly.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Anomaly Data", data=csv, file_name="anomaly_detection.csv", mime="text/csv")
