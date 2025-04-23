import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from utils.data_generator import load_data
from utils.risk_calculations import calculate_counterparty_risk_score, generate_risk_time_series
from utils.visualization import plot_risk_heatmap, plot_time_series, plot_risk_distribution

# Page config
st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.title("Risk Analytics Dashboard")

# Load data
df_counterparties = load_data("counterparties")
df_transactions = load_data("transactions")
df_market_data = load_data("market_data")
df_compliance = load_data("compliance")

# Summary metrics row
st.subheader("Key Risk Metrics")
col1, col2, col3, col4 = st.columns(4)

# Calculate metrics
total_exposure = df_counterparties['current_exposure'].sum()
avg_risk_score = df_counterparties['risk_score'].mean()
high_risk_counterparties = len(df_counterparties[df_counterparties['risk_score'] > 0.7])
non_compliant = len(df_compliance[df_compliance['status'] == 'Non-Compliant'])

with col1:
    st.metric(
        "Total Exposure", 
        f"${total_exposure/1000000:.2f}M",
        delta=f"{np.random.uniform(-5, 5):.1f}%"
    )

with col2:
    st.metric(
        "Average Risk Score", 
        f"{avg_risk_score:.2f}",
        delta=f"{np.random.uniform(-0.1, 0.1):.2f}"
    )

with col3:
    st.metric(
        "High-Risk Counterparties", 
        f"{high_risk_counterparties}",
        delta=f"{np.random.randint(-2, 3)}"
    )

with col4:
    st.metric(
        "Non-Compliant Entities", 
        f"{non_compliant}",
        delta=f"{np.random.randint(-1, 2)}"
    )

# Risk over time
st.subheader("Risk Trends")
tab1, tab2 = st.tabs(["Overall Risk", "Risk by Sector"])

with tab1:
    # Generate time series data
    dates, risk_values = generate_risk_time_series(days=30, base_value=avg_risk_score, volatility=0.03)
    
    # Plot time series
    fig = plot_time_series(dates, risk_values, title="Overall Risk Score Trend", y_label="Risk Score")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Aggregate risk by sector
    sector_risk = df_counterparties.groupby('sector')['risk_score'].mean().reset_index()
    sector_risk['exposure'] = df_counterparties.groupby('sector')['current_exposure'].sum().values
    
    # Create bar chart
    fig = px.bar(
        sector_risk,
        x='sector',
        y='risk_score',
        color='risk_score',
        color_continuous_scale='RdYlGn_r',
        title="Average Risk Score by Sector",
        hover_data=['exposure']
    )
    
    fig.update_layout(
        xaxis_title="Sector",
        yaxis_title="Average Risk Score",
        coloraxis_colorbar=dict(title="Risk Score")
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Risk Distribution
st.subheader("Risk Distribution")
col1, col2 = st.columns(2)

with col1:
    # Risk score distribution
    fig = plot_risk_distribution(df_counterparties['risk_score'].values, title="Counterparty Risk Score Distribution")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Risk by country
    country_risk = df_counterparties.groupby('country')['risk_score'].mean().reset_index()
    country_exposure = df_counterparties.groupby('country')['current_exposure'].sum().reset_index()
    country_risk['exposure'] = country_exposure['current_exposure']
    
    fig = px.scatter(
        country_risk,
        x='risk_score',
        y='exposure',
        color='risk_score',
        size='exposure',
        hover_name='country',
        color_continuous_scale='RdYlGn_r',
        title="Risk vs. Exposure by Country"
    )
    
    fig.update_layout(
        xaxis_title="Average Risk Score",
        yaxis_title="Total Exposure",
        coloraxis_colorbar=dict(title="Risk Score")
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Recent transactions
st.subheader("Recent High-Risk Transactions")

# Sort by timestamp and risk score
recent_transactions = df_transactions.sort_values(['timestamp', 'risk_score'], ascending=[False, False]).head(10)

# Format for display
display_transactions = recent_transactions[['counterparty_name', 'transaction_type', 'amount', 'currency', 'risk_score', 'timestamp']]
display_transactions = display_transactions.rename(columns={
    'counterparty_name': 'Counterparty',
    'transaction_type': 'Type',
    'amount': 'Amount',
    'currency': 'Currency',
    'risk_score': 'Risk Score',
    'timestamp': 'Timestamp'
})

# Use DataFrame styling
def highlight_risk(val):
    if isinstance(val, float):
        color = f'rgba({int(255*val)}, {int(255*(1-val))}, 0, 0.2)'
        return f'background-color: {color}'
    return ''

styled_transactions = display_transactions.style.applymap(highlight_risk, subset=['Risk Score'])

st.dataframe(display_transactions, height=300, use_container_width=True)

# Risk Heatmap
st.subheader("Risk Heatmap: Sector vs. Credit Rating")

# Create heatmap
sector_credit_risk = df_counterparties.pivot_table(
    index='sector', 
    columns='credit_rating', 
    values='risk_score', 
    aggfunc='mean'
).fillna(0)

fig = plot_risk_heatmap(
    df_counterparties, 
    'credit_rating', 
    'sector', 
    'risk_score', 
    title="Risk Heatmap by Sector and Credit Rating"
)

st.plotly_chart(fig, use_container_width=True)

# Alerts and notifications
st.sidebar.header("Risk Alerts")

# Generate some sample alerts
alerts = [
    {"level": "High", "message": "Counterparty XYZ exceeded exposure limit", "time": "10 minutes ago"},
    {"level": "Medium", "message": "Unusual transaction pattern detected", "time": "1 hour ago"},
    {"level": "Low", "message": "Market volatility increased in Energy sector", "time": "3 hours ago"},
    {"level": "High", "message": "Compliance check failed for Counterparty ABC", "time": "5 hours ago"}
]

for alert in alerts:
    level_color = {
        "High": "red",
        "Medium": "orange",
        "Low": "green"
    }.get(alert["level"], "gray")
    
    st.sidebar.markdown(
        f"""
        <div style="padding: 10px; border-left: 4px solid {level_color}; margin-bottom: 10px;">
            <strong>{alert["level"]} Alert:</strong> {alert["message"]}<br>
            <small>{alert["time"]}</small>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Refresh data button
if st.sidebar.button("Refresh Data"):
    st.rerun()

# Last updated time
st.sidebar.markdown("---")
st.sidebar.text(f"Last updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
