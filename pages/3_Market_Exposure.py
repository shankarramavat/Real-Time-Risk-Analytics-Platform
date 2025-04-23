import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.data_generator import load_data
from utils.risk_calculations import calculate_portfolio_concentration, generate_risk_time_series
from utils.visualization import plot_time_series
from utils.session_helper import initialize_session_state

# Page config
st.set_page_config(page_title="Market Exposure", page_icon="📈", layout="wide")

# Initialize session state
initialize_session_state()

st.title("Market Exposure Analysis")

# Load data
df_market_data = load_data("market_data")
df_counterparties = load_data("counterparties")

# Sidebar filters
st.sidebar.header("Filters")

# Region filter
all_regions = df_market_data['region'].unique().tolist()
selected_regions = st.sidebar.multiselect(
    "Filter by Region",
    options=all_regions,
    default=all_regions
)

# Sector filter
all_sectors = df_market_data['sector'].unique().tolist()
selected_sectors = st.sidebar.multiselect(
    "Filter by Sector",
    options=all_sectors,
    default=all_sectors
)

# Apply filters
filtered_market_data = df_market_data[
    (df_market_data['region'].isin(selected_regions)) &
    (df_market_data['sector'].isin(selected_sectors))
]

# Market risk overview
st.subheader("Market Risk Overview")

# Get the most recent date in the dataset
most_recent_date = filtered_market_data['date'].max()

# Filter for most recent data
recent_market_data = filtered_market_data[filtered_market_data['date'] == most_recent_date]

# Market risk metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_volatility = recent_market_data['volatility'].mean()
    st.metric("Average Volatility", f"{avg_volatility:.2f}")

with col2:
    avg_market_risk = recent_market_data['market_risk'].mean()
    st.metric("Market Risk Index", f"{avg_market_risk:.2f}")

with col3:
    avg_liquidity = recent_market_data['liquidity'].mean()
    st.metric("Liquidity Index", f"{avg_liquidity:.2f}")

with col4:
    avg_change = recent_market_data['change_pct'].mean()
    st.metric("Average Daily Change", f"{avg_change:.2f}%")

# Heatmap of market risk by sector and region
st.subheader("Market Risk Heatmap")

# Pivot table for heatmap
heatmap_data = recent_market_data.pivot_table(
    index='sector',
    columns='region',
    values='market_risk',
    aggfunc='mean'
)

# Create heatmap
fig = px.imshow(
    heatmap_data,
    color_continuous_scale='RdYlGn_r',  # Red (high risk) to Green (low risk)
    title="Market Risk by Sector and Region",
    labels=dict(x="Region", y="Sector", color="Risk Score")
)

fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Market trends over time
st.subheader("Market Trends Over Time")

# Select sector and region for time series
col1, col2 = st.columns(2)

with col1:
    ts_sector = st.selectbox("Select Sector for Trend Analysis", options=selected_sectors)

with col2:
    ts_region = st.selectbox("Select Region for Trend Analysis", options=selected_regions)

# Filter data for selected sector and region
ts_data = filtered_market_data[
    (filtered_market_data['sector'] == ts_sector) &
    (filtered_market_data['region'] == ts_region)
]

# Sort by date
ts_data = ts_data.sort_values('date')

# Create tabs for different metrics
ts_tabs = st.tabs(["Market Risk", "Volatility", "Liquidity", "Index Value"])

with ts_tabs[0]:
    # Plot market risk over time
    fig = plot_time_series(
        ts_data['date'].tolist(),
        ts_data['market_risk'].tolist(),
        title=f"Market Risk Over Time - {ts_sector} in {ts_region}",
        y_label="Market Risk"
    )
    st.plotly_chart(fig, use_container_width=True)

with ts_tabs[1]:
    # Plot volatility over time
    fig = plot_time_series(
        ts_data['date'].tolist(),
        ts_data['volatility'].tolist(),
        title=f"Volatility Over Time - {ts_sector} in {ts_region}",
        y_label="Volatility"
    )
    st.plotly_chart(fig, use_container_width=True)

with ts_tabs[2]:
    # Plot liquidity over time
    fig = plot_time_series(
        ts_data['date'].tolist(),
        ts_data['liquidity'].tolist(),
        title=f"Liquidity Over Time - {ts_sector} in {ts_region}",
        y_label="Liquidity"
    )
    st.plotly_chart(fig, use_container_width=True)

with ts_tabs[3]:
    # Plot index value over time
    fig = plot_time_series(
        ts_data['date'].tolist(),
        ts_data['index_value'].tolist(),
        title=f"Index Value Over Time - {ts_sector} in {ts_region}",
        y_label="Index Value"
    )
    st.plotly_chart(fig, use_container_width=True)

# Portfolio concentration analysis
st.subheader("Portfolio Concentration Analysis")

# Calculate exposure by sector
sector_exposure = df_counterparties.groupby('sector')['current_exposure'].sum().to_dict()

# Calculate HHI
hhi = calculate_portfolio_concentration(sector_exposure)

col1, col2 = st.columns(2)

with col1:
    # Display HHI
    st.metric("Herfindahl-Hirschman Index (HHI)", f"{hhi:.4f}")
    
    if hhi < 0.1:
        st.success("Low concentration risk")
    elif hhi < 0.18:
        st.warning("Moderate concentration risk")
    else:
        st.error("High concentration risk")
    
    st.markdown("""
    **HHI Interpretation:**
    - < 0.1: Low concentration
    - 0.1 - 0.18: Moderate concentration
    - > 0.18: High concentration
    """)

with col2:
    # Create pie chart of exposure by sector
    sector_exposure_df = pd.DataFrame({
        'Sector': sector_exposure.keys(),
        'Exposure': sector_exposure.values()
    })
    
    fig = px.pie(
        sector_exposure_df,
        values='Exposure',
        names='Sector',
        title="Exposure Distribution by Sector"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Correlation matrix
st.subheader("Market Correlation Matrix")

# Calculate correlation between sectors based on their index values
pivot_sectors = filtered_market_data.pivot_table(
    index='date',
    columns='sector',
    values='index_value'
)

# Calculate correlation matrix
corr_matrix = pivot_sectors.corr()

# Create heatmap
fig = px.imshow(
    corr_matrix,
    color_continuous_scale='RdBu_r',  # Blue (negative correlation) to Red (positive correlation)
    title="Correlation Between Sectors",
    labels=dict(x="Sector", y="Sector", color="Correlation")
)

fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

# Market outliers
st.subheader("Market Outliers")

# Identify outliers (values more than 2 standard deviations from the mean)
sectors_means = recent_market_data.groupby('sector')['market_risk'].mean()
sectors_stds = recent_market_data.groupby('sector')['market_risk'].std()

outliers = []
for sector in selected_sectors:
    sector_data = recent_market_data[recent_market_data['sector'] == sector]
    
    for _, row in sector_data.iterrows():
        mean = sectors_means[sector]
        std = max(0.01, sectors_stds[sector])  # Avoid division by zero
        
        z_score = (row['market_risk'] - mean) / std
        
        if abs(z_score) > 2:
            outliers.append({
                'Sector': sector,
                'Region': row['region'],
                'Market Risk': row['market_risk'],
                'Z-Score': z_score,
                'Volatility': row['volatility'],
                'Liquidity': row['liquidity'],
                'Change %': row['change_pct']
            })

if outliers:
    outliers_df = pd.DataFrame(outliers)
    outliers_df = outliers_df.sort_values('Z-Score', ascending=False)
    
    # Display outliers
    st.dataframe(outliers_df, use_container_width=True)
else:
    st.info("No significant outliers found in the current market data.")

# Refresh data button
if st.sidebar.button("Refresh Data"):
    st.rerun()

# Last updated time
st.sidebar.markdown("---")
st.sidebar.text(f"Last updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
