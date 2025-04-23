import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_generator import load_data
from utils.risk_calculations import calculate_counterparty_risk_score, calculate_var, calculate_expected_shortfall
from utils.visualization import plot_radar_chart, plot_counterparty_network, plot_time_series

# Page config
st.set_page_config(page_title="Counterparty Risk", page_icon="🔄", layout="wide")

st.title("Counterparty Risk Analysis")

# Load data
df_counterparties = load_data("counterparties")
df_transactions = load_data("transactions")
df_relationships = load_data("relationships")

# Sidebar filter
st.sidebar.header("Filters")

# Sector filter
all_sectors = df_counterparties['sector'].unique().tolist()
selected_sectors = st.sidebar.multiselect(
    "Filter by Sector",
    options=all_sectors,
    default=all_sectors
)

# Risk score filter
min_risk, max_risk = st.sidebar.slider(
    "Risk Score Range",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.1
)

# Apply filters
filtered_counterparties = df_counterparties[
    (df_counterparties['sector'].isin(selected_sectors)) &
    (df_counterparties['risk_score'] >= min_risk) &
    (df_counterparties['risk_score'] <= max_risk)
]

# Summary metrics
st.subheader("Counterparty Risk Overview")
col1, col2, col3, col4 = st.columns(4)

total_exposure = filtered_counterparties['current_exposure'].sum()
avg_risk_score = filtered_counterparties['risk_score'].mean() if len(filtered_counterparties) > 0 else 0
exposure_limit_pct = (filtered_counterparties['current_exposure'].sum() / filtered_counterparties['exposure_limit'].sum()) * 100 if len(filtered_counterparties) > 0 else 0
highest_risk = filtered_counterparties['risk_score'].max() if len(filtered_counterparties) > 0 else 0

with col1:
    st.metric("Total Exposure", f"${total_exposure/1000000:.2f}M")

with col2:
    st.metric("Average Risk Score", f"{avg_risk_score:.2f}")

with col3:
    st.metric("Exposure Limit Usage", f"{exposure_limit_pct:.1f}%")

with col4:
    st.metric("Highest Risk Score", f"{highest_risk:.2f}")

# Top counterparties by exposure
st.subheader("Top Counterparties by Exposure")

# Sort and get top 10 counterparties by exposure
top_counterparties = filtered_counterparties.sort_values('current_exposure', ascending=False).head(10)

# Create bar chart
fig = px.bar(
    top_counterparties,
    x='name',
    y='current_exposure',
    color='risk_score',
    color_continuous_scale='RdYlGn_r',
    hover_data=['sector', 'credit_rating'],
    labels={
        'name': 'Counterparty',
        'current_exposure': 'Current Exposure',
        'risk_score': 'Risk Score'
    },
    title="Top Counterparties by Exposure"
)

fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# Exposure by sector and credit rating
st.subheader("Exposure Analysis")
col1, col2 = st.columns(2)

with col1:
    # Exposure by sector
    sector_exposure = filtered_counterparties.groupby('sector')[['current_exposure', 'exposure_limit']].sum().reset_index()
    sector_exposure['utilization'] = sector_exposure['current_exposure'] / sector_exposure['exposure_limit'] * 100
    
    fig = px.bar(
        sector_exposure,
        x='sector',
        y=['current_exposure', 'exposure_limit'],
        barmode='group',
        labels={
            'sector': 'Sector',
            'value': 'Exposure',
            'variable': 'Type'
        },
        title="Exposure by Sector"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Exposure by credit rating
    rating_exposure = filtered_counterparties.groupby('credit_rating')['current_exposure'].sum().reset_index()
    
    # Sort by credit rating
    rating_order = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB']
    rating_exposure['rating_order'] = rating_exposure['credit_rating'].apply(lambda x: rating_order.index(x) if x in rating_order else 999)
    rating_exposure = rating_exposure.sort_values('rating_order')
    
    fig = px.pie(
        rating_exposure,
        values='current_exposure',
        names='credit_rating',
        title="Exposure Distribution by Credit Rating",
        hole=0.4
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Detailed counterparty analysis
st.subheader("Detailed Counterparty Analysis")

# Counterparty selector
selected_counterparty = st.selectbox(
    "Select Counterparty",
    options=filtered_counterparties['name'].tolist()
)

# Get selected counterparty data
counterparty_data = filtered_counterparties[filtered_counterparties['name'] == selected_counterparty].iloc[0]
counterparty_id = counterparty_data['id']

# Get transactions for this counterparty
counterparty_transactions = df_transactions[df_transactions['counterparty_id'] == counterparty_id]

# Display counterparty details
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(f"{selected_counterparty}")
    st.markdown(f"**Sector:** {counterparty_data['sector']}")
    st.markdown(f"**Country:** {counterparty_data['country']}")
    st.markdown(f"**Credit Rating:** {counterparty_data['credit_rating']}")
    st.markdown(f"**Risk Score:** {counterparty_data['risk_score']:.2f}")
    st.markdown(f"**Current Exposure:** ${counterparty_data['current_exposure']:,}")
    st.markdown(f"**Exposure Limit:** ${counterparty_data['exposure_limit']:,}")
    st.markdown(f"**Last Review Date:** {counterparty_data['last_review_date']}")

with col2:
    # Create radar chart for risk components
    risk_categories = ['Credit Risk', 'Market Risk', 'Liquidity Risk', 'Operational Risk', 'Concentration Risk']
    
    # Generate sample risk values (in a real app, these would come from a detailed risk model)
    np.random.seed(int(counterparty_id))  # Use counterparty ID as seed for reproducibility
    
    # Map credit ratings to numerical scores
    credit_rating_map = {
        'AAA': 0.1, 'AA+': 0.15, 'AA': 0.2, 'AA-': 0.25,
        'A+': 0.3, 'A': 0.35, 'A-': 0.4,
        'BBB+': 0.5, 'BBB': 0.55, 'BBB-': 0.6,
        'BB+': 0.7, 'BB': 0.75, 'BB-': 0.8,
        'B+': 0.85, 'B': 0.9, 'B-': 0.95, 'CCC': 1.0
    }
    
    # Get credit risk score based on rating
    credit_rating = counterparty_data['credit_rating']
    credit_risk = credit_rating_map.get(credit_rating, 0.5)  # Default to 0.5 if rating not found
    
    risk_values = [
        credit_risk,  # Credit risk based on rating
        max(0.1, min(1.0, counterparty_data['risk_score'] * 0.8 + np.random.uniform(-0.1, 0.1))),  # Market risk
        max(0.1, min(1.0, 0.5 + np.random.uniform(-0.2, 0.2))),  # Liquidity risk
        max(0.1, min(1.0, counterparty_data['risk_score'] * 0.6 + np.random.uniform(-0.1, 0.1))),  # Operational risk
        max(0.1, min(1.0, (counterparty_data['current_exposure'] / counterparty_data['exposure_limit']) * 0.8))  # Concentration risk
    ]
    
    # Plot radar chart
    fig = plot_radar_chart(risk_categories, risk_values, title="Risk Component Analysis")
    st.plotly_chart(fig, use_container_width=True)

# Transaction history
st.subheader(f"Transaction History for {selected_counterparty}")

if len(counterparty_transactions) > 0:
    # Display recent transactions
    recent_txns = counterparty_transactions.sort_values('timestamp', ascending=False).head(10)
    st.dataframe(recent_txns[['transaction_type', 'amount', 'currency', 'risk_score', 'timestamp']], use_container_width=True)
    
    # Transaction amount over time
    if len(counterparty_transactions) > 1:
        # Convert timestamp to datetime
        counterparty_transactions['timestamp'] = pd.to_datetime(counterparty_transactions['timestamp'])
        
        # Sort by timestamp
        counterparty_transactions = counterparty_transactions.sort_values('timestamp')
        
        # Plot transaction amounts over time
        fig = px.scatter(
            counterparty_transactions,
            x='timestamp',
            y='amount',
            color='risk_score',
            size='amount',
            color_continuous_scale='RdYlGn_r',
            title="Transaction Amounts Over Time",
            hover_data=['transaction_type', 'currency']
        )
        
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No transactions found for this counterparty.")

# VaR and ES calculations
st.subheader("Value at Risk (VaR) Analysis")

# Generate historical exposure data (in a real app, this would come from a database)
np.random.seed(42)
days = 250
base_exposure = counterparty_data['current_exposure']
historical_exposures = base_exposure * (1 + np.random.normal(0, 0.02, days).cumsum() * 0.1)
historical_exposures = np.abs(historical_exposures)  # Ensure positive exposures

# Calculate VaR and ES
confidence_levels = [0.95, 0.99]
time_horizons = [1, 10]

col1, col2 = st.columns(2)

with col1:
    for cl in confidence_levels:
        st.subheader(f"{cl*100:.0f}% Confidence Level")
        for th in time_horizons:
            var = calculate_var(historical_exposures, cl, th)
            es = calculate_expected_shortfall(historical_exposures, cl)
            
            st.markdown(f"**{th}-Day VaR:** ${var:,.2f}")
            st.markdown(f"**Expected Shortfall:** ${es:,.2f}")

with col2:
    # Plot historical exposures
    dates = [(pd.Timestamp.now() - pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    dates.reverse()  # Oldest first
    
    fig = plot_time_series(dates, historical_exposures, title="Historical Exposure", y_label="Exposure ($)")
    st.plotly_chart(fig, use_container_width=True)

# Counterparty network visualization
st.subheader("Counterparty Network Analysis")

# Prepare nodes (counterparties)
nodes_df = df_counterparties[['id', 'name', 'risk_score']].copy()

# Add random positions (in a real app, you'd use a layout algorithm like force-directed)
np.random.seed(42)
nodes_df['x'] = np.random.uniform(-1, 1, len(nodes_df))
nodes_df['y'] = np.random.uniform(-1, 1, len(nodes_df))

# Plot network
fig = plot_counterparty_network(nodes_df, df_relationships, highlight_node=counterparty_id)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Note:** This network visualization shows relationships between counterparties.
- Node size indicates exposure amount
- Node color indicates risk level (red = higher risk)
- Line thickness represents relationship strength
""")

# Refresh data button
if st.sidebar.button("Refresh Data"):
    st.rerun()

# Last updated time
st.sidebar.markdown("---")
st.sidebar.text(f"Last updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
