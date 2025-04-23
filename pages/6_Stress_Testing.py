import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.data_generator import load_data
from utils.risk_calculations import calculate_stress_test_impact, calculate_var, calculate_expected_shortfall
from utils.visualization import plot_stress_test_results
from utils.openai_helper import generate_market_scenario

# Page config
st.set_page_config(page_title="Stress Testing", page_icon="⚡", layout="wide")

st.title("Stress Testing & Scenario Analysis")

# Load data
df_counterparties = load_data("counterparties")
df_transactions = load_data("transactions")
df_market_data = load_data("market_data")

# Sidebar options
st.sidebar.header("Stress Test Configuration")

# Define stress test scenarios
standard_scenarios = {
    "Mild Recession": {
        "market_shock": -0.15,
        "interest_rate_change": 0.01,
        "credit_spread_widening": 0.025,
        "liquidity_reduction": 0.20,
        "default_rate_increase": 0.03,
        "gdp_impact": -0.02
    },
    "Severe Recession": {
        "market_shock": -0.30,
        "interest_rate_change": 0.02,
        "credit_spread_widening": 0.05,
        "liquidity_reduction": 0.40,
        "default_rate_increase": 0.08,
        "gdp_impact": -0.05
    },
    "Market Crash": {
        "market_shock": -0.40,
        "interest_rate_change": -0.005,
        "credit_spread_widening": 0.07,
        "liquidity_reduction": 0.60,
        "default_rate_increase": 0.07,
        "gdp_impact": -0.03
    },
    "Interest Rate Spike": {
        "market_shock": -0.10,
        "interest_rate_change": 0.03,
        "credit_spread_widening": 0.02,
        "liquidity_reduction": 0.15,
        "default_rate_increase": 0.02,
        "gdp_impact": -0.01
    },
    "Liquidity Crisis": {
        "market_shock": -0.20,
        "interest_rate_change": 0.015,
        "credit_spread_widening": 0.04,
        "liquidity_reduction": 0.70,
        "default_rate_increase": 0.05,
        "gdp_impact": -0.02
    },
    "Custom Scenario": {
        "market_shock": -0.10,
        "interest_rate_change": 0.01,
        "credit_spread_widening": 0.02,
        "liquidity_reduction": 0.15,
        "default_rate_increase": 0.02,
        "gdp_impact": -0.01
    }
}

# Check if there's an AI-generated scenario from the Insights page
if hasattr(st.session_state, 'selected_scenario') and st.session_state.selected_scenario:
    # Add the AI-generated scenario to the list
    scenario_name = st.session_state.selected_scenario.get('scenario_name', 'AI Scenario')
    standard_scenarios[scenario_name] = st.session_state.selected_scenario.get('parameters', {})
    
    # Default to the AI-generated scenario
    default_scenario = scenario_name
else:
    default_scenario = "Mild Recession"

# Scenario selection
selected_scenario = st.sidebar.selectbox(
    "Select Stress Test Scenario",
    options=list(standard_scenarios.keys()),
    index=list(standard_scenarios.keys()).index(default_scenario)
)

# If custom scenario, show sliders to adjust parameters
if selected_scenario == "Custom Scenario":
    st.sidebar.subheader("Custom Scenario Parameters")
    
    market_shock = st.sidebar.slider(
        "Market Shock",
        min_value=-0.5,
        max_value=0.1,
        value=standard_scenarios["Custom Scenario"]["market_shock"],
        step=0.01,
        format="%.2f"
    )
    
    interest_rate_change = st.sidebar.slider(
        "Interest Rate Change (pp)",
        min_value=-0.02,
        max_value=0.05,
        value=standard_scenarios["Custom Scenario"]["interest_rate_change"],
        step=0.005,
        format="%.3f"
    )
    
    credit_spread_widening = st.sidebar.slider(
        "Credit Spread Widening (pp)",
        min_value=0.0,
        max_value=0.1,
        value=standard_scenarios["Custom Scenario"]["credit_spread_widening"],
        step=0.005,
        format="%.3f"
    )
    
    liquidity_reduction = st.sidebar.slider(
        "Liquidity Reduction",
        min_value=0.0,
        max_value=0.8,
        value=standard_scenarios["Custom Scenario"]["liquidity_reduction"],
        step=0.05,
        format="%.2f"
    )
    
    default_rate_increase = st.sidebar.slider(
        "Default Rate Increase (pp)",
        min_value=0.0,
        max_value=0.15,
        value=standard_scenarios["Custom Scenario"]["default_rate_increase"],
        step=0.01,
        format="%.2f"
    )
    
    gdp_impact = st.sidebar.slider(
        "GDP Impact",
        min_value=-0.1,
        max_value=0.02,
        value=standard_scenarios["Custom Scenario"]["gdp_impact"],
        step=0.01,
        format="%.2f"
    )
    
    # Update custom scenario
    standard_scenarios["Custom Scenario"] = {
        "market_shock": market_shock,
        "interest_rate_change": interest_rate_change,
        "credit_spread_widening": credit_spread_widening,
        "liquidity_reduction": liquidity_reduction,
        "default_rate_increase": default_rate_increase,
        "gdp_impact": gdp_impact
    }

# Get the selected scenario parameters
scenario_params = standard_scenarios[selected_scenario]

# Display stress test description
st.subheader(f"Scenario: {selected_scenario}")

# Format the parameters for display
formatted_params = {}
for key, value in scenario_params.items():
    if key in ["market_shock", "liquidity_reduction", "gdp_impact"]:
        formatted_params[key] = f"{value:.1%}"
    else:
        formatted_params[key] = f"{value:.2%}"

# Display parameters in multi-columns
param_cols = st.columns(3)
for i, (key, value) in enumerate(formatted_params.items()):
    col_idx = i % 3
    with param_cols[col_idx]:
        label = " ".join(word.capitalize() for word in key.split("_"))
        st.metric(label, value)

# Run the stress test
st.subheader("Running Stress Test...")

# Create tabs for different analyses
stress_tabs = st.tabs([
    "Portfolio Impact", 
    "Counterparty Stress", 
    "VaR Analysis", 
    "Market Impact"
])

# Portfolio Impact tab
with stress_tabs[0]:
    st.markdown("### Portfolio-Wide Impact")
    
    # Calculate stress impact on total exposure
    total_exposure = df_counterparties['current_exposure'].sum()
    stressed_exposure = total_exposure * (1 + scenario_params['market_shock'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Total Exposure",
            f"${total_exposure/1000000:.2f}M",
            delta=f"{scenario_params['market_shock']:.1%}",
            delta_color="inverse"
        )
        
        st.metric(
            "Stressed Exposure",
            f"${stressed_exposure/1000000:.2f}M"
        )
    
    with col2:
        # Calculate impact on portfolio metrics
        avg_risk_score = df_counterparties['risk_score'].mean()
        
        # Higher risk after stress test
        stressed_risk = min(1.0, avg_risk_score * (1 - scenario_params['market_shock'] * 0.5))
        
        st.metric(
            "Average Risk Score",
            f"{avg_risk_score:.2f}",
            delta=f"{(stressed_risk - avg_risk_score):.2f}",
            delta_color="inverse"
        )
        
        st.metric(
            "Stressed Risk Score",
            f"{stressed_risk:.2f}"
        )
    
    # Impact by sector
    st.markdown("### Impact by Sector")
    
    # Calculate sector exposures
    sector_exposure = df_counterparties.groupby('sector')['current_exposure'].sum().reset_index()
    
    # Apply sector-specific stress factors (some sectors might be more affected)
    sector_stress_factors = {
        'Banking': scenario_params['market_shock'] * 1.2,
        'Insurance': scenario_params['market_shock'] * 1.1,
        'Asset Management': scenario_params['market_shock'] * 1.3,
        'Hedge Fund': scenario_params['market_shock'] * 1.5,
        'Pension Fund': scenario_params['market_shock'] * 0.8,
        'Corporate': scenario_params['market_shock'] * 1.0,
        'Government': scenario_params['market_shock'] * 0.5
    }
    
    # Apply stress to each sector
    sector_exposure['stressed_exposure'] = sector_exposure.apply(
        lambda row: row['current_exposure'] * (1 + sector_stress_factors.get(row['sector'], scenario_params['market_shock'])),
        axis=1
    )
    
    # Calculate impact
    sector_exposure['impact'] = sector_exposure['stressed_exposure'] - sector_exposure['current_exposure']
    sector_exposure['impact_pct'] = sector_exposure['impact'] / sector_exposure['current_exposure']
    
    # Sort by impact percentage
    sector_exposure = sector_exposure.sort_values('impact_pct')
    
    # Plot sector impact
    fig = px.bar(
        sector_exposure,
        x='sector',
        y='impact_pct',
        color='impact_pct',
        color_continuous_scale='RdYlGn',
        labels={
            'sector': 'Sector',
            'impact_pct': 'Impact (%)'
        },
        title="Stress Impact by Sector"
    )
    
    # Format y-axis as percentage
    fig.update_layout(
        yaxis=dict(tickformat='.1%'),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show sector data
    sector_exposure['current_exposure_m'] = sector_exposure['current_exposure'] / 1000000
    sector_exposure['stressed_exposure_m'] = sector_exposure['stressed_exposure'] / 1000000
    sector_exposure['impact_m'] = sector_exposure['impact'] / 1000000
    
    display_df = sector_exposure[['sector', 'current_exposure_m', 'stressed_exposure_m', 'impact_m', 'impact_pct']]
    display_df.columns = ['Sector', 'Current Exposure ($M)', 'Stressed Exposure ($M)', 'Impact ($M)', 'Impact (%)']
    
    # Format for display
    pd.options.display.float_format = '{:,.2f}'.format
    st.dataframe(display_df, use_container_width=True)

# Counterparty Stress tab
with stress_tabs[1]:
    st.markdown("### Counterparty Stress Analysis")
    
    # Apply stress factors to each counterparty
    cp_stress = df_counterparties.copy()
    
    # Apply credit rating-specific stress factors (lower ratings are more affected)
    rating_stress_factors = {
        'AAA': scenario_params['market_shock'] * 0.5,
        'AA+': scenario_params['market_shock'] * 0.6,
        'AA': scenario_params['market_shock'] * 0.7,
        'AA-': scenario_params['market_shock'] * 0.8,
        'A+': scenario_params['market_shock'] * 0.9,
        'A': scenario_params['market_shock'] * 1.0,
        'A-': scenario_params['market_shock'] * 1.1,
        'BBB+': scenario_params['market_shock'] * 1.2,
        'BBB': scenario_params['market_shock'] * 1.3,
        'BBB-': scenario_params['market_shock'] * 1.4,
        'BB+': scenario_params['market_shock'] * 1.5,
        'BB': scenario_params['market_shock'] * 1.6
    }
    
    # Apply sector stress factors from previous tab
    cp_stress['stressed_exposure'] = cp_stress.apply(
        lambda row: row['current_exposure'] * (
            1 + rating_stress_factors.get(row['credit_rating'], scenario_params['market_shock']) *
            (1 + sector_stress_factors.get(row['sector'], 0) / scenario_params['market_shock'])
        ),
        axis=1
    )
    
    # Calculate new risk scores (higher after stress)
    cp_stress['stressed_risk_score'] = cp_stress.apply(
        lambda row: min(1.0, row['risk_score'] * (
            1 - min(0, rating_stress_factors.get(row['credit_rating'], scenario_params['market_shock'])) * 0.5
        )),
        axis=1
    )
    
    # Identify counterparties that exceed their exposure limit under stress
    cp_stress['exceeds_limit'] = cp_stress['stressed_exposure'] > cp_stress['exposure_limit']
    cp_stress['limit_headroom'] = cp_stress['exposure_limit'] - cp_stress['stressed_exposure']
    
    # Sort by limit headroom (negative values indicate exceedance)
    cp_stress = cp_stress.sort_values('limit_headroom')
    
    # Show top counterparties at risk
    st.markdown("### Top 10 Counterparties Exceeding Limits Under Stress")
    
    at_risk_cps = cp_stress[cp_stress['exceeds_limit']].head(10)
    
    if len(at_risk_cps) > 0:
        # Calculate exceedance amount and percentage
        at_risk_cps['exceedance'] = at_risk_cps['stressed_exposure'] - at_risk_cps['exposure_limit']
        at_risk_cps['exceedance_pct'] = at_risk_cps['exceedance'] / at_risk_cps['exposure_limit'] * 100
        
        # Create visualization
        fig = px.bar(
            at_risk_cps,
            x='name',
            y='exceedance',
            color='exceedance_pct',
            color_continuous_scale='Reds',
            hover_data=['sector', 'credit_rating', 'current_exposure', 'stressed_exposure', 'exposure_limit'],
            labels={
                'name': 'Counterparty',
                'exceedance': 'Limit Exceedance',
                'exceedance_pct': 'Exceedance (%)'
            },
            title="Counterparties Exceeding Limits Under Stress"
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data
        display_cols = ['name', 'sector', 'credit_rating', 'current_exposure', 
                          'stressed_exposure', 'exposure_limit', 'exceedance', 'exceedance_pct']
        display_df = at_risk_cps[display_cols].copy()
        display_df.columns = ['Name', 'Sector', 'Credit Rating', 'Current Exposure', 
                              'Stressed Exposure', 'Exposure Limit', 'Exceedance', 'Exceedance (%)']
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.success("No counterparties exceed their limits under this stress scenario.")
    
    # Show risk score changes
    st.markdown("### Counterparty Risk Score Changes")
    
    # Calculate risk score change
    cp_stress['risk_change'] = cp_stress['stressed_risk_score'] - cp_stress['risk_score']
    cp_stress['risk_change_pct'] = cp_stress['risk_change'] / cp_stress['risk_score'] * 100
    
    # Sort by risk change
    cp_stress = cp_stress.sort_values('risk_change', ascending=False)
    
    # Plot top 15 by risk increase
    top_risk_increase = cp_stress.head(15)
    
    fig = px.bar(
        top_risk_increase,
        x='name',
        y=['risk_score', 'stressed_risk_score'],
        barmode='group',
        labels={
            'name': 'Counterparty',
            'value': 'Risk Score',
            'variable': 'Scenario'
        },
        title="Top 15 Counterparties by Risk Score Increase",
        color_discrete_map={
            'risk_score': 'blue',
            'stressed_risk_score': 'red'
        }
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# VaR Analysis tab
with stress_tabs[2]:
    st.markdown("### Value at Risk (VaR) Analysis Under Stress")
    
    # Generate historical exposure data (in a real app, this would come from a database)
    np.random.seed(42)
    days = 250
    base_exposure = df_counterparties['current_exposure'].sum()
    historical_exposures = base_exposure * (1 + np.random.normal(0, 0.02, days).cumsum() * 0.1)
    historical_exposures = np.abs(historical_exposures)  # Ensure positive exposures
    
    # Apply stress to historical data
    stressed_exposures = calculate_stress_test_impact(historical_exposures, scenario_params['market_shock'])
    
    # Calculate VaR and ES for both base and stressed scenarios
    confidence_levels = [0.95, 0.99]
    time_horizons = [1, 10]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Base Case")
        for cl in confidence_levels:
            st.markdown(f"##### {cl*100:.0f}% Confidence Level")
            for th in time_horizons:
                var = calculate_var(historical_exposures, cl, th)
                es = calculate_expected_shortfall(historical_exposures, cl)
                
                st.markdown(f"**{th}-Day VaR:** ${var/1000000:,.2f}M")
                st.markdown(f"**Expected Shortfall:** ${es/1000000:,.2f}M")
    
    with col2:
        st.markdown("#### Stressed Case")
        for cl in confidence_levels:
            st.markdown(f"##### {cl*100:.0f}% Confidence Level")
            for th in time_horizons:
                stressed_var = calculate_var(stressed_exposures, cl, th)
                stressed_es = calculate_expected_shortfall(stressed_exposures, cl)
                
                var_change = (stressed_var / calculate_var(historical_exposures, cl, th) - 1) * 100
                es_change = (stressed_es / calculate_expected_shortfall(historical_exposures, cl) - 1) * 100
                
                st.markdown(f"**{th}-Day VaR:** ${stressed_var/1000000:,.2f}M ({var_change:+.1f}%)")
                st.markdown(f"**Expected Shortfall:** ${stressed_es/1000000:,.2f}M ({es_change:+.1f}%)")
    
    # Plot distribution
    st.markdown("### Exposure Distribution: Base vs. Stressed")
    
    # Create a dataframe for plotting
    hist_df = pd.DataFrame({
        'Date': pd.date_range(end=pd.Timestamp.now(), periods=days),
        'Base Exposure': historical_exposures,
        'Stressed Exposure': stressed_exposures
    })
    
    # Plot historical and stressed data
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(
            x=hist_df['Base Exposure'] / 1000000,
            name='Base Case',
            opacity=0.7,
            marker_color='blue',
            nbinsx=30
        )
    )
    
    fig.add_trace(
        go.Histogram(
            x=hist_df['Stressed Exposure'] / 1000000,
            name='Stressed Case',
            opacity=0.7,
            marker_color='red',
            nbinsx=30
        )
    )
    
    fig.update_layout(
        title="Exposure Distribution Under Stress",
        xaxis_title="Exposure ($M)",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot time series
    st.markdown("### Exposure Over Time: Base vs. Stressed")
    
    fig = px.line(
        hist_df,
        x='Date',
        y=['Base Exposure', 'Stressed Exposure'],
        labels={
            'value': 'Exposure',
            'variable': 'Scenario'
        },
        title="Exposure Trajectory Under Stress Scenario"
    )
    
    # Format y-axis values as millions
    fig.update_layout(
        yaxis=dict(tickformat='$,.0f'),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Market Impact tab
with stress_tabs[3]:
    st.markdown("### Market Impact Analysis")
    
    # Apply stress to market data
    market_stress = df_market_data.copy()
    
    # Get the most recent date
    recent_date = market_stress['date'].max()
    recent_data = market_stress[market_stress['date'] == recent_date]
    
    # Apply sector-specific stress
    sector_volatility_impact = {
        'Banking': 1.5,
        'Insurance': 1.4,
        'Technology': 1.6,
        'Healthcare': 1.2,
        'Energy': 1.8,
        'Utilities': 1.1,
        'Consumer Goods': 1.3,
        'Telecommunications': 1.4
    }
    
    # Apply region-specific stress
    region_volatility_impact = {
        'North America': 1.3,
        'Europe': 1.4,
        'Asia': 1.5,
        'Latin America': 1.7,
        'Middle East': 1.6,
        'Africa': 1.8,
        'Oceania': 1.2
    }
    
    # Create stressed market data
    stressed_market = recent_data.copy()
    stressed_market['stressed_volatility'] = stressed_market.apply(
        lambda row: min(1.0, row['volatility'] * sector_volatility_impact.get(row['sector'], 1.5) * 
                                           region_volatility_impact.get(row['region'], 1.5)),
        axis=1
    )
    
    stressed_market['stressed_market_risk'] = stressed_market.apply(
        lambda row: min(1.0, row['market_risk'] * sector_volatility_impact.get(row['sector'], 1.5)),
        axis=1
    )
    
    stressed_market['stressed_liquidity'] = stressed_market.apply(
        lambda row: max(0.1, row['liquidity'] * (1 - scenario_params['liquidity_reduction'] * 1.2)),
        axis=1
    )
    
    stressed_market['stressed_index_value'] = stressed_market['index_value'] * (1 + scenario_params['market_shock'])
    
    # Market metrics comparison
    st.markdown("### Market Metrics: Base vs. Stressed")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_vol = recent_data['volatility'].mean()
        stressed_vol = stressed_market['stressed_volatility'].mean()
        vol_change = (stressed_vol / avg_vol - 1) * 100
        
        st.metric(
            "Volatility",
            f"{avg_vol:.2f}",
            delta=f"{vol_change:+.1f}%",
            delta_color="inverse"
        )
    
    with col2:
        avg_risk = recent_data['market_risk'].mean()
        stressed_risk = stressed_market['stressed_market_risk'].mean()
        risk_change = (stressed_risk / avg_risk - 1) * 100
        
        st.metric(
            "Market Risk",
            f"{avg_risk:.2f}",
            delta=f"{risk_change:+.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        avg_liq = recent_data['liquidity'].mean()
        stressed_liq = stressed_market['stressed_liquidity'].mean()
        liq_change = (stressed_liq / avg_liq - 1) * 100
        
        st.metric(
            "Liquidity",
            f"{avg_liq:.2f}",
            delta=f"{liq_change:+.1f}%"
        )
    
    # Heatmap of stressed market risk
    st.markdown("### Market Risk Heatmap Under Stress")
    
    # Pivot table for heatmap
    base_heatmap = recent_data.pivot_table(
        index='sector',
        columns='region',
        values='market_risk',
        aggfunc='mean'
    )
    
    stressed_heatmap = stressed_market.pivot_table(
        index='sector',
        columns='region',
        values='stressed_market_risk',
        aggfunc='mean'
    )
    
    # Create tabs for base and stressed heatmaps
    hm_tabs = st.tabs(["Base Case", "Stressed Case"])
    
    with hm_tabs[0]:
        fig = px.imshow(
            base_heatmap,
            color_continuous_scale='RdYlGn_r',
            title="Base Market Risk by Sector and Region",
            labels=dict(x="Region", y="Sector", color="Risk Score")
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with hm_tabs[1]:
        fig = px.imshow(
            stressed_heatmap,
            color_continuous_scale='RdYlGn_r',
            title="Stressed Market Risk by Sector and Region",
            labels=dict(x="Region", y="Sector", color="Risk Score")
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Impact by sector
    st.markdown("### Sector Impact Analysis")
    
    # Calculate sector averages
    sector_impact = pd.DataFrame({
        'Sector': stressed_market['sector'].unique()
    })
    
    # Calculate base metrics by sector
    sector_vol_base = recent_data.groupby('sector')['volatility'].mean()
    sector_risk_base = recent_data.groupby('sector')['market_risk'].mean()
    sector_liq_base = recent_data.groupby('sector')['liquidity'].mean()
    sector_idx_base = recent_data.groupby('sector')['index_value'].mean()
    
    # Calculate stressed metrics by sector
    sector_vol_stress = stressed_market.groupby('sector')['stressed_volatility'].mean()
    sector_risk_stress = stressed_market.groupby('sector')['stressed_market_risk'].mean()
    sector_liq_stress = stressed_market.groupby('sector')['stressed_liquidity'].mean()
    sector_idx_stress = stressed_market.groupby('sector')['stressed_index_value'].mean()
    
    # Calculate impact percentages
    sector_impact['volatility_change'] = [(sector_vol_stress[s] / sector_vol_base[s] - 1) * 100 for s in sector_impact['Sector']]
    sector_impact['risk_change'] = [(sector_risk_stress[s] / sector_risk_base[s] - 1) * 100 for s in sector_impact['Sector']]
    sector_impact['liquidity_change'] = [(sector_liq_stress[s] / sector_liq_base[s] - 1) * 100 for s in sector_impact['Sector']]
    sector_impact['index_change'] = [(sector_idx_stress[s] / sector_idx_base[s] - 1) * 100 for s in sector_impact['Sector']]
    
    # Sort by overall impact
    sector_impact['overall_impact'] = (
        abs(sector_impact['volatility_change']) + 
        abs(sector_impact['risk_change']) + 
        abs(sector_impact['liquidity_change']) + 
        abs(sector_impact['index_change'])
    )
    sector_impact = sector_impact.sort_values('overall_impact', ascending=False)
    
    # Create visual comparison
    fig = px.bar(
        sector_impact,
        x='Sector',
        y=['volatility_change', 'risk_change', 'liquidity_change', 'index_change'],
        barmode='group',
        labels={
            'value': 'Change (%)',
            'variable': 'Metric'
        },
        title="Sector Impact Analysis",
        color_discrete_map={
            'volatility_change': 'red',
            'risk_change': 'orange',
            'liquidity_change': 'blue',
            'index_change': 'green'
        }
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Stress test report
st.subheader("Stress Test Summary")

# Calculate overall impact metrics
total_impact = total_exposure - stressed_exposure
impact_pct = (stressed_exposure / total_exposure - 1) * 100

# Count counterparties exceeding limits
exceeding_limits = len(cp_stress[cp_stress['exceeds_limit']])
exceeding_pct = exceeding_limits / len(cp_stress) * 100

# Capital impact (simulated)
capital_ratio_base = 0.12  # 12% base capital ratio
capital_impact = min(0, scenario_params['market_shock']) * 0.8  # Capital impact is 80% of market shock (if negative)
stressed_capital_ratio = capital_ratio_base * (1 + capital_impact)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Portfolio Impact",
        f"${abs(total_impact)/1000000:.2f}M",
        delta=f"{impact_pct:.1f}%",
        delta_color="inverse"
    )

with col2:
    st.metric(
        "Counterparties Exceeding Limits",
        f"{exceeding_limits}",
        delta=f"{exceeding_pct:.1f}%",
        delta_color="inverse"
    )

with col3:
    st.metric(
        "Capital Ratio",
        f"{capital_ratio_base:.1%}",
        delta=f"{capital_impact:.1%}",
        delta_color="inverse"
    )

# Generate AI scenario button
st.markdown("### Generate Custom Scenario with AI")

if st.button("Generate New AI Scenario"):
    scenario_types = [
        "recession", 
        "inflation", 
        "market_crash", 
        "liquidity_crisis", 
        "interest_rate_hike",
        "geopolitical_crisis",
        "commodity_shock"
    ]
    
    scenario_type = np.random.choice(scenario_types)
    
    with st.spinner(f"Generating {scenario_type} scenario..."):
        ai_scenario = generate_market_scenario(scenario_type)
    
    # Display scenario
    st.markdown(f"### {ai_scenario.get('scenario_name', 'Market Scenario')}")
    st.markdown(f"**Description:** {ai_scenario.get('description', 'No description available')}")
    
    # Display parameters
    st.markdown("### Scenario Parameters")
    
    parameters = ai_scenario.get('parameters', {})
    if parameters:
        params_df = pd.DataFrame({
            'Parameter': parameters.keys(),
            'Value': parameters.values()
        })
        
        # Format values as percentages where appropriate
        params_df['Formatted Value'] = params_df.apply(
            lambda row: f"{row['Value']:.2%}" if isinstance(row['Value'], (int, float)) and -1 <= row['Value'] <= 1 else row['Value'],
            axis=1
        )
        
        st.table(params_df[['Parameter', 'Formatted Value']])
    
    # Save to session state for future use
    st.session_state.selected_scenario = ai_scenario
    
    if st.button("Run Stress Test with This Scenario"):
        st.rerun()

# Refresh data button
if st.sidebar.button("Refresh Data"):
    st.rerun()

# Last updated time
st.sidebar.markdown("---")
st.sidebar.text(f"Last updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
