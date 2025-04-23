import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.data_generator import load_data
from utils.openai_helper import get_risk_insights, analyze_compliance_risk, generate_market_scenario

# Page config
st.set_page_config(page_title="AI Insights", page_icon="🧠", layout="wide")

st.title("AI-Powered Risk Insights")

# Load data
df_counterparties = load_data("counterparties")
df_transactions = load_data("transactions")
df_market_data = load_data("market_data")
df_compliance = load_data("compliance")

# Add a note about OpenAI API requirement
if not st.session_state.get("api_key_note_shown"):
    st.info(
        "Note: This page uses OpenAI API for generating insights. "
        "Set the OPENAI_API_KEY environment variable to enable all features. "
        "Sample insights will be shown if the API key is not configured."
    )
    st.session_state.api_key_note_shown = True

# Create tabs for different types of insights
insight_tabs = st.tabs([
    "Portfolio Insights", 
    "Counterparty Analysis", 
    "Market Intelligence", 
    "Compliance Insights"
])

# Portfolio Insights Tab
with insight_tabs[0]:
    st.subheader("Portfolio Risk Analysis")
    
    # Create summary of portfolio data
    total_exposure = df_counterparties['current_exposure'].sum()
    avg_risk = df_counterparties['risk_score'].mean()
    high_risk_count = len(df_counterparties[df_counterparties['risk_score'] > 0.7])
    
    portfolio_data = {
        "total_exposure": f"${total_exposure/1000000:.2f}M",
        "average_risk_score": f"{avg_risk:.2f}",
        "high_risk_counterparties": high_risk_count,
        "total_counterparties": len(df_counterparties),
        "exposure_by_sector": df_counterparties.groupby('sector')['current_exposure'].sum().to_dict(),
        "exposure_by_country": df_counterparties.groupby('country')['current_exposure'].sum().to_dict(),
        "risk_by_credit_rating": df_counterparties.groupby('credit_rating')['risk_score'].mean().to_dict()
    }
    
    # Get AI insights on portfolio
    with st.spinner("Analyzing portfolio data..."):
        portfolio_insights = get_risk_insights(portfolio_data, "Analyze the portfolio risk based on exposure, sector concentration, and counterparty risk scores.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### Portfolio Risk Summary")
        st.markdown(f"**Analysis:** {portfolio_insights.get('summary', 'Analysis not available')}")
        
        # Display risk level with appropriate color
        risk_level = portfolio_insights.get('risk_level', 'unknown')
        risk_color = {
            'low': 'green',
            'moderate': 'blue',
            'elevated': 'orange',
            'high': 'red',
            'severe': 'darkred',
            'unknown': 'gray'
        }.get(risk_level, 'gray')
        
        st.markdown(f"**Risk Level:** <span style='color:{risk_color};font-weight:bold;'>{risk_level.title()}</span>", unsafe_allow_html=True)
        
        st.markdown("### Key Risk Factors")
        for factor in portfolio_insights.get('key_factors', ['No key factors identified']):
            st.markdown(f"- {factor}")
        
        st.markdown("### Recommendations")
        for rec in portfolio_insights.get('recommendations', ['No recommendations available']):
            st.markdown(f"- {rec}")
    
    with col2:
        # Show a summary of portfolio metrics
        st.markdown("### Portfolio Metrics")
        
        metrics = [
            {"label": "Total Exposure", "value": f"${total_exposure/1000000:.2f}M"},
            {"label": "Avg Risk Score", "value": f"{avg_risk:.2f}"},
            {"label": "High Risk Entities", "value": f"{high_risk_count}/{len(df_counterparties)}"},
            {"label": "Non-Compliant", "value": f"{len(df_compliance[df_compliance['status'] == 'Non-Compliant'])}"}
        ]
        
        for metric in metrics:
            st.metric(metric["label"], metric["value"])
    
    # Top sectors by risk
    st.subheader("Sector Risk Analysis")
    
    # Calculate sector metrics
    sector_metrics = []
    for sector in df_counterparties['sector'].unique():
        sector_data = df_counterparties[df_counterparties['sector'] == sector]
        sector_metrics.append({
            "sector": sector,
            "total_exposure": sector_data['current_exposure'].sum(),
            "average_risk": sector_data['risk_score'].mean(),
            "counterparty_count": len(sector_data),
            "high_risk_count": len(sector_data[sector_data['risk_score'] > 0.7])
        })
    
    sector_df = pd.DataFrame(sector_metrics)
    sector_df = sector_df.sort_values('average_risk', ascending=False)
    
    # Create a scatter plot
    fig = px.scatter(
        sector_df,
        x='total_exposure',
        y='average_risk',
        size='counterparty_count',
        color='average_risk',
        hover_name='sector',
        color_continuous_scale='RdYlGn_r',
        size_max=40,
        title="Sector Risk Analysis"
    )
    
    fig.update_layout(
        xaxis_title="Total Exposure",
        yaxis_title="Average Risk Score",
        coloraxis_colorbar=dict(title="Risk Score"),
        height=500
    )
    
    # Format x-axis as currency
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    
    st.plotly_chart(fig, use_container_width=True)

# Counterparty Analysis Tab
with insight_tabs[1]:
    st.subheader("Counterparty Risk Assessment")
    
    # Counterparty selector
    counterparty_options = df_counterparties['name'].tolist()
    selected_counterparty = st.selectbox(
        "Select Counterparty for Analysis",
        options=counterparty_options
    )
    
    # Get counterparty data
    counterparty_data = df_counterparties[df_counterparties['name'] == selected_counterparty].iloc[0]
    counterparty_id = counterparty_data['id']
    
    # Get transactions
    cp_transactions = df_transactions[df_transactions['counterparty_id'] == counterparty_id]
    
    # Prepare data for AI analysis
    analysis_data = {
        "counterparty": counterparty_data.to_dict(),
        "transaction_count": len(cp_transactions),
        "average_transaction": cp_transactions['amount'].mean() if len(cp_transactions) > 0 else 0,
        "transaction_types": cp_transactions['transaction_type'].value_counts().to_dict() if len(cp_transactions) > 0 else {},
        "recent_activity": cp_transactions.sort_values('timestamp', ascending=False).head(5).to_dict('records') if len(cp_transactions) > 0 else [],
    }
    
    # Get AI insights
    with st.spinner(f"Analyzing {selected_counterparty}..."):
        cp_insights = get_risk_insights(
            analysis_data, 
            f"Analyze the risk profile of {selected_counterparty} based on their data, transaction history, and risk metrics."
        )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### Risk Assessment for {selected_counterparty}")
        st.markdown(f"**Analysis:** {cp_insights.get('summary', 'Analysis not available')}")
        
        # Display risk level with appropriate color
        risk_level = cp_insights.get('risk_level', 'unknown')
        risk_color = {
            'low': 'green',
            'moderate': 'blue',
            'elevated': 'orange',
            'high': 'red',
            'severe': 'darkred',
            'unknown': 'gray'
        }.get(risk_level, 'gray')
        
        st.markdown(f"**Risk Level:** <span style='color:{risk_color};font-weight:bold;'>{risk_level.title()}</span>", unsafe_allow_html=True)
        
        st.markdown("### Key Risk Factors")
        for factor in cp_insights.get('key_factors', ['No key factors identified']):
            st.markdown(f"- {factor}")
        
        st.markdown("### Recommendations")
        for rec in cp_insights.get('recommendations', ['No recommendations available']):
            st.markdown(f"- {rec}")
    
    with col2:
        # Counterparty details
        st.markdown("### Counterparty Details")
        st.markdown(f"**Sector:** {counterparty_data['sector']}")
        st.markdown(f"**Country:** {counterparty_data['country']}")
        st.markdown(f"**Credit Rating:** {counterparty_data['credit_rating']}")
        st.markdown(f"**Current Exposure:** ${counterparty_data['current_exposure']:,}")
        st.markdown(f"**Exposure Limit:** ${counterparty_data['exposure_limit']:,}")
        st.markdown(f"**Risk Score:** {counterparty_data['risk_score']:.2f}")
        st.markdown(f"**Last Review:** {counterparty_data['last_review_date']}")
    
    # Show transaction history if available
    if len(cp_transactions) > 0:
        st.subheader("Transaction Analysis")
        
        # Create time series of transaction amounts
        cp_transactions['timestamp'] = pd.to_datetime(cp_transactions['timestamp'])
        cp_transactions = cp_transactions.sort_values('timestamp')
        
        fig = px.scatter(
            cp_transactions,
            x='timestamp',
            y='amount',
            color='risk_score',
            size='amount',
            hover_data=['transaction_type', 'currency'],
            color_continuous_scale='RdYlGn_r',
            title=f"Transaction History for {selected_counterparty}"
        )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Amount",
            coloraxis_colorbar=dict(title="Risk Score"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Transaction type breakdown
        tx_types = cp_transactions['transaction_type'].value_counts().reset_index()
        tx_types.columns = ['Transaction Type', 'Count']
        
        fig = px.pie(
            tx_types,
            values='Count',
            names='Transaction Type',
            title=f"Transaction Types for {selected_counterparty}"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No transactions found for {selected_counterparty}")

# Market Intelligence Tab
with insight_tabs[2]:
    st.subheader("Market Intelligence")
    
    # Get most recent market data
    recent_date = df_market_data['date'].max()
    recent_market = df_market_data[df_market_data['date'] == recent_date]
    
    # Prepare data for AI analysis
    market_summary = {
        "date": recent_date,
        "average_volatility": recent_market['volatility'].mean(),
        "average_market_risk": recent_market['market_risk'].mean(),
        "average_liquidity": recent_market['liquidity'].mean(),
        "average_change": recent_market['change_pct'].mean(),
        "sector_risks": recent_market.groupby('sector')['market_risk'].mean().to_dict(),
        "region_risks": recent_market.groupby('region')['market_risk'].mean().to_dict(),
        "highest_volatility": {
            "sector": recent_market.loc[recent_market['volatility'].idxmax()]['sector'],
            "region": recent_market.loc[recent_market['volatility'].idxmax()]['region'],
            "value": recent_market['volatility'].max()
        },
        "lowest_liquidity": {
            "sector": recent_market.loc[recent_market['liquidity'].idxmin()]['sector'],
            "region": recent_market.loc[recent_market['liquidity'].idxmin()]['region'],
            "value": recent_market['liquidity'].min()
        }
    }
    
    # Get AI insights on market conditions
    with st.spinner("Analyzing market conditions..."):
        market_insights = get_risk_insights(
            market_summary, 
            "Analyze the current market conditions based on volatility, market risk, and liquidity metrics across sectors and regions."
        )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Market Conditions Analysis")
        st.markdown(f"**Analysis:** {market_insights.get('summary', 'Analysis not available')}")
        
        # Display risk level with appropriate color
        risk_level = market_insights.get('risk_level', 'unknown')
        risk_color = {
            'low': 'green',
            'moderate': 'blue',
            'elevated': 'orange',
            'high': 'red',
            'severe': 'darkred',
            'unknown': 'gray'
        }.get(risk_level, 'gray')
        
        st.markdown(f"**Market Risk Level:** <span style='color:{risk_color};font-weight:bold;'>{risk_level.title()}</span>", unsafe_allow_html=True)
        
        st.markdown("### Key Market Factors")
        for factor in market_insights.get('key_factors', ['No key factors identified']):
            st.markdown(f"- {factor}")
        
        st.markdown("### Market Outlook & Recommendations")
        for rec in market_insights.get('recommendations', ['No recommendations available']):
            st.markdown(f"- {rec}")
    
    with col2:
        # Market metrics
        st.markdown("### Market Metrics")
        
        metrics = [
            {"label": "Avg Volatility", "value": f"{market_summary['average_volatility']:.2f}"},
            {"label": "Avg Market Risk", "value": f"{market_summary['average_market_risk']:.2f}"},
            {"label": "Avg Liquidity", "value": f"{market_summary['average_liquidity']:.2f}"},
            {"label": "Avg Daily Change", "value": f"{market_summary['average_change']:.2f}%"}
        ]
        
        for metric in metrics:
            st.metric(metric["label"], metric["value"])
    
    # Generate market scenarios
    st.subheader("Market Scenario Generator")
    
    scenario_types = [
        "recession", 
        "inflation", 
        "market_crash", 
        "liquidity_crisis", 
        "interest_rate_hike",
        "commodity_shock",
        "currency_crisis",
        "tech_sector_correction"
    ]
    
    selected_scenario = st.selectbox(
        "Select Market Scenario Type",
        options=scenario_types
    )
    
    if st.button("Generate Market Scenario"):
        with st.spinner(f"Generating {selected_scenario} scenario..."):
            scenario = generate_market_scenario(selected_scenario)
        
        # Display scenario
        st.markdown(f"### {scenario.get('scenario_name', 'Market Scenario')}")
        st.markdown(f"**Description:** {scenario.get('description', 'No description available')}")
        
        # Display parameters
        st.markdown("### Scenario Parameters")
        
        parameters = scenario.get('parameters', {})
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
        else:
            st.info("No scenario parameters available")
        
        # Add Apply Scenario button
        if st.button("Apply This Scenario to Portfolio"):
            st.session_state.selected_scenario = scenario
            st.success("Scenario applied! Go to the Stress Testing page to view results.")

# Compliance Insights Tab
with insight_tabs[3]:
    st.subheader("Compliance Intelligence")
    
    # Prepare compliance data for analysis
    compliance_summary = {
        "total_entities": len(df_compliance),
        "compliance_status": df_compliance['status'].value_counts().to_dict(),
        "compliance_types": df_compliance['compliance_type'].value_counts().to_dict(),
        "average_risk": df_compliance['risk_score'].mean(),
        "high_risk_entities": len(df_compliance[df_compliance['risk_score'] > 0.7]),
        "non_compliant": len(df_compliance[df_compliance['status'] == 'Non-Compliant']),
        "under_investigation": len(df_compliance[df_compliance['status'] == 'Under Investigation']),
        "flagged_entities": df_compliance[df_compliance['flagged'] == True]['counterparty_name'].tolist()
    }
    
    # Get AI insights
    with st.spinner("Analyzing compliance data..."):
        compliance_insights = analyze_compliance_risk(
            compliance_summary,
            "overall compliance status"
        )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Compliance Status Analysis")
        st.markdown(f"**Analysis:** {compliance_insights.get('summary', 'Analysis not available')}")
        
        # Display compliance status with appropriate color
        compliance_status = compliance_insights.get('compliance_status', 'unknown')
        status_color = {
            'compliant': 'green',
            'minor_issues': 'blue',
            'significant_concerns': 'orange',
            'non_compliant': 'red',
            'critical_violations': 'darkred',
            'unknown': 'gray'
        }.get(compliance_status, 'gray')
        
        st.markdown(f"**Compliance Status:** <span style='color:{status_color};font-weight:bold;'>{compliance_status.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
        
        st.markdown("### Risk Factors")
        for factor in compliance_insights.get('risk_factors', ['No risk factors identified']):
            st.markdown(f"- {factor}")
        
        st.markdown("### Compliance Recommendations")
        for rec in compliance_insights.get('recommendations', ['No recommendations available']):
            st.markdown(f"- {rec}")
    
    with col2:
        # Compliance metrics
        st.markdown("### Compliance Metrics")
        
        # Get counts for different statuses
        compliant_count = compliance_summary['compliance_status'].get('Compliant', 0)
        non_compliant_count = compliance_summary['compliance_status'].get('Non-Compliant', 0)
        pending_count = compliance_summary['compliance_status'].get('Pending Review', 0)
        investigation_count = compliance_summary['compliance_status'].get('Under Investigation', 0)
        
        metrics = [
            {"label": "Compliant", "value": compliant_count},
            {"label": "Non-Compliant", "value": non_compliant_count},
            {"label": "Pending Review", "value": pending_count},
            {"label": "Under Investigation", "value": investigation_count}
        ]
        
        for metric in metrics:
            st.metric(metric["label"], metric["value"])
    
    # Show regulatory updates
    st.subheader("Simulated Regulatory Updates")
    
    regulatory_updates = [
        {
            "title": "AML Directive Update",
            "authority": "Financial Action Task Force (FATF)",
            "date": (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            "summary": "New requirements for enhanced due diligence for high-risk jurisdictions. Firms must update their risk assessment methodologies by Q3 2023.",
            "impact": "medium"
        },
        {
            "title": "KYC Verification Standards",
            "authority": "Financial Conduct Authority (FCA)",
            "date": (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d'),
            "summary": "Updated guidelines for remote customer verification procedures, including requirements for biometric verification for high-value accounts.",
            "impact": "high"
        },
        {
            "title": "Sanctions List Update",
            "authority": "Office of Foreign Assets Control (OFAC)",
            "date": (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'),
            "summary": "Addition of 12 entities and 7 individuals to the sanctions list related to recent geopolitical developments.",
            "impact": "medium"
        }
    ]
    
    for update in regulatory_updates:
        impact_color = {
            "low": "green",
            "medium": "orange",
            "high": "red"
        }.get(update["impact"], "gray")
        
        st.markdown(
            f"""
            <div style="padding: 10px; border-left: 4px solid {impact_color}; margin-bottom: 10px;">
                <strong>{update["title"]}</strong> ({update["date"]})<br>
                <em>{update["authority"]}</em><br>
                {update["summary"]}<br>
                <span style="color:{impact_color};font-weight:bold;">Impact: {update["impact"].title()}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Get AI analysis of the regulatory updates
    if st.button("Analyze Regulatory Impact"):
        with st.spinner("Analyzing regulatory impact..."):
            regulatory_analysis = get_risk_insights(
                {"updates": regulatory_updates, "portfolio": portfolio_data},
                "Analyze the impact of these regulatory updates on our compliance posture and portfolio."
            )
        
        # Display analysis
        st.markdown("### Regulatory Impact Analysis")
        st.markdown(f"**Analysis:** {regulatory_analysis.get('summary', 'Analysis not available')}")
        
        st.markdown("### Actionable Steps")
        for rec in regulatory_analysis.get('recommendations', ['No recommendations available']):
            st.markdown(f"- {rec}")

# Refresh data button
if st.sidebar.button("Refresh Data"):
    st.rerun()

# Last updated time
st.sidebar.markdown("---")
st.sidebar.text(f"Last updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
