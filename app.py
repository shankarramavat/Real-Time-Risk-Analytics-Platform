import streamlit as st
import os
import pandas as pd
from datetime import datetime
from utils.data_generator import generate_sample_data, load_data

# Page configuration
st.set_page_config(
    page_title="Risk Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state if not already done
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.last_updated = datetime.now()
    st.session_state.alert_count = 0
    # Generate initial data
    generate_sample_data()

# Display sidebar logo and navigation
st.sidebar.title("Risk Analytics Platform")
st.sidebar.info("Select a page from the sidebar to explore different aspects of risk analytics.")

# Last updated timestamp
st.sidebar.markdown("---")
st.sidebar.text(f"Last updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")

# Main page content
st.title("Real-Time Risk Analytics Platform")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Platform Overview")
    st.markdown("""
    Welcome to the Real-Time Risk Analytics Platform, a comprehensive solution for financial institutions to monitor:
    
    - **Counterparty Risk**: Track exposure to individual counterparties
    - **Market Exposure**: Analyze exposure across different markets and sectors
    - **Compliance**: Ensure adherence to AML, KYC, and sanctions regulations
    - **AI-Powered Insights**: Leverage advanced analytics for risk assessment
    
    Navigate through the different pages using the sidebar to explore various risk metrics and analytics.
    """)

with col2:
    # Quick stats
    st.subheader("Quick Stats")
    
    df_counterparties = load_data("counterparties")
    df_transactions = load_data("transactions")
    df_market_data = load_data("market_data")
    df_compliance = load_data("compliance")
    
    st.metric("Active Counterparties", len(df_counterparties))
    st.metric("Daily Transactions", len(df_transactions))
    st.metric("Markets Monitored", df_market_data['sector'].nunique())
    st.metric("Compliance Alerts", st.session_state.alert_count)

st.markdown("---")

# Features overview
st.header("Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Real-time Monitoring")
    st.markdown("""
    - Track risk metrics in real-time
    - Receive instant alerts for threshold breaches
    - Monitor counterparty exposures
    """)

with col2:
    st.subheader("Advanced Analytics")
    st.markdown("""
    - AI-powered risk assessment
    - Stress testing simulations
    - Predictive risk modeling
    """)

with col3:
    st.subheader("Compliance Tools")
    st.markdown("""
    - Automated AML/KYC checks
    - Sanctions screening
    - Regulatory reporting
    """)

# Call to action
st.markdown("---")
st.markdown("### Get Started")
st.markdown("Select a module from the sidebar to begin exploring the platform's capabilities.")

# Refresh data button
if st.button("Refresh Data"):
    st.session_state.last_updated = datetime.now()
    generate_sample_data()
    st.rerun()
