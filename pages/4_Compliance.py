import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.data_generator import load_data
from utils.compliance import validate_kyc_compliance, validate_aml_compliance, check_sanctions, generate_compliance_report
from utils.visualization import plot_compliance_status
from utils.openai_helper import analyze_compliance_risk
from utils.session_helper import initialize_session_state

# Page config
st.set_page_config(page_title="Compliance", page_icon="✅", layout="wide")

# Initialize session state
initialize_session_state()

st.title("Compliance Monitoring")

# Load data
df_compliance = load_data("compliance")
df_counterparties = load_data("counterparties")
df_transactions = load_data("transactions")

# Sidebar filters
st.sidebar.header("Filters")

# Compliance type filter
all_types = df_compliance['compliance_type'].unique().tolist()
selected_types = st.sidebar.multiselect(
    "Filter by Compliance Type",
    options=all_types,
    default=all_types
)

# Status filter
all_statuses = df_compliance['status'].unique().tolist()
selected_statuses = st.sidebar.multiselect(
    "Filter by Status",
    options=all_statuses,
    default=all_statuses
)

# Apply filters
filtered_compliance = df_compliance[
    (df_compliance['compliance_type'].isin(selected_types)) &
    (df_compliance['status'].isin(selected_statuses))
]

# Compliance overview
st.subheader("Compliance Overview")

# Compliance statistics
col1, col2, col3, col4 = st.columns(4)

compliant_count = len(filtered_compliance[filtered_compliance['status'] == 'Compliant'])
non_compliant_count = len(filtered_compliance[filtered_compliance['status'] == 'Non-Compliant'])
pending_count = len(filtered_compliance[filtered_compliance['status'] == 'Pending Review'])
total_count = len(filtered_compliance)

with col1:
    st.metric(
        "Compliant", 
        f"{compliant_count}",
        delta=f"{100 * compliant_count / total_count:.1f}%" if total_count > 0 else "0%"
    )

with col2:
    st.metric(
        "Non-Compliant", 
        f"{non_compliant_count}",
        delta=f"{100 * non_compliant_count / total_count:.1f}%" if total_count > 0 else "0%",
        delta_color="inverse"
    )

with col3:
    st.metric(
        "Pending Review", 
        f"{pending_count}"
    )

with col4:
    avg_risk = filtered_compliance['risk_score'].mean() if len(filtered_compliance) > 0 else 0
    st.metric("Average Risk Score", f"{avg_risk:.2f}")

# Compliance status distribution chart
st.subheader("Compliance Status Distribution")

col1, col2 = st.columns(2)

with col1:
    fig = plot_compliance_status(filtered_compliance)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Compliance by type
    type_counts = filtered_compliance.groupby(['compliance_type', 'status']).size().reset_index(name='count')
    
    fig = px.bar(
        type_counts,
        x='compliance_type',
        y='count',
        color='status',
        title="Compliance Status by Type",
        labels={
            'compliance_type': 'Compliance Type',
            'count': 'Count',
            'status': 'Status'
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Compliance calendar
st.subheader("Upcoming Reviews")

# Filter for upcoming reviews in the next 30 days
now = datetime.now()
upcoming_reviews = filtered_compliance[
    (pd.to_datetime(filtered_compliance['next_review_date']) <= now + timedelta(days=30)) &
    (pd.to_datetime(filtered_compliance['next_review_date']) >= now)
]

# Sort by next review date
upcoming_reviews = upcoming_reviews.sort_values('next_review_date')

if len(upcoming_reviews) > 0:
    # Display upcoming reviews
    st.dataframe(
        upcoming_reviews[['counterparty_name', 'compliance_type', 'status', 'next_review_date', 'risk_score']],
        use_container_width=True
    )
else:
    st.info("No upcoming reviews in the next 30 days.")

# Detailed compliance analysis for a selected counterparty
st.subheader("Counterparty Compliance Analysis")

# Counterparty selector
counterparty_options = df_counterparties['name'].tolist()
selected_counterparty = st.selectbox(
    "Select Counterparty",
    options=counterparty_options
)

# Get counterparty data
selected_cp_data = df_counterparties[df_counterparties['name'] == selected_counterparty].iloc[0]
selected_cp_id = selected_cp_data['id']

# Get compliance records for this counterparty
cp_compliance = df_compliance[df_compliance['counterparty_id'] == selected_cp_id]

# Get transactions for this counterparty
cp_transactions = df_transactions[df_transactions['counterparty_id'] == selected_cp_id]

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(f"{selected_counterparty}")
    st.markdown(f"**Sector:** {selected_cp_data['sector']}")
    st.markdown(f"**Country:** {selected_cp_data['country']}")
    st.markdown(f"**Credit Rating:** {selected_cp_data['credit_rating']}")
    st.markdown(f"**Last Review Date:** {selected_cp_data['last_review_date']}")
    
    # Run KYC compliance check
    kyc_check = validate_kyc_compliance(selected_cp_data)
    
    st.markdown("---")
    st.markdown(f"**KYC Status:** {kyc_check['status']}")
    st.markdown(f"**KYC Risk Score:** {kyc_check['risk_score']:.2f}")
    
    if kyc_check['issues']:
        st.markdown("**KYC Issues:**")
        for issue in kyc_check['issues']:
            st.markdown(f"- {issue}")
    else:
        st.markdown("**KYC Issues:** None")

with col2:
    # Run AML check on recent transactions
    if len(cp_transactions) > 0:
        recent_tx = cp_transactions.sort_values('timestamp', ascending=False).iloc[0]
        aml_check = validate_aml_compliance(recent_tx)
        
        # Run sanctions check
        sanctions_check = check_sanctions(selected_counterparty)
        
        # Create gauge charts for different compliance aspects
        fig = go.Figure()
        
        # KYC risk gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=kyc_check['risk_score'],
            title={'text': "KYC Risk"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "green"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ]
            },
            domain={'row': 0, 'column': 0}
        ))
        
        # AML risk gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=aml_check['risk_score'],
            title={'text': "AML Risk"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "green"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ]
            },
            domain={'row': 0, 'column': 1}
        ))
        
        # Sanctions risk gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=sanctions_check['risk_score'],
            title={'text': "Sanctions Risk"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "green"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ]
            },
            domain={'row': 1, 'column': 0}
        ))
        
        # Overall risk gauge (average of the three)
        overall_risk = (kyc_check['risk_score'] + aml_check['risk_score'] + sanctions_check['risk_score']) / 3
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=overall_risk,
            title={'text': "Overall Compliance Risk"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "green"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ]
            },
            domain={'row': 1, 'column': 1}
        ))
        
        fig.update_layout(
            grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No transactions found for this counterparty.")

# AI-powered compliance analysis
st.subheader("AI-Powered Compliance Analysis")

# Create tabs for different analyses
ai_tabs = st.tabs(["Counterparty Analysis", "Transaction Analysis", "Recommendations"])

with ai_tabs[0]:
    # Convert counterparty data to JSON for OpenAI analysis
    counterparty_json = selected_cp_data.to_dict()
    
    # Get AI analysis
    with st.spinner("Analyzing counterparty data..."):
        analysis = analyze_compliance_risk(counterparty_json, selected_counterparty)
    
    # Display analysis
    st.markdown(f"**Summary:** {analysis.get('summary', 'No summary available')}")
    st.markdown(f"**Compliance Status:** {analysis.get('compliance_status', 'Unknown')}")
    
    st.markdown("**Risk Factors:**")
    for factor in analysis.get('risk_factors', ['No risk factors identified']):
        st.markdown(f"- {factor}")

with ai_tabs[1]:
    if len(cp_transactions) > 0:
        # Convert recent transactions to JSON for OpenAI analysis
        recent_txs = cp_transactions.sort_values('timestamp', ascending=False).head(5)
        txs_json = recent_txs.to_dict(orient='records')
        
        # Get AI analysis
        with st.spinner("Analyzing transaction data..."):
            tx_analysis = analyze_compliance_risk(
                {"counterparty": selected_counterparty, "transactions": txs_json},
                f"Transactions for {selected_counterparty}"
            )
        
        # Display analysis
        st.markdown(f"**Summary:** {tx_analysis.get('summary', 'No summary available')}")
        st.markdown(f"**Transaction Risk Level:** {tx_analysis.get('compliance_status', 'Unknown')}")
        
        st.markdown("**Transaction Risk Factors:**")
        for factor in tx_analysis.get('risk_factors', ['No risk factors identified']):
            st.markdown(f"- {factor}")
    else:
        st.info("No transactions available for analysis.")

with ai_tabs[2]:
    # Display recommendations from both analyses
    st.markdown("### Recommendations")
    
    recommendations = []
    if 'recommendations' in analysis:
        recommendations.extend(analysis['recommendations'])
    
    if len(cp_transactions) > 0 and 'recommendations' in tx_analysis:
        recommendations.extend(tx_analysis['recommendations'])
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.info("No recommendations available.")

# Compliance checklist
st.subheader("Compliance Checklist")

# Create tabs for different compliance types
checklist_tabs = st.tabs(["KYC", "AML", "Sanctions", "Regulatory"])

with checklist_tabs[0]:
    st.markdown("### KYC Requirements")
    
    kyc_requirements = [
        "Customer identification verified",
        "Proof of address obtained",
        "Beneficial owners identified",
        "PEP screening conducted",
        "Risk assessment completed",
        "Enhanced due diligence (if required)",
        "Documentation stored securely",
        "Periodic review scheduled"
    ]
    
    for req in kyc_requirements:
        st.checkbox(req, value=np.random.random() > 0.3)

with checklist_tabs[1]:
    st.markdown("### AML Requirements")
    
    aml_requirements = [
        "Transaction monitoring in place",
        "Suspicious activity reporting process",
        "Staff AML training conducted",
        "Risk-based approach implemented",
        "Customer risk profiles maintained",
        "Transaction thresholds established",
        "AML policies documented",
        "Independent audit completed"
    ]
    
    for req in aml_requirements:
        st.checkbox(req, value=np.random.random() > 0.3)

with checklist_tabs[2]:
    st.markdown("### Sanctions Requirements")
    
    sanctions_requirements = [
        "Screening against global sanctions lists",
        "Real-time sanctions screening for transactions",
        "Automated screening system implemented",
        "Sanctions updates monitored",
        "False positive handling process",
        "Screening records maintained",
        "Blocked transactions process",
        "Sanctions compliance training"
    ]
    
    for req in sanctions_requirements:
        st.checkbox(req, value=np.random.random() > 0.3)

with checklist_tabs[3]:
    st.markdown("### Regulatory Requirements")
    
    regulatory_requirements = [
        "Regulatory reporting framework",
        "Compliance officer appointed",
        "Board oversight established",
        "Regulatory changes monitored",
        "Internal controls documented",
        "Compliance testing conducted",
        "Regulatory examinations prepared",
        "Remediation process in place"
    ]
    
    for req in regulatory_requirements:
        st.checkbox(req, value=np.random.random() > 0.3)

# Export compliance report
st.subheader("Compliance Reporting")

if st.button("Generate Compliance Report"):
    with st.spinner("Generating compliance report..."):
        report = generate_compliance_report(df_counterparties, df_transactions)
        
        # Display report summary
        st.json(report["summary"])
        
        # Option to download full report
        st.download_button(
            label="Download Full Report",
            data=str(report),
            file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Refresh data button
if st.sidebar.button("Refresh Data"):
    st.rerun()

# Last updated time
st.sidebar.markdown("---")
st.sidebar.text(f"Last updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
