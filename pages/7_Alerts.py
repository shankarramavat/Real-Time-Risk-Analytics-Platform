import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.data_generator import load_data
from utils.compliance import validate_kyc_compliance, validate_aml_compliance, check_sanctions

# Page config
st.set_page_config(page_title="Alerts", page_icon="🚨", layout="wide")

st.title("Risk Alerts & Notifications")

# Load data
df_counterparties = load_data("counterparties")
df_transactions = load_data("transactions")
df_market_data = load_data("market_data")
df_compliance = load_data("compliance")

# Initialize alert count in session state if not exists
if 'alert_count' not in st.session_state:
    st.session_state.alert_count = 5

# Generate alerts based on data
def generate_alerts():
    alerts = []
    
    # Counterparty alerts
    high_risk_counterparties = df_counterparties[df_counterparties['risk_score'] > 0.8]
    for _, cp in high_risk_counterparties.head(3).iterrows():
        alerts.append({
            "level": "High",
            "category": "Counterparty Risk",
            "message": f"High risk score for {cp['name']} ({cp['risk_score']:.2f})",
            "entity": cp['name'],
            "timestamp": datetime.now() - timedelta(minutes=np.random.randint(5, 120)),
            "details": {
                "counterparty_id": cp['id'],
                "risk_score": cp['risk_score'],
                "sector": cp['sector'],
                "country": cp['country'],
                "credit_rating": cp['credit_rating']
            }
        })
    
    # Exposure limit alerts
    near_limit_cps = df_counterparties[df_counterparties['current_exposure'] > 0.9 * df_counterparties['exposure_limit']]
    for _, cp in near_limit_cps.head(3).iterrows():
        limit_pct = (cp['current_exposure'] / cp['exposure_limit']) * 100
        alerts.append({
            "level": "Medium" if limit_pct < 100 else "High",
            "category": "Exposure Limits",
            "message": f"{cp['name']} at {limit_pct:.1f}% of exposure limit",
            "entity": cp['name'],
            "timestamp": datetime.now() - timedelta(minutes=np.random.randint(10, 180)),
            "details": {
                "counterparty_id": cp['id'],
                "current_exposure": cp['current_exposure'],
                "exposure_limit": cp['exposure_limit'],
                "percentage": limit_pct,
                "sector": cp['sector']
            }
        })
    
    # Transaction alerts
    suspicious_txns = df_transactions[df_transactions['risk_score'] > 0.75]
    for _, tx in suspicious_txns.head(4).iterrows():
        cp_name = tx['counterparty_name']
        alerts.append({
            "level": "High" if tx['risk_score'] > 0.85 else "Medium",
            "category": "Suspicious Transaction",
            "message": f"High-risk transaction ({tx['transaction_type']}) with {cp_name}",
            "entity": cp_name,
            "timestamp": datetime.strptime(tx['timestamp'], '%Y-%m-%d %H:%M:%S'),
            "details": {
                "transaction_id": tx['id'],
                "counterparty_id": tx['counterparty_id'],
                "amount": tx['amount'],
                "currency": tx['currency'],
                "risk_score": tx['risk_score'],
                "flagged": tx['flagged']
            }
        })
    
    # Compliance alerts
    non_compliant = df_compliance[df_compliance['status'].isin(['Non-Compliant', 'Under Investigation'])]
    for _, cp in non_compliant.head(3).iterrows():
        alerts.append({
            "level": "High",
            "category": "Compliance Issue",
            "message": f"{cp['compliance_type']} compliance issue with {cp['counterparty_name']}",
            "entity": cp['counterparty_name'],
            "timestamp": datetime.now() - timedelta(hours=np.random.randint(1, 24)),
            "details": {
                "compliance_id": cp['id'],
                "counterparty_id": cp['counterparty_id'],
                "compliance_type": cp['compliance_type'],
                "status": cp['status'],
                "risk_score": cp['risk_score'],
                "next_review_date": cp['next_review_date']
            }
        })
    
    # Market risk alerts
    recent_market = df_market_data[df_market_data['date'] == df_market_data['date'].max()]
    high_vol_markets = recent_market[recent_market['volatility'] > 0.7]
    for _, mkt in high_vol_markets.head(2).iterrows():
        alerts.append({
            "level": "Medium",
            "category": "Market Volatility",
            "message": f"High volatility in {mkt['sector']} sector ({mkt['region']})",
            "entity": f"{mkt['sector']} - {mkt['region']}",
            "timestamp": datetime.now() - timedelta(hours=np.random.randint(2, 8)),
            "details": {
                "sector": mkt['sector'],
                "region": mkt['region'],
                "volatility": mkt['volatility'],
                "market_risk": mkt['market_risk'],
                "liquidity": mkt['liquidity'],
                "change_pct": mkt['change_pct']
            }
        })
    
    # Sort by timestamp (most recent first)
    alerts.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return alerts

alerts = generate_alerts()
st.session_state.alert_count = len(alerts)

# Alert dashboard
st.subheader("Active Alerts")

# Alert statistics
col1, col2, col3, col4 = st.columns(4)

high_alerts = len([a for a in alerts if a['level'] == 'High'])
medium_alerts = len([a for a in alerts if a['level'] == 'Medium'])
low_alerts = len([a for a in alerts if a['level'] == 'Low'])

with col1:
    st.metric("Total Alerts", f"{len(alerts)}")

with col2:
    st.metric("High Priority", f"{high_alerts}", delta=f"{high_alerts}" if high_alerts > 0 else None, delta_color="inverse")

with col3:
    st.metric("Medium Priority", f"{medium_alerts}")

with col4:
    st.metric("Low Priority", f"{low_alerts}")

# Alert filters
st.sidebar.header("Filter Alerts")

# Priority filter
priority_options = ["All", "High", "Medium", "Low"]
selected_priority = st.sidebar.selectbox(
    "Priority Level",
    options=priority_options,
    index=0
)

# Category filter
all_categories = sorted(list(set([a['category'] for a in alerts])))
selected_category = st.sidebar.multiselect(
    "Alert Category",
    options=["All"] + all_categories,
    default=["All"]
)

# Time range filter
time_range_options = ["All Time", "Last 24 Hours", "Last 7 Days", "Last 30 Days"]
selected_time_range = st.sidebar.selectbox(
    "Time Range",
    options=time_range_options,
    index=1
)

# Apply filters
filtered_alerts = alerts.copy()

# Priority filter
if selected_priority != "All":
    filtered_alerts = [a for a in filtered_alerts if a['level'] == selected_priority]

# Category filter
if "All" not in selected_category:
    filtered_alerts = [a for a in filtered_alerts if a['category'] in selected_category]

# Time range filter
now = datetime.now()
if selected_time_range == "Last 24 Hours":
    filtered_alerts = [a for a in filtered_alerts if (now - a['timestamp']).total_seconds() / 3600 <= 24]
elif selected_time_range == "Last 7 Days":
    filtered_alerts = [a for a in filtered_alerts if (now - a['timestamp']).days <= 7]
elif selected_time_range == "Last 30 Days":
    filtered_alerts = [a for a in filtered_alerts if (now - a['timestamp']).days <= 30]

# Display alerts
if not filtered_alerts:
    st.info("No alerts match the selected filters.")
else:
    # Alert list
    for alert in filtered_alerts:
        level_color = {
            "High": "red",
            "Medium": "orange",
            "Low": "green"
        }.get(alert["level"], "gray")
        
        # Calculate time ago
        time_diff = now - alert["timestamp"]
        if time_diff.days > 0:
            time_ago = f"{time_diff.days} day{'s' if time_diff.days != 1 else ''} ago"
        elif time_diff.seconds // 3600 > 0:
            hours = time_diff.seconds // 3600
            time_ago = f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif time_diff.seconds // 60 > 0:
            minutes = time_diff.seconds // 60
            time_ago = f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            time_ago = "just now"
        
        # Create alert container
        col1, col2 = st.columns([5, 1])
        
        with col1:
            expander = st.expander(
                f"[{alert['category']}] {alert['message']} ({time_ago})",
                expanded=False
            )
            
            with expander:
                st.markdown(f"**Priority:** {alert['level']}")
                st.markdown(f"**Entity:** {alert['entity']}")
                st.markdown(f"**Time:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ({time_ago})")
                
                # Display details
                st.markdown("**Details:**")
                details_df = pd.DataFrame([alert['details']])
                st.dataframe(details_df, use_container_width=True)
                
                # Add action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Mark as Reviewed", key=f"review_{alert['category']}_{alerts.index(alert)}"):
                        st.success("Alert marked as reviewed")
                with col2:
                    if st.button("Escalate", key=f"escalate_{alert['category']}_{alerts.index(alert)}"):
                        st.warning("Alert escalated to risk manager")
                with col3:
                    if st.button("Dismiss", key=f"dismiss_{alert['category']}_{alerts.index(alert)}"):
                        st.info("Alert dismissed")
        
        with col2:
            st.markdown(
                f"""
                <div style="
                    width: 20px;
                    height: 20px;
                    border-radius: 50%;
                    background-color: {level_color};
                    margin-top: 10px;
                "></div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")

# Alert trends
st.subheader("Alert Trends")

# Create sample historical alert data
days = 30
np.random.seed(42)

dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
dates.reverse()  # oldest first

alert_history = []
for date in dates:
    # Generate random counts with an increasing trend
    day_offset = dates.index(date)
    base_high = max(1, min(8, int(high_alerts * 0.8 + np.random.normal(0, 1) + day_offset * 0.1)))
    base_medium = max(1, min(12, int(medium_alerts * 0.8 + np.random.normal(0, 2) + day_offset * 0.15)))
    base_low = max(0, min(5, int(low_alerts * 0.7 + np.random.normal(0, 1) + day_offset * 0.05)))
    
    alert_history.append({
        'date': date,
        'High': base_high,
        'Medium': base_medium,
        'Low': base_low,
        'Total': base_high + base_medium + base_low
    })

# Convert to dataframe
alert_df = pd.DataFrame(alert_history)

# Plot alert trends
fig = px.line(
    alert_df,
    x='date',
    y=['High', 'Medium', 'Low', 'Total'],
    title="Alert Trends (Last 30 Days)",
    labels={'value': 'Number of Alerts', 'variable': 'Priority', 'date': 'Date'},
    color_discrete_map={
        'High': 'red',
        'Medium': 'orange',
        'Low': 'green',
        'Total': 'blue'
    }
)

st.plotly_chart(fig, use_container_width=True)

# Alert notifications settings
st.subheader("Alert Notification Settings")

# Create tabs for different settings
settings_tabs = st.tabs(["Notification Channels", "Alert Thresholds", "Alert Rules"])

with settings_tabs[0]:
    st.markdown("### Notification Channels")
    
    # Email notifications
    st.markdown("#### Email Notifications")
    email_enabled = st.checkbox("Enable Email Notifications", value=True)
    
    if email_enabled:
        email_col1, email_col2 = st.columns(2)
        
        with email_col1:
            email_recipients = st.text_area("Email Recipients (one per line)", "risk_manager@example.com\ncompliance@example.com")
        
        with email_col2:
            email_frequency = st.radio(
                "Email Frequency",
                options=["Real-time", "Hourly Digest", "Daily Digest"],
                index=1
            )
    
    # SMS notifications
    st.markdown("#### SMS Notifications")
    sms_enabled = st.checkbox("Enable SMS Notifications", value=True)
    
    if sms_enabled:
        sms_col1, sms_col2 = st.columns(2)
        
        with sms_col1:
            sms_recipients = st.text_area("SMS Recipients (one per line)", "+1234567890\n+0987654321")
        
        with sms_col2:
            sms_priority = st.multiselect(
                "SMS Priority Levels",
                options=["High", "Medium", "Low"],
                default=["High"]
            )
    
    # In-app notifications
    st.markdown("#### In-App Notifications")
    inapp_enabled = st.checkbox("Enable In-App Notifications", value=True)
    
    if inapp_enabled:
        inapp_col1, inapp_col2 = st.columns(2)
        
        with inapp_col1:
            inapp_sound = st.checkbox("Enable Sound Alerts", value=True)
        
        with inapp_col2:
            inapp_desktop = st.checkbox("Enable Desktop Notifications", value=True)
    
    # Save settings button
    if st.button("Save Notification Settings"):
        st.success("Notification settings saved successfully!")

with settings_tabs[1]:
    st.markdown("### Alert Thresholds")
    
    # Risk score thresholds
    st.markdown("#### Risk Score Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        high_risk_threshold = st.slider(
            "High Risk Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05
        )
    
    with col2:
        medium_risk_threshold = st.slider(
            "Medium Risk Threshold",
            min_value=0.3,
            max_value=0.7,
            value=0.5,
            step=0.05
        )
    
    # Exposure thresholds
    st.markdown("#### Exposure Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        exposure_alert_threshold = st.slider(
            "Exposure Alert Threshold (% of limit)",
            min_value=50,
            max_value=100,
            value=90,
            step=5
        )
    
    with col2:
        large_transaction_threshold = st.number_input(
            "Large Transaction Threshold",
            min_value=100000,
            max_value=10000000,
            value=1000000,
            step=100000,
            format="%d"
        )
    
    # Compliance thresholds
    st.markdown("#### Compliance Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        compliance_risk_threshold = st.slider(
            "Compliance Risk Alert Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
    
    with col2:
        review_reminder_days = st.slider(
            "Days Before Review to Send Reminder",
            min_value=1,
            max_value=30,
            value=7,
            step=1
        )
    
    # Save settings button
    if st.button("Save Threshold Settings"):
        st.success("Threshold settings saved successfully!")

with settings_tabs[2]:
    st.markdown("### Alert Rules")
    
    # Sample alert rules
    alert_rules = [
        {
            "rule_id": 1,
            "name": "High Risk Counterparty",
            "description": "Alert when counterparty risk score exceeds high risk threshold",
            "enabled": True,
            "priority": "High"
        },
        {
            "rule_id": 2,
            "name": "Near Exposure Limit",
            "description": "Alert when counterparty exposure approaches limit",
            "enabled": True,
            "priority": "Medium"
        },
        {
            "rule_id": 3,
            "name": "Suspicious Transaction",
            "description": "Alert on transactions with high risk scores",
            "enabled": True,
            "priority": "High"
        },
        {
            "rule_id": 4,
            "name": "Compliance Issue",
            "description": "Alert on non-compliant counterparties",
            "enabled": True,
            "priority": "High"
        },
        {
            "rule_id": 5,
            "name": "Market Volatility",
            "description": "Alert on unusually high market volatility",
            "enabled": True,
            "priority": "Medium"
        }
    ]
    
    # Convert to dataframe for display
    rules_df = pd.DataFrame(alert_rules)
    
    # Add toggle column
    rules_df["Action"] = "Enabled"
    
    # Display rules with editable cells
    edited_df = st.data_editor(
        rules_df,
        column_config={
            "rule_id": st.column_config.NumberColumn("ID", disabled=True),
            "name": st.column_config.TextColumn("Rule Name"),
            "description": st.column_config.TextColumn("Description"),
            "priority": st.column_config.SelectboxColumn(
                "Priority",
                options=["High", "Medium", "Low"],
            ),
            "Action": st.column_config.SelectboxColumn(
                "Action",
                options=["Enabled", "Disabled"],
                width="small",
            ),
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Add rule
    st.markdown("#### Add New Alert Rule")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_rule_name = st.text_input("Rule Name")
        new_rule_description = st.text_area("Rule Description")
    
    with col2:
        new_rule_priority = st.selectbox(
            "Priority",
            options=["High", "Medium", "Low"]
        )
        new_rule_enabled = st.checkbox("Enable Rule", value=True)
    
    if st.button("Add Rule"):
        st.success("New alert rule added successfully!")

# Alert distribution
st.subheader("Alert Distribution")

col1, col2 = st.columns(2)

with col1:
    # Alert distribution by category
    category_counts = {}
    for alert in alerts:
        category = alert['category']
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1
    
    category_df = pd.DataFrame({
        'Category': list(category_counts.keys()),
        'Count': list(category_counts.values())
    })
    
    fig = px.pie(
        category_df,
        values='Count',
        names='Category',
        title="Alerts by Category"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Alert distribution by priority
    priority_counts = {
        'High': len([a for a in alerts if a['level'] == 'High']),
        'Medium': len([a for a in alerts if a['level'] == 'Medium']),
        'Low': len([a for a in alerts if a['level'] == 'Low'])
    }
    
    priority_df = pd.DataFrame({
        'Priority': list(priority_counts.keys()),
        'Count': list(priority_counts.values())
    })
    
    fig = px.bar(
        priority_df,
        x='Priority',
        y='Count',
        color='Priority',
        title="Alerts by Priority",
        color_discrete_map={
            'High': 'red',
            'Medium': 'orange',
            'Low': 'green'
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Refresh button
if st.button("Refresh Alerts"):
    alerts = generate_alerts()
    st.session_state.alert_count = len(alerts)
    st.rerun()

# Last updated time
st.sidebar.markdown("---")
st.sidebar.text(f"Last updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
