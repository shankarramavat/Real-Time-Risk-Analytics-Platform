import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import re

from utils.data_generator import load_data
from utils.notification_service import (
    send_transaction_alert, 
    send_account_security_alert,
    get_notification_history,
    generate_transaction_alert_message
)

# Page config
st.set_page_config(page_title="Mobile Notifications", page_icon="📱", layout="wide")

st.title("Mobile Fraud Alerts & Notifications")

# Load data
df_transactions = load_data("transactions")
df_counterparties = load_data("counterparties")

# Initialize user data in session state if not exists
if 'user_profiles' not in st.session_state:
    st.session_state.user_profiles = [
        {
            "id": 1,
            "name": "John Smith",
            "phone": "+12345678900",  # Example phone number
            "email": "john.smith@example.com",
            "notification_preferences": {
                "sms": True,
                "email": True,
                "push": True
            },
            "biometric_enabled": True,
            "transaction_threshold": 1000,  # Threshold for transaction alerts
            "last_login": (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S'),
            "devices": ["iPhone 13", "MacBook Pro"],
            "location": "New York, USA"
        },
        {
            "id": 2,
            "name": "Jane Doe",
            "phone": "+19876543210",  # Example phone number
            "email": "jane.doe@example.com",
            "notification_preferences": {
                "sms": True,
                "email": False,
                "push": True
            },
            "biometric_enabled": False,
            "transaction_threshold": 500,  # Threshold for transaction alerts
            "last_login": (datetime.now() - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M:%S'),
            "devices": ["Samsung Galaxy S22", "Windows PC"],
            "location": "Chicago, USA"
        }
    ]

# Sidebar - User selection and notification settings
st.sidebar.header("User & Notification Settings")

# User selection
selected_user_index = st.sidebar.selectbox(
    "Select User",
    options=range(len(st.session_state.user_profiles)),
    format_func=lambda x: st.session_state.user_profiles[x]["name"]
)
selected_user = st.session_state.user_profiles[selected_user_index]

# Display user notification preferences
st.sidebar.subheader("Notification Preferences")
sms_enabled = st.sidebar.checkbox("SMS Notifications", value=selected_user["notification_preferences"]["sms"])
email_enabled = st.sidebar.checkbox("Email Notifications", value=selected_user["notification_preferences"]["email"])
push_enabled = st.sidebar.checkbox("Push Notifications", value=selected_user["notification_preferences"]["push"])

# Update preferences if changed
if (sms_enabled != selected_user["notification_preferences"]["sms"] or
    email_enabled != selected_user["notification_preferences"]["email"] or
    push_enabled != selected_user["notification_preferences"]["push"]):
    
    selected_user["notification_preferences"]["sms"] = sms_enabled
    selected_user["notification_preferences"]["email"] = email_enabled
    selected_user["notification_preferences"]["push"] = push_enabled
    
    st.sidebar.success("Notification preferences updated!")

# Transaction threshold for alerts
transaction_threshold = st.sidebar.number_input(
    "Transaction Alert Threshold ($)",
    min_value=100,
    max_value=10000,
    value=selected_user["transaction_threshold"],
    step=100
)

if transaction_threshold != selected_user["transaction_threshold"]:
    selected_user["transaction_threshold"] = transaction_threshold
    st.sidebar.success("Alert threshold updated!")

# Biometric authentication toggle
biometric_enabled = st.sidebar.checkbox(
    "Require Biometric Verification for High-Risk Transactions",
    value=selected_user["biometric_enabled"]
)

if biometric_enabled != selected_user["biometric_enabled"]:
    selected_user["biometric_enabled"] = biometric_enabled
    st.sidebar.success("Biometric settings updated!")

# Phone number input for alerts
phone_number = st.sidebar.text_input(
    "Phone Number for Alerts",
    value=selected_user["phone"]
)

if phone_number != selected_user["phone"]:
    # Basic phone number validation
    phone_pattern = r'^\+\d{1,15}$'
    if re.match(phone_pattern, phone_number):
        selected_user["phone"] = phone_number
        st.sidebar.success("Phone number updated!")
    else:
        st.sidebar.error("Invalid phone number format. Please use international format (e.g., +12345678900)")
        phone_number = selected_user["phone"]  # Revert to original

# Main content - Tabs for different sections
tabs = st.tabs([
    "Fraud Alert Simulation", 
    "Notification History", 
    "Device Management",
    "Security Settings"
])

# Fraud Alert Simulation tab
with tabs[0]:
    st.header("Fraud Alert Simulation")
    st.markdown("""
    This panel allows you to simulate fraud alerts being sent to users' mobile devices. 
    In a real-world scenario, these alerts would be triggered automatically by the risk analytics system.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Simulate Transaction Alert")
        
        # Get high risk transactions
        high_risk_txns = df_transactions[df_transactions['risk_score'] > 0.7]
        
        if len(high_risk_txns) == 0:
            st.warning("No high-risk transactions found for simulation.")
        else:
            selected_txn_index = st.selectbox(
                "Select High-Risk Transaction",
                options=range(len(high_risk_txns)),
                format_func=lambda x: f"{high_risk_txns.iloc[x]['counterparty_name']} - ${high_risk_txns.iloc[x]['amount']:,.2f} ({high_risk_txns.iloc[x]['transaction_type']})"
            )
            
            selected_txn = high_risk_txns.iloc[selected_txn_index].to_dict()
            
            st.markdown("#### Transaction Details")
            st.markdown(f"**Counterparty:** {selected_txn['counterparty_name']}")
            st.markdown(f"**Amount:** ${selected_txn['amount']:,.2f} {selected_txn['currency']}")
            st.markdown(f"**Type:** {selected_txn['transaction_type']}")
            st.markdown(f"**Risk Score:** {selected_txn['risk_score']:.2f}")
            
            # Preview the SMS message
            message = generate_transaction_alert_message(selected_txn)
            
            st.markdown("#### SMS Preview")
            st.info(message)
            
            if st.button("Send Transaction Alert"):
                result = send_transaction_alert(selected_txn, phone_number)
                
                if result["success"]:
                    st.success("Transaction alert sent successfully!")
                else:
                    st.error(f"Failed to send alert: {result['message']}")
                    if "Twilio credentials not configured" in result["message"]:
                        st.info("Alert saved to notification history for demonstration purposes.")
    
    with col2:
        st.subheader("Simulate Account Security Alert")
        
        # Simulate account security alert
        alert_types = [
            "New Login from Unknown Device",
            "Password Changed",
            "Failed Login Attempts",
            "Suspicious Location Access",
            "New Device Added"
        ]
        
        selected_alert_type = st.selectbox("Alert Type", options=alert_types)
        
        # Devices
        device_options = ["iPhone", "Android Phone", "iPad", "Windows PC", "MacBook", "Unknown Device"]
        selected_device = st.selectbox("Device", options=device_options)
        
        # Locations
        location_options = ["New York, USA", "London, UK", "Tokyo, Japan", "Sydney, Australia", "Moscow, Russia", "Unknown Location"]
        selected_location = st.selectbox("Location", options=location_options)
        
        # Generate alert data
        alert_data = {
            "id": f"SEC{np.random.randint(10000, 99999)}",
            "type": selected_alert_type,
            "device": selected_device,
            "location": selected_location,
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "risk_score": np.random.uniform(0.7, 0.95)
        }
        
        # Preview security alert
        st.markdown("#### Alert Preview")
        
        message = (
            f"SECURITY ALERT: {selected_alert_type} detected from {selected_device} in {selected_location}. "
            f"Reply YES {alert_data['id']} if this was you or NO {alert_data['id']} to secure your account."
        )
        
        st.info(message)
        
        if st.button("Send Security Alert"):
            result = send_account_security_alert(alert_data, phone_number)
            
            if result["success"]:
                st.success("Security alert sent successfully!")
            else:
                st.error(f"Failed to send alert: {result['message']}")
                if "Twilio credentials not configured" in result["message"]:
                    st.info("Alert saved to notification history for demonstration purposes.")

# Notification History tab
with tabs[1]:
    st.header("Notification History")
    
    # Get notification history
    notification_history = get_notification_history()
    
    if not notification_history:
        st.info("No notification history found. Send an alert to see it in the history.")
    else:
        # Convert history to DataFrame for display
        history_data = []
        for entry in notification_history:
            history_data.append({
                "timestamp": entry.get("timestamp", ""),
                "recipient": entry.get("to", ""),
                "type": entry.get("alert_type", "unknown").capitalize(),
                "status": "Sent" if entry.get("success", False) else "Failed",
                "content": entry.get("content", "")
            })
        
        history_df = pd.DataFrame(history_data)
        
        # Sort by timestamp, most recent first
        history_df = history_df.sort_values("timestamp", ascending=False)
        
        # Display history
        st.dataframe(history_df, use_container_width=True)
        
        # Chart of notifications over time
        if len(history_df) > 1:
            st.subheader("Notifications Over Time")
            
            # Create a simplified timestamp for grouping
            history_df["date"] = pd.to_datetime(history_df["timestamp"]).dt.date
            
            # Count notifications by date and type
            notification_counts = history_df.groupby(["date", "type"]).size().reset_index(name="count")
            
            # Create chart
            fig = px.bar(
                notification_counts, 
                x="date", 
                y="count", 
                color="type",
                title="Notifications Sent by Type",
                labels={"date": "Date", "count": "Number of Notifications", "type": "Notification Type"}
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Device Management tab
with tabs[2]:
    st.header("Device Management")
    
    st.markdown("""
    Manage the devices associated with your account. Any unrecognized devices may indicate unauthorized access.
    """)
    
    # Display current devices
    st.subheader("Registered Devices")
    
    if "devices" in selected_user:
        for i, device in enumerate(selected_user["devices"]):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{device}**")
            
            with col2:
                st.markdown("Active" if i == 0 else "Last used 5 days ago")
            
            with col3:
                if st.button("Remove", key=f"remove_device_{i}"):
                    selected_user["devices"].pop(i)
                    st.rerun()
    else:
        st.info("No devices registered.")
    
    # Add new device
    st.subheader("Add New Device")
    
    new_device = st.text_input("Device Name")
    
    if st.button("Add Device") and new_device:
        if "devices" not in selected_user:
            selected_user["devices"] = []
        
        selected_user["devices"].append(new_device)
        st.success(f"Added {new_device} to your registered devices.")
        st.rerun()

# Security Settings tab
with tabs[3]:
    st.header("Security Settings")
    
    st.markdown("""
    Configure security settings for mobile fraud alerts and notifications.
    """)
    
    # Biometric authentication settings
    st.subheader("Biometric Authentication")
    
    biometric_types = ["Fingerprint", "Face ID", "Voice Recognition"]
    selected_biometric = st.multiselect("Enabled Biometric Methods", options=biometric_types, default=["Fingerprint"] if selected_user["biometric_enabled"] else [])
    
    if st.button("Update Biometric Settings"):
        selected_user["biometric_enabled"] = len(selected_biometric) > 0
        st.success("Biometric settings updated.")
    
    # Transaction verification settings
    st.subheader("Transaction Verification")
    
    verification_threshold = st.slider(
        "Require Verification for Transactions Above ($)",
        min_value=100,
        max_value=5000,
        value=selected_user.get("verification_threshold", 1000),
        step=100
    )
    
    high_risk_always = st.checkbox(
        "Always Verify High-Risk Transactions (Risk Score > 0.7)",
        value=selected_user.get("high_risk_always_verify", True)
    )
    
    if st.button("Update Verification Settings"):
        selected_user["verification_threshold"] = verification_threshold
        selected_user["high_risk_always_verify"] = high_risk_always
        st.success("Transaction verification settings updated.")
    
    # Block suspicious logins
    st.subheader("Login Security")
    
    block_suspicious = st.checkbox(
        "Block Suspicious Login Attempts",
        value=selected_user.get("block_suspicious_logins", True)
    )
    
    location_tracking = st.checkbox(
        "Enable Location-Based Authentication",
        value=selected_user.get("location_tracking", True),
        help="Track login locations and flag unusual login attempts."
    )
    
    if st.button("Update Login Security"):
        selected_user["block_suspicious_logins"] = block_suspicious
        selected_user["location_tracking"] = location_tracking
        st.success("Login security settings updated.")

# Information about the feature
st.markdown("---")
st.markdown("""
### About Mobile Fraud Alerts

This feature enables real-time fraud detection and notification via SMS, allowing users to:

- Receive immediate alerts for suspicious transactions
- Confirm or deny transactions directly via SMS reply
- Get security alerts for unusual account activity
- Configure biometric verification for high-risk transactions
- Manage trusted devices and security settings

For demonstration purposes, alerts are simulated and displayed in the notification history.
In a production environment, these would be triggered automatically by the risk analytics system.
""")

# Refresh button
if st.sidebar.button("Refresh Data"):
    st.rerun()

# Last updated time
st.sidebar.markdown("---")
st.sidebar.text(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")