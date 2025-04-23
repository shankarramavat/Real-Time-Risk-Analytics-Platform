import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import json

from utils.data_generator import load_data
from utils.notification_service import (
    send_transaction_alert, 
    get_notification_history
)
from utils.session_helper import initialize_session_state

# Page config
st.set_page_config(page_title="Mobile Notifications", page_icon="📱", layout="wide")

# Initialize session state
initialize_session_state()

st.title("Mobile Fraud Alerts & Notifications")

# Load transaction data
df_transactions = load_data("transactions")

# Create tabs
tabs = st.tabs([
    "Notification Dashboard",
    "Test Alert System",
    "Notification History",
    "User Preferences"
])

# Notification Dashboard Tab
with tabs[0]:
    st.header("Notification Dashboard")
    
    # Metrics Section
    st.subheader("Notification Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Create some sample metrics
    notification_history = get_notification_history()
    alerts_sent = len(notification_history)
    
    with col1:
        st.metric("Alerts Sent", alerts_sent)
    
    with col2:
        user_count = len(st.session_state.user_profiles) if 'user_profiles' in st.session_state else 0
        st.metric("Registered Users", user_count)
    
    with col3:
        # Calculate engagement rate based on responses to notifications (simulated)
        engagement_rate = np.random.uniform(65, 95) if alerts_sent > 0 else 0
        st.metric("User Engagement", f"{engagement_rate:.1f}%")
    
    with col4:
        # False positive rate (simulated)
        false_positive_rate = np.random.uniform(2, 15) if alerts_sent > 0 else 0
        st.metric("False Positive Rate", f"{false_positive_rate:.1f}%")
    
    # Charts and Analytics
    st.subheader("Notification Analytics")
    
    # Create sample notification data if history is empty
    if not notification_history:
        # Generate sample notification history for demonstration
        sample_history = []
        np.random.seed(42)  # For reproducibility
        
        # Generate notifications over the past 30 days
        start_date = datetime.now() - timedelta(days=30)
        
        # Alert types
        alert_types = ["Fraud Alert", "Suspicious Transaction", "Account Security", "Login Attempt"]
        alert_weights = [0.4, 0.3, 0.2, 0.1]  # Relative frequencies
        
        # Generate random alerts
        for i in range(50):
            # Random date within the past 30 days
            alert_date = start_date + timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            
            # Random alert type
            alert_type = np.random.choice(alert_types, p=alert_weights)
            
            # Random device
            device = np.random.choice(["Mobile", "Web", "ATM", "POS Terminal"], p=[0.6, 0.2, 0.1, 0.1])
            
            # Random amount (for transaction alerts)
            amount = np.random.uniform(10, 5000) if alert_type in ["Fraud Alert", "Suspicious Transaction"] else None
            
            # Random user
            user_id = np.random.randint(1, 3)  # Assuming we have 2 users in session_state.user_profiles
            
            # Response (user interaction with alert)
            response = np.random.choice(["Confirmed", "Denied", "No Response"], p=[0.7, 0.2, 0.1])
            
            # Create alert record
            alert = {
                "id": f"ALT{i+1000}",
                "timestamp": alert_date.strftime('%Y-%m-%d %H:%M:%S'),
                "type": alert_type,
                "user_id": user_id,
                "user_name": f"User {user_id}",
                "device": device,
                "amount": amount,
                "message": f"Sample {alert_type.lower()} message for demonstration",
                "response": response,
                "medium": "SMS"
            }
            
            sample_history.append(alert)
        
        # Sort by timestamp (most recent first)
        sample_history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Store in session state
        notification_history = sample_history
        st.session_state.notification_history = notification_history
    
    # Create a DataFrame from notification history
    if notification_history:
        df_notifications = pd.DataFrame(notification_history)
        if 'timestamp' in df_notifications.columns:
            df_notifications['timestamp'] = pd.to_datetime(df_notifications['timestamp'])
            df_notifications['date'] = df_notifications['timestamp'].dt.date
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Alerts by date
                alerts_by_date = df_notifications.groupby('date').size().reset_index(name='count')
                alerts_by_date['date'] = pd.to_datetime(alerts_by_date['date'])
                
                fig = px.line(
                    alerts_by_date,
                    x='date',
                    y='count',
                    title="Alerts Sent by Date",
                    labels={'date': 'Date', 'count': 'Number of Alerts'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Alerts by type
                alerts_by_type = df_notifications.groupby('type').size().reset_index(name='count')
                
                fig = px.pie(
                    alerts_by_type,
                    values='count',
                    names='type',
                    title="Alerts by Type",
                    hole=0.4
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # User response analysis
            if 'response' in df_notifications.columns:
                st.subheader("User Response Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Response by type
                    response_by_type = df_notifications.groupby(['type', 'response']).size().reset_index(name='count')
                    
                    fig = px.bar(
                        response_by_type,
                        x='type',
                        y='count',
                        color='response',
                        title="User Response by Alert Type",
                        labels={'type': 'Alert Type', 'count': 'Count', 'response': 'Response'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Response time (simulated)
                    st.markdown("### Response Time Distribution")
                    
                    # Create sample response times
                    np.random.seed(42)
                    response_times = []
                    
                    for i in range(100):
                        # Most responses come in quickly, with a long tail
                        time_mins = np.random.exponential(5)
                        response_times.append(min(60, time_mins))  # Cap at 60 minutes
                    
                    fig = px.histogram(
                        response_times,
                        nbins=20,
                        title="Time to User Response (minutes)",
                        labels={'value': 'Response Time (minutes)', 'count': 'Number of Alerts'},
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No notification history available. Send a test alert to generate data.")

# Test Alert System Tab
with tabs[1]:
    st.header("Test Alert System")
    
    st.markdown("""
    Use this interface to test the fraud alert system. You can send a test alert to your phone number 
    to see how the alert would appear in a real situation.
    """)
    
    # User selection
    if 'user_profiles' not in st.session_state:
        # Create sample user profiles if not exist
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
                "last_login": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
                "last_login": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "devices": ["Samsung Galaxy S22", "Windows PC"],
                "location": "Chicago, USA"
            }
        ]
    
    selected_user_index = st.selectbox(
        "Select User",
        options=range(len(st.session_state.user_profiles)),
        format_func=lambda x: st.session_state.user_profiles[x]["name"]
    )
    
    selected_user = st.session_state.user_profiles[selected_user_index]
    
    # Display user info
    st.subheader("User Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Name:** {selected_user['name']}")
        st.markdown(f"**Phone:** {selected_user['phone']}")
        st.markdown(f"**Email:** {selected_user['email']}")
    
    with col2:
        st.markdown(f"**Default Location:** {selected_user.get('location', 'Unknown')}")
        st.markdown(f"**Devices:** {', '.join(selected_user.get('devices', ['Unknown']))}")
        st.markdown(f"**Transaction Alert Threshold:** ${selected_user.get('transaction_threshold', 1000):,.2f}")
    
    # Transaction selection
    st.subheader("Select Transaction for Alert")
    
    # Option for random transaction or custom
    transaction_option = st.radio(
        "Transaction Option",
        options=["Select from Recent Transactions", "Create Custom Transaction"]
    )
    
    selected_transaction = None
    
    if transaction_option == "Select from Recent Transactions":
        # Filter high-risk transactions
        high_risk_transactions = df_transactions[df_transactions['risk_score'] > 0.6].head(10)
        
        if not high_risk_transactions.empty:
            transaction_index = st.selectbox(
                "Select High-Risk Transaction",
                options=range(len(high_risk_transactions)),
                format_func=lambda x: f"{high_risk_transactions.iloc[x]['transaction_type']} - {high_risk_transactions.iloc[x]['counterparty_name']} - ${high_risk_transactions.iloc[x]['amount']:,.2f}"
            )
            
            selected_transaction = high_risk_transactions.iloc[transaction_index].to_dict()
        else:
            st.warning("No high-risk transactions found in the dataset.")
    else:
        # Custom transaction form
        st.subheader("Create Custom Transaction Alert")
        
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_type = st.selectbox(
                "Transaction Type",
                options=["Card Payment", "Wire Transfer", "ATM Withdrawal", "Online Purchase", "Mobile Payment"]
            )
            
            merchant = st.text_input("Merchant/Recipient", value="Example Merchant")
            
            location = st.text_input("Transaction Location", value="New York, USA")
        
        with col2:
            amount = st.number_input("Amount", min_value=10.0, max_value=10000.0, value=1500.0, step=100.0)
            
            currency = st.selectbox("Currency", options=["USD", "EUR", "GBP", "JPY", "CAD"])
            
            risk_score = st.slider("Risk Score", min_value=0.0, max_value=1.0, value=0.85, step=0.05)
        
        # Create custom transaction
        if merchant and amount > 0:
            selected_transaction = {
                "transaction_id": f"TX{np.random.randint(10000, 99999)}",
                "transaction_type": transaction_type,
                "counterparty_name": merchant,
                "amount": amount,
                "currency": currency,
                "location": location,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "risk_score": risk_score
            }
    
    # Display selected transaction
    if selected_transaction:
        st.subheader("Transaction Details")
        
        # Create card-like display for transaction
        st.markdown(
            f"""
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <h3 style="color: #ff4b4b;">{selected_transaction.get('transaction_type', 'Transaction')}</h3>
                <table style="width: 100%;">
                    <tr>
                        <td style="padding: 8px 0; width: 40%;"><strong>Merchant/Recipient:</strong></td>
                        <td style="padding: 8px 0;">{selected_transaction.get('counterparty_name', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0;"><strong>Amount:</strong></td>
                        <td style="padding: 8px 0;">${selected_transaction.get('amount', 0):,.2f} {selected_transaction.get('currency', 'USD')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0;"><strong>Date/Time:</strong></td>
                        <td style="padding: 8px 0;">{selected_transaction.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0;"><strong>Location:</strong></td>
                        <td style="padding: 8px 0;">{selected_transaction.get('location', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0;"><strong>Risk Score:</strong></td>
                        <td style="padding: 8px 0;">
                            <div style="background-color: #f0f0f0; border-radius: 3px; height: 20px; width: 100%;">
                                <div style="background-color: {'#ff4b4b' if selected_transaction.get('risk_score', 0) > 0.7 else '#ffae00' if selected_transaction.get('risk_score', 0) > 0.4 else '#4bb543'}; width: {int(selected_transaction.get('risk_score', 0) * 100)}%; height: 100%; border-radius: 3px;">
                                </div>
                            </div>
                            <div style="text-align: center;">{selected_transaction.get('risk_score', 0):.2f}</div>
                        </td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Option to customize the alert message
        st.subheader("Customize Alert Message")
        
        default_message = f"ALERT: Suspicious {selected_transaction.get('transaction_type', 'transaction')} of ${selected_transaction.get('amount', 0):,.2f} at {selected_transaction.get('counterparty_name', 'Unknown')}. Reply YES to confirm or NO to deny this transaction."
        
        alert_message = st.text_area("Alert Message", value=default_message, height=100)
        
        # Custom phone number field
        use_custom_phone = st.checkbox("Use custom phone number instead of user's phone")
        phone_number = None
        
        if use_custom_phone:
            phone_number = st.text_input(
                "Phone Number (E.164 format)",
                value="+12345678900",
                help="Enter phone number in E.164 format (e.g., +12345678900)"
            )
            
            # Validate phone number format
            if phone_number:
                if not re.match(r'^\+[1-9]\d{1,14}$', phone_number):
                    st.error("Please enter a valid phone number in E.164 format (e.g., +12345678900)")
                    phone_number = None
        else:
            phone_number = selected_user["phone"]
        
        # Send alert button
        if st.button("Send Test Alert"):
            if selected_transaction and phone_number:
                # Add message to transaction
                selected_transaction["alert_message"] = alert_message
                
                # Send transaction alert
                with st.spinner("Sending alert..."):
                    result = send_transaction_alert(selected_transaction, phone_number)
                
                if result["success"]:
                    st.success(f"Alert sent successfully! Message ID: {result.get('message_id', 'N/A')}")
                    st.info("Note: If Twilio credentials are not configured, the alert is saved to notification history for demonstration purposes.")
                else:
                    st.error(f"Failed to send alert: {result.get('message', 'Unknown error')}")
                    
                    if "Twilio credentials not configured" in result.get('message', ''):
                        st.info("Alert saved to notification history for demonstration purposes.")
            else:
                st.error("Please select a transaction and provide a valid phone number.")

# Notification History Tab
with tabs[2]:
    st.header("Notification History")
    
    # Get notification history
    notification_history = get_notification_history()
    
    if notification_history:
        # Convert to DataFrame for filtering and display
        df_history = pd.DataFrame(notification_history)
        
        # Add filter controls
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # Filter by notification type
            if 'type' in df_history.columns:
                available_types = sorted(df_history['type'].unique())
                selected_types = st.multiselect("Notification Type", options=available_types, default=available_types)
            else:
                selected_types = []
        
        with filter_col2:
            # Filter by date range
            if 'timestamp' in df_history.columns:
                df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
                min_date = df_history['timestamp'].min().date()
                max_date = df_history['timestamp'].max().date()
                
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            else:
                date_range = None
        
        with filter_col3:
            # Filter by user (if multiple users)
            if 'user_name' in df_history.columns:
                available_users = sorted(df_history['user_name'].unique())
                selected_users = st.multiselect("User", options=available_users, default=available_users)
            else:
                selected_users = []
        
        # Apply filters
        filtered_df = df_history.copy()
        
        if selected_types:
            filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
        
        if date_range and len(date_range) == 2 and 'timestamp' in filtered_df.columns:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= start_date) & 
                (filtered_df['timestamp'].dt.date <= end_date)
            ]
        
        if selected_users:
            filtered_df = filtered_df[filtered_df['user_name'].isin(selected_users)]
        
        # Display filtered history
        if not filtered_df.empty:
            # Format for display
            display_columns = ['timestamp', 'type', 'user_name', 'message', 'response']
            display_columns = [col for col in display_columns if col in filtered_df.columns]
            
            display_df = filtered_df[display_columns].copy()
            
            # Rename columns for display
            column_names = {
                'timestamp': 'Timestamp',
                'type': 'Type',
                'user_name': 'User',
                'message': 'Message',
                'response': 'Response'
            }
            
            display_df = display_df.rename(columns={col: column_names.get(col, col) for col in display_df.columns})
            
            # Sort by timestamp (most recent first)
            if 'Timestamp' in display_df.columns:
                display_df = display_df.sort_values('Timestamp', ascending=False)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Export button
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Export Filtered History",
                csv,
                "notification_history.csv",
                "text/csv",
                key="download-csv"
            )
        else:
            st.info("No notifications match the selected filters.")
    else:
        st.info("No notification history available. Send a test alert to generate data.")

# User Preferences Tab
with tabs[3]:
    st.header("User Notification Preferences")
    
    # User selection
    selected_user_index = st.selectbox(
        "Select User",
        options=range(len(st.session_state.user_profiles)),
        format_func=lambda x: st.session_state.user_profiles[x]["name"],
        key="user_prefs_select"
    )
    
    selected_user = st.session_state.user_profiles[selected_user_index]
    
    # Create columns for preferences
    pref_col1, pref_col2 = st.columns(2)
    
    with pref_col1:
        st.subheader("Contact Information")
        
        # Phone number
        phone = st.text_input(
            "Phone Number (E.164 format)",
            value=selected_user.get("phone", ""),
            help="Enter phone number in E.164 format (e.g., +12345678900)"
        )
        
        # Email
        email = st.text_input(
            "Email Address",
            value=selected_user.get("email", "")
        )
        
        # Preferred notification channels
        st.subheader("Notification Channels")
        
        notification_prefs = selected_user.get("notification_preferences", {
            "sms": True,
            "email": True,
            "push": False
        })
        
        sms_notifications = st.checkbox("SMS Notifications", value=notification_prefs.get("sms", True))
        email_notifications = st.checkbox("Email Notifications", value=notification_prefs.get("email", True))
        push_notifications = st.checkbox("Push Notifications", value=notification_prefs.get("push", False))
    
    with pref_col2:
        st.subheader("Alert Settings")
        
        # Transaction threshold
        transaction_threshold = st.number_input(
            "Transaction Alert Threshold ($)",
            min_value=0.0,
            max_value=10000.0,
            value=float(selected_user.get("transaction_threshold", 1000)),
            step=100.0,
            help="You will be alerted for transactions above this amount"
        )
        
        # Biometric verification
        biometric_enabled = st.checkbox(
            "Enable Biometric Verification",
            value=selected_user.get("biometric_enabled", False),
            help="Require biometric verification for high-risk transactions"
        )
        
        # Alert categories
        st.subheader("Alert Categories")
        
        alert_categories = selected_user.get("alert_categories", {
            "suspicious_transactions": True,
            "login_attempts": True,
            "account_changes": True,
            "security_issues": True
        })
        
        suspicious_transactions = st.checkbox(
            "Suspicious Transactions",
            value=alert_categories.get("suspicious_transactions", True)
        )
        
        login_attempts = st.checkbox(
            "Login Attempts",
            value=alert_categories.get("login_attempts", True)
        )
        
        account_changes = st.checkbox(
            "Account Changes",
            value=alert_categories.get("account_changes", True)
        )
        
        security_issues = st.checkbox(
            "Security Issues",
            value=alert_categories.get("security_issues", True)
        )
    
    # Save preferences button
    if st.button("Save Preferences"):
        # Update user preferences
        selected_user["phone"] = phone
        selected_user["email"] = email
        selected_user["notification_preferences"] = {
            "sms": sms_notifications,
            "email": email_notifications,
            "push": push_notifications
        }
        selected_user["transaction_threshold"] = transaction_threshold
        selected_user["biometric_enabled"] = biometric_enabled
        selected_user["alert_categories"] = {
            "suspicious_transactions": suspicious_transactions,
            "login_attempts": login_attempts,
            "account_changes": account_changes,
            "security_issues": security_issues
        }
        
        # Update user in session state
        st.session_state.user_profiles[selected_user_index] = selected_user
        
        st.success("User preferences saved successfully!")

# Information section
st.markdown("---")
st.markdown("""
### About Mobile Fraud Alerts

This feature provides real-time notifications for suspicious activity, allowing users to:

1. Receive instant SMS alerts for potentially fraudulent transactions
2. Respond directly via text message to confirm or deny transactions
3. Set custom alert thresholds and preferences
4. Enable biometric verification for high-risk activities

For demonstration purposes, the alerts are simulated. In a production environment, these 
would be sent using Twilio's SMS API to the user's actual phone number.
""")

# Last updated time
st.sidebar.markdown("---")
st.sidebar.text(f"Last updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")