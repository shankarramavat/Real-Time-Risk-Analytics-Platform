import os
from twilio.rest import Client
import streamlit as st
import pandas as pd
import json
from datetime import datetime

# Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None

def send_sms_alert(to_phone_number, message):
    """
    Send SMS alert using Twilio
    
    Parameters:
    - to_phone_number: Recipient's phone number (must be in E.164 format, e.g., +12345678900)
    - message: SMS content
    
    Returns:
    - Dict with status and details
    """
    result = {
        "success": False,
        "message": "",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "to": to_phone_number,
        "content": message
    }
    
    if not client or not TWILIO_PHONE_NUMBER:
        result["message"] = "Twilio credentials not configured. SMS not sent."
        return result
    
    try:
        # Send SMS using Twilio
        twilio_message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_phone_number
        )
        
        result["success"] = True
        result["message"] = f"SMS sent successfully. SID: {twilio_message.sid}"
        
    except Exception as e:
        result["message"] = f"Error sending SMS: {str(e)}"
    
    return result

def generate_transaction_alert_message(transaction_data):
    """
    Generate fraud alert message for a suspicious transaction
    
    Parameters:
    - transaction_data: Dict containing transaction details
    
    Returns:
    - Alert message (str)
    """
    amount = transaction_data.get('amount', 0)
    currency = transaction_data.get('currency', 'USD')
    merchant = transaction_data.get('counterparty_name', 'Unknown Merchant')
    txn_type = transaction_data.get('transaction_type', 'transaction')
    txn_id = transaction_data.get('id', '0000')
    
    message = (
        f"ALERT: Did you authorize a {currency} {amount:,.2f} {txn_type} with {merchant}? "
        f"Reply YES {txn_id} to confirm or NO {txn_id} to report fraud."
    )
    
    return message

def save_notification_to_history(notification_data):
    """
    Save notification data to history for tracking
    
    Parameters:
    - notification_data: Dict containing notification details
    """
    if 'notification_history' not in st.session_state:
        st.session_state.notification_history = []
    
    st.session_state.notification_history.append(notification_data)
    
    # Limit history size to prevent memory issues
    if len(st.session_state.notification_history) > 100:
        st.session_state.notification_history = st.session_state.notification_history[-100:]

def get_notification_history():
    """
    Get notification history from session state
    
    Returns:
    - List of notification records
    """
    if 'notification_history' not in st.session_state:
        st.session_state.notification_history = []
    
    return st.session_state.notification_history

def send_transaction_alert(transaction_data, phone_number):
    """
    Send transaction alert and save to history
    
    Parameters:
    - transaction_data: Dict containing transaction details
    - phone_number: Recipient's phone number
    
    Returns:
    - Dict with status and details
    """
    message = generate_transaction_alert_message(transaction_data)
    result = send_sms_alert(phone_number, message)
    
    # Add transaction data to the result for history
    result["transaction_data"] = transaction_data
    result["alert_type"] = "transaction"
    
    # Save to history
    save_notification_to_history(result)
    
    return result

def send_account_security_alert(alert_data, phone_number):
    """
    Send account security alert and save to history
    
    Parameters:
    - alert_data: Dict containing alert details
    - phone_number: Recipient's phone number
    
    Returns:
    - Dict with status and details
    """
    device = alert_data.get('device', 'unknown device')
    location = alert_data.get('location', 'unknown location')
    time = alert_data.get('time', 'recently')
    alert_id = alert_data.get('id', '0000')
    
    message = (
        f"SECURITY ALERT: New login to your account from {device} in {location} at {time}. "
        f"Reply YES {alert_id} if this was you or NO {alert_id} to secure your account."
    )
    
    result = send_sms_alert(phone_number, message)
    
    # Add alert data to the result for history
    result["alert_data"] = alert_data
    result["alert_type"] = "security"
    
    # Save to history
    save_notification_to_history(result)
    
    return result