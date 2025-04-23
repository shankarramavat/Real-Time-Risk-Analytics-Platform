import os
from twilio.rest import Client
import streamlit as st
import pandas as pd
from datetime import datetime
import json

def send_sms_alert(to_phone_number, message):
    """
    Send SMS alert using Twilio
    
    Parameters:
    - to_phone_number: Recipient's phone number (must be in E.164 format, e.g., +12345678900)
    - message: SMS content
    
    Returns:
    - Dict with status and details
    """
    # Get Twilio credentials from environment variables
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_phone = os.environ.get("TWILIO_PHONE_NUMBER")
    
    result = {
        "success": False,
        "message": "Failed to send SMS"
    }
    
    # Check if Twilio credentials are configured
    if not account_sid or not auth_token or not from_phone:
        result["message"] = "Twilio credentials not configured. Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER environment variables."
        return result
    
    try:
        # Initialize Twilio client
        client = Client(account_sid, auth_token)
        
        # Send the SMS
        sms = client.messages.create(
            body=message,
            from_=from_phone,
            to=to_phone_number
        )
        
        # Return success response
        result["success"] = True
        result["message"] = "SMS sent successfully"
        result["message_id"] = sms.sid
        
        return result
    except Exception as e:
        # Return error response
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
    # Extract transaction details
    transaction_type = transaction_data.get("transaction_type", "Transaction")
    merchant = transaction_data.get("counterparty_name", "Unknown Merchant")
    amount = transaction_data.get("amount", 0)
    currency = transaction_data.get("currency", "USD")
    location = transaction_data.get("location", "Unknown Location")
    timestamp = transaction_data.get("timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Generate message
    message = f"FRAUD ALERT: {transaction_type} of {amount:.2f} {currency} at {merchant} ({location}) on {timestamp}. "
    message += "Reply YES to confirm or NO to deny this transaction."
    
    return message

def save_notification_to_history(notification_data):
    """
    Save notification data to history for tracking
    
    Parameters:
    - notification_data: Dict containing notification details
    """
    # Initialize notification history if not exists
    if 'notification_history' not in st.session_state:
        st.session_state.notification_history = []
    
    # Add notification to history
    st.session_state.notification_history.append(notification_data)

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
    # Generate alert message if not provided
    message = transaction_data.get("alert_message")
    if not message:
        message = generate_transaction_alert_message(transaction_data)
    
    # Send SMS alert
    result = send_sms_alert(phone_number, message)
    
    # Save to notification history
    notification_data = {
        "id": f"TXA{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "type": "Fraud Alert",
        "user_id": transaction_data.get("user_id", "Unknown"),
        "user_name": transaction_data.get("user_name", "Unknown User"),
        "message": message,
        "transaction_id": transaction_data.get("transaction_id", "Unknown"),
        "transaction_type": transaction_data.get("transaction_type", "Unknown"),
        "amount": transaction_data.get("amount", 0),
        "currency": transaction_data.get("currency", "USD"),
        "merchant": transaction_data.get("counterparty_name", "Unknown Merchant"),
        "risk_score": transaction_data.get("risk_score", 0),
        "medium": "SMS",
        "recipient": phone_number,
        "status": "Sent" if result["success"] else "Failed",
        "response": "Pending"
    }
    
    save_notification_to_history(notification_data)
    
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
    # Create alert message
    alert_type = alert_data.get("type", "Security Alert")
    device = alert_data.get("device", "Unknown device")
    location = alert_data.get("location", "Unknown location")
    time = alert_data.get("time", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    message = f"SECURITY ALERT: {alert_type} detected from {device} in {location} at {time}. "
    message += "If this wasn't you, reply BLOCK to lock your account or call customer service immediately."
    
    # Send SMS alert
    result = send_sms_alert(phone_number, message)
    
    # Save to notification history
    notification_data = {
        "id": f"SEC{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "type": "Account Security",
        "user_id": alert_data.get("user_id", "Unknown"),
        "user_name": alert_data.get("user_name", "Unknown User"),
        "message": message,
        "device": device,
        "location": location,
        "risk_score": alert_data.get("risk_score", 0),
        "medium": "SMS",
        "recipient": phone_number,
        "status": "Sent" if result["success"] else "Failed",
        "response": "Pending"
    }
    
    save_notification_to_history(notification_data)
    
    return result