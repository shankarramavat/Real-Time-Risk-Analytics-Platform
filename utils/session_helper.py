import streamlit as st
from datetime import datetime

def initialize_session_state():
    """
    Initialize all session state variables needed across pages
    """
    # Core session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.last_updated = datetime.now()
        st.session_state.alert_count = 0
    
    # Ensure last_updated exists for all pages
    if 'last_updated' not in st.session_state:
        st.session_state.last_updated = datetime.now()
    
    # Notification system
    if 'notification_history' not in st.session_state:
        st.session_state.notification_history = []
    
    # User profiles for mobile/ATO features
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
    
    # Login history for account security
    if 'login_history' not in st.session_state:
        st.session_state.login_history = []