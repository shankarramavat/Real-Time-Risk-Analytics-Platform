import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import re

from utils.data_generator import load_data
from utils.account_security import (
    check_ip_reputation, 
    analyze_login_behavior,
    analyze_keystroke_dynamics,
    detect_account_takeover
)
from utils.notification_service import send_account_security_alert
from utils.session_helper import initialize_session_state

# Page config
st.set_page_config(page_title="Account Security", page_icon="🔐", layout="wide")

# Initialize session state
initialize_session_state()

st.title("Account Takeover (ATO) Detection")

# Load user data
if 'user_profiles' not in st.session_state:
    st.error("User profiles not found. Please visit the Mobile Notifications page first.")
    st.stop()

# Initialize login history if not exists
if 'login_history' not in st.session_state:
    # Generate sample login history
    np.random.seed(42)
    
    login_history = []
    
    # For each user
    for user in st.session_state.user_profiles:
        user_id = user["id"]
        user_name = user["name"]
        
        # Generate 20 login events per user
        for i in range(20):
            # Random date within the last 30 days
            login_date = datetime.now() - timedelta(days=np.random.randint(0, 30), 
                                                   hours=np.random.randint(0, 24),
                                                   minutes=np.random.randint(0, 60))
            
            # Usually login from known device
            device = np.random.choice(user.get("devices", ["Unknown Device"]) + ["Unknown Device"], 
                                      p=[0.45, 0.45, 0.1] if len(user.get("devices", [])) == 2 else [0.9, 0.1])
            
            # Usually login from typical location
            typical_location = user.get("location", "Unknown Location")
            unusual_locations = ["Chicago, USA", "Dallas, USA", "Miami, USA", "London, UK", "Paris, France", 
                               "Tokyo, Japan", "Sydney, Australia", "Unknown Location"]
            unusual_locations = [loc for loc in unusual_locations if loc != typical_location]
            
            location = np.random.choice([typical_location] + unusual_locations, 
                                      p=[0.85] + [0.15/len(unusual_locations)] * len(unusual_locations))
            
            # Generate IP based on location
            ip_base = {
                "New York, USA": "101.123",
                "Chicago, USA": "192.168",
                "Dallas, USA": "172.16",
                "Miami, USA": "203.0",
                "London, UK": "82.5",
                "Paris, France": "91.198",
                "Tokyo, Japan": "116.58",
                "Sydney, Australia": "1.152",
            }.get(location, "45.85")
            
            ip_address = f"{ip_base}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}"
            
            # Rarely from TOR or malicious IP
            from utils.account_security import TOR_EXIT_NODES, MALICIOUS_IPS
            if np.random.random() < 0.05:  # 5% chance
                ip_address = np.random.choice(TOR_EXIT_NODES + MALICIOUS_IPS)
            
            # Browser
            browsers = ["Chrome", "Firefox", "Safari", "Edge", "Unknown"]
            browser = np.random.choice(browsers, p=[0.5, 0.2, 0.2, 0.05, 0.05])
            
            # Success rate
            success = np.random.random() < 0.9  # 90% success rate
            
            # Create login record
            login_record = {
                "user_id": user_id,
                "user_name": user_name,
                "timestamp": login_date.strftime('%Y-%m-%d %H:%M:%S'),
                "ip_address": ip_address,
                "device": device,
                "browser": browser,
                "location": location,
                "success": success,
                "failure_reason": "Invalid credentials" if not success else None,
                "session_id": f"SES{np.random.randint(10000, 99999)}",
                "risk_score": np.random.uniform(0.1, 0.9) if not success or np.random.random() < 0.2 else np.random.uniform(0.1, 0.4)
            }
            
            login_history.append(login_record)
    
    # Sort by timestamp (most recent first)
    login_history.sort(key=lambda x: x["timestamp"], reverse=True)
    
    # Store in session state
    st.session_state.login_history = login_history

# Sidebar - User selection
st.sidebar.header("User Selection")

# User selection
selected_user_index = st.sidebar.selectbox(
    "Select User",
    options=range(len(st.session_state.user_profiles)),
    format_func=lambda x: st.session_state.user_profiles[x]["name"]
)
selected_user = st.session_state.user_profiles[selected_user_index]

# Filter login history for selected user
user_login_history = [login for login in st.session_state.login_history if login["user_id"] == selected_user["id"]]

# Main content - Tabs
tabs = st.tabs([
    "Login Security Dashboard",
    "Account Takeover Simulation",
    "Login History",
    "Security Settings"
])

# Login Security Dashboard tab
with tabs[0]:
    st.header("Login Security Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Failed logins in the past 7 days
    week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    recent_logins = [login for login in user_login_history 
                    if login["timestamp"] > week_ago]
    
    failed_logins = len([login for login in recent_logins if not login["success"]])
    suspicious_logins = len([login for login in recent_logins 
                           if login["risk_score"] > 0.6])
    
    # Unique login locations
    unique_locations = set(login["location"] for login in recent_logins)
    
    # Unique devices
    unique_devices = set(login["device"] for login in recent_logins)
    
    with col1:
        st.metric("Total Logins (7d)", len(recent_logins))
    
    with col2:
        st.metric("Failed Logins", failed_logins, 
                 delta=f"{failed_logins/len(recent_logins)*100:.1f}%" if recent_logins else "0%",
                 delta_color="inverse")
    
    with col3:
        st.metric("Suspicious Logins", suspicious_logins,
                 delta=f"{suspicious_logins/len(recent_logins)*100:.1f}%" if recent_logins else "0%",
                 delta_color="inverse")
    
    with col4:
        st.metric("Unique Locations", len(unique_locations))
    
    # Login activity charts
    st.subheader("Login Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Convert timestamps to datetime
        login_df = pd.DataFrame(recent_logins)
        if "timestamp" in login_df.columns and len(login_df) > 0:
            login_df["timestamp"] = pd.to_datetime(login_df["timestamp"])
            login_df["date"] = login_df["timestamp"].dt.date
            
            # Count logins by date and success
            login_counts = login_df.groupby(["date", "success"]).size().reset_index(name="count")
        else:
            # Create empty dataframe with required columns
            login_counts = pd.DataFrame(columns=["date", "success", "count"])
        
        # Create success/failure chart
        login_fig = px.bar(
            login_counts,
            x="date",
            y="count",
            color="success",
            color_discrete_map={True: "green", False: "red"},
            title="Login Attempts by Day",
            labels={"success": "Status", "count": "Number of Logins", "date": "Date"},
            category_orders={"success": [True, False]}
        )
        
        login_fig.update_layout(legend=dict(
            title="Login Status",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        
        st.plotly_chart(login_fig, use_container_width=True)
    
    with col2:
        # Login risk distribution
        risk_fig = px.histogram(
            login_df,
            x="risk_score",
            nbins=20,
            title="Login Risk Score Distribution",
            labels={"risk_score": "Risk Score", "count": "Number of Logins"}
        )
        
        # Add a vertical line for high risk threshold
        risk_fig.add_vline(
            x=0.6,
            line_dash="dash",
            line_color="red",
            annotation_text="High Risk Threshold"
        )
        
        st.plotly_chart(risk_fig, use_container_width=True)
    
    # Geo distribution of logins
    st.subheader("Geographic Distribution of Logins")
    
    if len(login_df) > 0 and "location" in login_df.columns:
        # Count logins by location
        location_counts = login_df.groupby("location").size().reset_index(name="count")
        location_success = login_df.groupby("location")["success"].mean().reset_index(name="success_rate")
        location_risk = login_df.groupby("location")["risk_score"].mean().reset_index(name="avg_risk")
        
        # Merge the dataframes
        location_stats = pd.merge(location_counts, location_success, on="location")
        location_stats = pd.merge(location_stats, location_risk, on="location")
    else:
        # Create empty dataframe with required columns
        location_stats = pd.DataFrame(columns=["location", "count", "success_rate", "avg_risk"])
    
    # Create location map
    loc_fig = px.scatter_geo(
        location_stats,
        locationmode="country names",
        # In a real app, these would be actual lat/lon
        # For demo, we'll use a simplified approach
        lat=[40.7, 41.9, 32.8, 25.8, 51.5, 48.9, 35.7, -33.9] if len(location_stats) > 1 else [40.7],
        lon=[-74.0, -87.6, -96.8, -80.2, -0.1, 2.3, 139.8, 151.2] if len(location_stats) > 1 else [-74.0],
        hover_name="location",
        size="count",
        color="avg_risk",
        color_continuous_scale="RdYlGn_r",
        size_max=30,
        hover_data={"count": True, "success_rate": True, "avg_risk": True}
    )
    
    loc_fig.update_layout(
        title="Login Locations",
        geo=dict(
            showland=True,
            landcolor="rgb(217, 217, 217)",
            showocean=True,
            oceancolor="rgb(204, 229, 255)",
            showcountries=True,
            countrycolor="rgb(150, 150, 150)"
        )
    )
    
    st.plotly_chart(loc_fig, use_container_width=True)
    
    # Device distribution
    st.subheader("Login Devices")
    
    if len(login_df) > 0 and "device" in login_df.columns:
        # Count logins by device
        device_counts = login_df.groupby("device").size().reset_index(name="count")
        device_success = login_df.groupby("device")["success"].mean().reset_index(name="success_rate")
        device_risk = login_df.groupby("device")["risk_score"].mean().reset_index(name="avg_risk")
        
        # Merge the dataframes
        device_stats = pd.merge(device_counts, device_success, on="device")
        device_stats = pd.merge(device_stats, device_risk, on="device")
        
        # Sort by count (descending)
        device_stats = device_stats.sort_values("count", ascending=False)
    else:
        # Create empty dataframe with required columns
        device_stats = pd.DataFrame(columns=["device", "count", "success_rate", "avg_risk"])
    
    # Create device bar chart
    device_fig = px.bar(
        device_stats,
        x="device",
        y="count",
        color="avg_risk",
        hover_data=["success_rate", "avg_risk"],
        color_continuous_scale="RdYlGn_r",
        title="Login Devices"
    )
    
    st.plotly_chart(device_fig, use_container_width=True)

# Account Takeover Simulation tab
with tabs[1]:
    st.header("Account Takeover Simulation")
    
    st.markdown("""
    This simulation allows you to test the account takeover detection system with different scenarios.
    In a real-world implementation, these checks would be performed automatically on every login attempt.
    """)
    
    # Create columns for simulation form
    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        # Device selection
        sim_device_options = selected_user.get("devices", []) + ["Unknown Device", "New iPhone", "New Android"]
        sim_device = st.selectbox("Device", options=sim_device_options)
        
        # Location selection
        typical_location = selected_user.get("location", "New York, USA")
        sim_location_options = [typical_location, "Chicago, USA", "London, UK", "Tokyo, Japan", 
                              "Sydney, Australia", "Moscow, Russia", "Unknown Location"]
        sim_location = st.selectbox("Location", options=sim_location_options)
        
        # IP Address
        sim_ip_options = {
            "Normal IP": "101.123.45.67",
            "Datacenter IP": "192.168.1.1",
            "TOR Exit Node": "185.220.101.21",
            "Known Malicious IP": "45.132.192.15"
        }
        sim_ip_type = st.selectbox("IP Address Type", options=list(sim_ip_options.keys()))
        sim_ip = sim_ip_options[sim_ip_type]
        
        # Time of login
        current_time = datetime.now()
        sim_time_options = [
            ("Current time", current_time),
            ("Business hours", current_time.replace(hour=14, minute=30)),
            ("Late night", current_time.replace(hour=3, minute=15)),
            ("Weekend", current_time + timedelta(days=(5 - current_time.weekday()) % 7))
        ]
        sim_time_selection = st.selectbox(
            "Login Time", 
            options=range(len(sim_time_options)),
            format_func=lambda x: sim_time_options[x][0]
        )
        sim_time = sim_time_options[sim_time_selection][1]
        
    with sim_col2:
        # Additional risk factors
        st.subheader("Account Activity")
        
        sim_password_changed = st.checkbox("Password Changed")
        sim_financial_activity = st.checkbox("Financial Transaction")
        sim_security_settings = st.checkbox("Security Settings Changed")
        sim_added_device = st.checkbox("Added New Device")
        
        # Typing pattern match
        st.subheader("Behavioral Biometrics")
        sim_typing_match = st.slider(
            "Typing Pattern Match Score", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.8,
            step=0.05,
            help="Higher values mean typing patterns match the user's profile"
        )
        
        # Login history context
        st.subheader("Login History Context")
        sim_frequent_location = st.checkbox("Frequent Location", value=sim_location == typical_location)
        sim_recent_similar_login = st.checkbox("Recent Similar Login Pattern", value=True)
        
    # Build simulation session data
    session_data = {
        "user_id": selected_user["id"],
        "user_name": selected_user["name"],
        "device": sim_device,
        "location": sim_location,
        "ip_address": sim_ip,
        "timestamp": sim_time.strftime('%Y-%m-%d %H:%M:%S'),
        "password_changed": sim_password_changed,
        "financial_activity": sim_financial_activity,
        "security_settings_changed": sim_security_settings,
        "added_new_device": sim_added_device,
        "browser": "Chrome",
        "keystroke_data": {
            "match_score": sim_typing_match,
            "sample_size": 25
        }
    }
    
    # Run simulation button
    if st.button("Run ATO Detection Simulation"):
        with st.spinner("Analyzing login attempt..."):
            # Detect account takeover
            ato_result = detect_account_takeover(selected_user, session_data)
            
            # Display results
            st.subheader("Detection Results")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                # Risk gauge
                risk_score = ato_result["risk_score"]
                risk_color = "red" if risk_score > 0.7 else "orange" if risk_score > 0.4 else "green"
                
                risk_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "ATO Risk Score"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": risk_color},
                        "steps": [
                            {"range": [0, 0.4], "color": "green"},
                            {"range": [0.4, 0.7], "color": "yellow"},
                            {"range": [0.7, 1.0], "color": "red"}
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 0.7
                        }
                    }
                ))
                
                st.plotly_chart(risk_fig, use_container_width=True)
                
                # Decision
                is_ato = ato_result["is_ato_attempt"]
                confidence = ato_result["confidence"]
                action = ato_result["recommended_action"]
                
                st.markdown(f"**Account Takeover Detected:** {'Yes' if is_ato else 'No'}")
                st.markdown(f"**Confidence:** {confidence:.0%}")
                st.markdown(f"**Recommended Action:** {action.replace('_', ' ').title()}")
            
            with result_col2:
                # Risk factors
                st.subheader("Risk Factors")
                
                if not ato_result["flags"]:
                    st.info("No risk factors detected.")
                else:
                    for flag in ato_result["flags"]:
                        st.warning(flag)
                
                # IP reputation
                if "ip_reputation" in ato_result.get("login_analysis", {}):
                    ip_rep = ato_result["login_analysis"]["ip_reputation"]
                    
                    st.subheader("IP Reputation")
                    st.markdown(f"**IP Address:** {ip_rep['ip']}")
                    st.markdown(f"**Country:** {ip_rep['country_code']}")
                    st.markdown(f"**Risk Score:** {ip_rep['risk_score']:.2f}")
                    
                    if ip_rep["is_tor"]:
                        st.error("TOR Exit Node Detected")
                    
                    if ip_rep["is_malicious"]:
                        st.error("Known Malicious IP Address")
                    
                    if ip_rep["is_datacenter"]:
                        st.warning("Datacenter/Private IP")
            
            # Actions section
            st.subheader("Recommended Actions")
            
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                if st.button("Allow Login"):
                    st.success("Login allowed. No further action taken.")
            
            with action_col2:
                if st.button("Request Additional Verification"):
                    verification_type = "biometric" if ato_result["risk_score"] > 0.5 else "2fa"
                    st.info(f"Additional {verification_type} verification requested.")
            
            with action_col3:
                if st.button("Block and Alert User"):
                    # Create alert data
                    alert_data = {
                        "id": f"ATO{np.random.randint(10000, 99999)}",
                        "type": "Suspicious Login Attempt",
                        "device": session_data["device"],
                        "location": session_data["location"],
                        "time": session_data["timestamp"],
                        "risk_score": ato_result["risk_score"]
                    }
                    
                    # Send security alert
                    alert_result = send_account_security_alert(alert_data, selected_user["phone"])
                    
                    if alert_result["success"]:
                        st.success("Login blocked and security alert sent to user.")
                    else:
                        st.warning(f"Login blocked but alert could not be sent: {alert_result['message']}")
                        if "Twilio credentials not configured" in alert_result["message"]:
                            st.info("Alert saved to notification history for demonstration purposes.")
            
            # Add to login history for tracking
            session_data["success"] = not is_ato
            session_data["risk_score"] = risk_score
            session_data["failure_reason"] = "Suspected account takeover" if is_ato else None
            session_data["session_id"] = f"SIM{np.random.randint(10000, 99999)}"
            
            st.session_state.login_history.insert(0, session_data)

# Login History tab
with tabs[2]:
    st.header("Login History")
    
    if user_login_history:
        # Convert to DataFrame for display
        login_df = pd.DataFrame(user_login_history)
        
        # Add filter controls
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # Filter by success
            success_filter = st.multiselect(
                "Login Status",
                options=[True, False],
                default=[True, False],
                format_func=lambda x: "Success" if x else "Failed"
            )
        
        with filter_col2:
            # Filter by date range
            date_range = None
            if "timestamp" in login_df.columns and not login_df.empty:
                login_df["timestamp"] = pd.to_datetime(login_df["timestamp"])
                min_date = login_df["timestamp"].min().date()
                max_date = login_df["timestamp"].max().date()
                
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
        
        with filter_col3:
            # Filter by risk level
            risk_level = st.multiselect(
                "Risk Level",
                options=["Low", "Medium", "High"],
                default=["Low", "Medium", "High"]
            )
            
            # Map risk levels to score ranges
            risk_map = {
                "Low": (0.0, 0.4),
                "Medium": (0.4, 0.7),
                "High": (0.7, 1.0)
            }
        
        # Apply filters
        filtered_df = login_df.copy()
        
        if success_filter:
            filtered_df = filtered_df[filtered_df["success"].isin(success_filter)]
        
        # Apply date range filter if defined and valid
        if "timestamp" in filtered_df.columns and 'date_range' in locals() and date_range is not None and isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df["timestamp"].dt.date >= start_date) & 
                (filtered_df["timestamp"].dt.date <= end_date)
            ]
        
        if risk_level:
            risk_conditions = []
            for level in risk_level:
                low, high = risk_map[level]
                risk_conditions.append((filtered_df["risk_score"] >= low) & (filtered_df["risk_score"] < high))
            
            if risk_conditions:
                risk_mask = risk_conditions[0]
                for condition in risk_conditions[1:]:
                    risk_mask = risk_mask | condition
                
                filtered_df = filtered_df[risk_mask]
    else:
        st.info("No login history available for this user.")
        filtered_df = pd.DataFrame()
    
    # Display filtered login history
    if len(filtered_df) > 0:
        # Add risk level column
        filtered_df["risk_level"] = filtered_df["risk_score"].apply(
            lambda x: "High" if x >= 0.7 else "Medium" if x >= 0.4 else "Low"
        )
        
        # Format for display
        display_df = filtered_df[["timestamp", "device", "location", "success", "risk_level", "ip_address"]]
        display_df.columns = ["Timestamp", "Device", "Location", "Success", "Risk Level", "IP Address"]
        
        # Sort by timestamp (most recent first)
        display_df = display_df.sort_values("Timestamp", ascending=False)
        
        # Display as table
        st.dataframe(display_df, use_container_width=True)
        
        # Export button
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Export Login History",
            csv,
            "login_history.csv",
            "text/csv",
            key="download-login-csv"
        )
    else:
        st.info("No login history matching the selected filters.")

# Security Settings tab
with tabs[3]:
    st.header("Security Settings")
    
    st.markdown("""
    Configure security settings to protect against account takeover attempts.
    """)
    
    # Advanced security settings
    st.subheader("Login Security")
    
    security_col1, security_col2 = st.columns(2)
    
    with security_col1:
        # Location-based authentication
        location_auth = st.checkbox(
            "Enable Location-Based Authentication",
            value=selected_user.get("location_auth", True),
            help="Require additional verification when logging in from a new location."
        )
        
        # IP reputation checking
        ip_reputation = st.checkbox(
            "Check IP Reputation",
            value=selected_user.get("ip_reputation_check", True),
            help="Block login attempts from known malicious IP addresses or TOR exit nodes."
        )
        
        # Device fingerprinting
        device_fingerprint = st.checkbox(
            "Enable Device Fingerprinting",
            value=selected_user.get("device_fingerprint", True),
            help="Track device characteristics to identify suspicious logins."
        )
    
    with security_col2:
        # Behavioral biometrics
        behavioral_biometrics = st.checkbox(
            "Enable Behavioral Biometrics",
            value=selected_user.get("behavioral_biometrics", False),
            help="Analyze typing patterns and mouse movements to verify user identity."
        )
        
        # Continuous authentication
        continuous_auth = st.checkbox(
            "Enable Continuous Authentication",
            value=selected_user.get("continuous_auth", False),
            help="Continuously verify user identity throughout the session."
        )
        
        # MFA requirement
        mfa_required = st.checkbox(
            "Require Multi-Factor Authentication",
            value=selected_user.get("mfa_required", True),
            help="Require MFA for all logins."
        )
    
    # Risk thresholds
    st.subheader("Risk Thresholds")
    
    threshold_col1, threshold_col2 = st.columns(2)
    
    with threshold_col1:
        # High risk threshold
        high_risk = st.slider(
            "High Risk Threshold",
            min_value=0.5,
            max_value=0.9,
            value=selected_user.get("high_risk_threshold", 0.7),
            step=0.05,
            help="Risk score above which a login is considered high risk."
        )
        
        # Block threshold
        block_threshold = st.slider(
            "Automatic Block Threshold",
            min_value=0.5,
            max_value=0.95,
            value=selected_user.get("block_threshold", 0.85),
            step=0.05,
            help="Risk score above which a login is automatically blocked."
        )
    
    with threshold_col2:
        # MFA threshold
        mfa_threshold = st.slider(
            "MFA Requirement Threshold",
            min_value=0.3,
            max_value=0.9,
            value=selected_user.get("mfa_threshold", 0.5),
            step=0.05,
            help="Risk score above which MFA is required for login."
        )
        
        # Notification threshold
        notification_threshold = st.slider(
            "Notification Threshold",
            min_value=0.3,
            max_value=0.9,
            value=selected_user.get("notification_threshold", 0.6),
            step=0.05,
            help="Risk score above which the user is notified of the login attempt."
        )
    
    # Save settings
    if st.button("Save Security Settings"):
        # Update user settings
        selected_user["location_auth"] = location_auth
        selected_user["ip_reputation_check"] = ip_reputation
        selected_user["device_fingerprint"] = device_fingerprint
        selected_user["behavioral_biometrics"] = behavioral_biometrics
        selected_user["continuous_auth"] = continuous_auth
        selected_user["mfa_required"] = mfa_required
        
        # Update thresholds
        selected_user["high_risk_threshold"] = high_risk
        selected_user["block_threshold"] = block_threshold
        selected_user["mfa_threshold"] = mfa_threshold
        selected_user["notification_threshold"] = notification_threshold
        
        st.success("Security settings saved successfully!")

# Information about the feature
st.markdown("---")
st.markdown("""
### About Account Takeover (ATO) Detection

This feature helps protect user accounts from unauthorized access attempts by:

- Analyzing login behavior for suspicious patterns
- Checking IP reputation and location
- Using behavioral biometrics to verify user identity
- Detecting anomalies in account activity
- Applying risk-based authentication

For demonstration purposes, the ATO detection is simulated. In a production environment, 
these checks would be performed automatically on every login attempt and suspicious activity.
""")

# Refresh button
if st.sidebar.button("Refresh Data"):
    st.rerun()

# Last updated time
st.sidebar.markdown("---")
st.sidebar.text(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")