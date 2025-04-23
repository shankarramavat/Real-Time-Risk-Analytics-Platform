import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import ipaddress
import re

# TOR exit node list (simulated)
TOR_EXIT_NODES = [
    "185.220.101.21",
    "185.220.101.22",
    "51.15.43.205",
    "195.176.3.19",
    "192.42.116.16",
    "199.249.230.140",
    "104.244.76.13",
    "193.218.118.231",
    "109.70.100.23",
    "162.247.74.204"
]

# Malicious IP addresses (simulated)
MALICIOUS_IPS = [
    "45.132.192.15",
    "23.106.223.55",
    "191.101.31.35",
    "185.156.73.54",
    "193.27.228.27",
    "45.148.10.65",
    "104.223.31.42",
    "193.38.235.234",
    "185.202.2.36",
    "205.185.116.142"
]

# Country risk scores (simulated, 0-1 scale)
COUNTRY_RISK_SCORES = {
    "US": 0.2,
    "CA": 0.2,
    "GB": 0.3,
    "FR": 0.3,
    "DE": 0.3,
    "JP": 0.3,
    "AU": 0.3,
    "SG": 0.4,
    "RU": 0.7,
    "CN": 0.7,
    "NK": 0.9,
    "IR": 0.8,
    "UNKNOWN": 0.8
}

def check_ip_reputation(ip_address):
    """
    Check reputation of an IP address
    
    Parameters:
    - ip_address: IP address to check
    
    Returns:
    - Dict with reputation details
    """
    result = {
        "ip": ip_address,
        "risk_score": 0.2,  # Default low risk
        "is_tor": False,
        "is_malicious": False,
        "is_datacenter": False,
        "country_code": "US",  # Default
        "asn": "AS13335",  # Default
        "flags": []
    }
    
    # Check if IP is valid
    try:
        ipaddress.ip_address(ip_address)
    except ValueError:
        result["risk_score"] = 0.7
        result["flags"].append("Invalid IP format")
        return result
    
    # Check if IP is in TOR exit node list
    if ip_address in TOR_EXIT_NODES:
        result["is_tor"] = True
        result["risk_score"] += 0.4
        result["flags"].append("TOR exit node")
    
    # Check if IP is in malicious list
    if ip_address in MALICIOUS_IPS:
        result["is_malicious"] = True
        result["risk_score"] += 0.5
        result["flags"].append("Known malicious IP")
    
    # Simulate datacenter detection (simple heuristic)
    if re.match(r'^(192\.168|10\.|172\.(1[6-9]|2[0-9]|3[0-1]))', ip_address):
        result["is_datacenter"] = True
        result["risk_score"] += 0.3
        result["flags"].append("Datacenter/Private IP")
    
    # Simulate country detection
    # This would usually involve a GeoIP database
    country_codes = list(COUNTRY_RISK_SCORES.keys())
    country_code = np.random.choice(country_codes) if np.random.random() < 0.7 else "US"
    result["country_code"] = country_code
    
    # Add country risk
    country_risk = COUNTRY_RISK_SCORES.get(country_code, 0.5)
    result["risk_score"] += country_risk * 0.3  # Weight country risk as 30% of overall score
    
    # Cap risk score at 1.0
    result["risk_score"] = min(1.0, result["risk_score"])
    
    # Add high-risk country flag if applicable
    if country_risk > 0.5:
        result["flags"].append(f"High-risk country ({country_code})")
    
    return result

def analyze_login_behavior(user_data, login_data):
    """
    Analyze login behavior for anomalies
    
    Parameters:
    - user_data: Dict containing user profile data
    - login_data: Dict containing current login data
    
    Returns:
    - Dict with behavioral analysis
    """
    result = {
        "risk_score": 0.2,  # Default low risk
        "flags": [],
        "is_suspicious": False,
        "recommended_action": "allow"
    }
    
    # Extract current login data
    current_device = login_data.get("device", "unknown")
    current_location = login_data.get("location", "unknown")
    current_ip = login_data.get("ip_address", "0.0.0.0")
    current_time = datetime.strptime(login_data.get("timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S')), '%Y-%m-%d %H:%M:%S')
    
    # Check if device is known
    known_devices = user_data.get("devices", [])
    if current_device not in known_devices and current_device != "unknown":
        result["risk_score"] += 0.3
        result["flags"].append("New device")
    
    # Check location against user's typical location
    typical_location = user_data.get("location", "unknown")
    if current_location != typical_location and typical_location != "unknown":
        result["risk_score"] += 0.2
        result["flags"].append("Unusual location")
    
    # Check time of day (assuming user usually logs in during business hours)
    hour_of_day = current_time.hour
    if hour_of_day < 6 or hour_of_day > 22:  # Outside 6 AM - 10 PM
        result["risk_score"] += 0.1
        result["flags"].append("Unusual time of day")
    
    # Check login velocity (time since last login)
    last_login_str = user_data.get("last_login", (current_time - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
    last_login = datetime.strptime(last_login_str, '%Y-%m-%d %H:%M:%S')
    time_since_last_login = (current_time - last_login).total_seconds() / 3600  # in hours
    
    if time_since_last_login < 0.05:  # Less than 3 minutes
        result["risk_score"] += 0.4
        result["flags"].append("Multiple rapid login attempts")
    
    # Check IP reputation
    ip_reputation = check_ip_reputation(current_ip)
    result["ip_reputation"] = ip_reputation
    
    # Add IP risk to overall risk
    result["risk_score"] = 0.6 * result["risk_score"] + 0.4 * ip_reputation["risk_score"]
    
    # Add IP flags to overall flags
    result["flags"].extend(ip_reputation["flags"])
    
    # Determine if login is suspicious
    result["is_suspicious"] = result["risk_score"] > 0.6
    
    # Recommended action based on risk score
    if result["risk_score"] > 0.8:
        result["recommended_action"] = "block"
    elif result["risk_score"] > 0.6:
        result["recommended_action"] = "mfa_required"
    elif result["risk_score"] > 0.4:
        result["recommended_action"] = "biometric_verification"
    else:
        result["recommended_action"] = "allow"
    
    return result

def analyze_keystroke_dynamics(keystroke_data):
    """
    Analyze keystroke dynamics for behavioral biometrics
    
    Parameters:
    - keystroke_data: Dict containing keystroke timing data
    
    Returns:
    - Dict with analysis results
    """
    # In a real system, this would use machine learning to analyze typing patterns
    # For demo purposes, we'll use a simplified approach
    
    result = {
        "match_score": 0.0,
        "is_match": False,
        "confidence": 0.0,
        "flags": []
    }
    
    # Simulate keystroke analysis result
    match_score = np.random.uniform(0.4, 0.95)
    result["match_score"] = match_score
    
    # Determine if it's a match
    result["is_match"] = match_score > 0.7
    
    # Confidence based on sample size
    sample_size = keystroke_data.get("sample_size", 10)
    result["confidence"] = min(1.0, 0.5 + sample_size / 100)
    
    # Flags based on analysis
    if match_score < 0.5:
        result["flags"].append("Significantly different typing pattern")
    elif match_score < 0.7:
        result["flags"].append("Somewhat different typing pattern")
    
    return result

def detect_account_takeover(user_data, session_data):
    """
    Detect potential account takeover attempts
    
    Parameters:
    - user_data: Dict containing user profile data
    - session_data: Dict containing current session data
    
    Returns:
    - Dict with ATO detection results
    """
    result = {
        "risk_score": 0.0,
        "is_ato_attempt": False,
        "confidence": 0.0,
        "flags": [],
        "recommended_action": "allow"
    }
    
    # Analyze login behavior
    login_analysis = analyze_login_behavior(user_data, session_data)
    result["login_analysis"] = login_analysis
    
    # Start with login risk score
    result["risk_score"] = login_analysis["risk_score"]
    
    # Check for password change activity
    if session_data.get("password_changed", False):
        result["risk_score"] += 0.2
        result["flags"].append("Password changed in this session")
    
    # Check for financial activity
    if session_data.get("financial_activity", False):
        result["risk_score"] += 0.2
        result["flags"].append("Financial activity in this session")
    
    # Check for changes to security settings
    if session_data.get("security_settings_changed", False):
        result["risk_score"] += 0.3
        result["flags"].append("Security settings changed")
    
    # Check for new devices/phones added
    if session_data.get("added_new_device", False):
        result["risk_score"] += 0.2
        result["flags"].append("New device added")
    
    # Cap risk score at 1.0
    result["risk_score"] = min(1.0, result["risk_score"])
    
    # Determine if this is an ATO attempt
    result["is_ato_attempt"] = result["risk_score"] > 0.7
    
    # Confidence based on number of flags
    result["confidence"] = min(1.0, 0.3 + len(result["flags"]) * 0.1)
    
    # Add login flags to overall flags
    result["flags"].extend(login_analysis["flags"])
    
    # Recommended action based on risk score
    if result["risk_score"] > 0.8:
        result["recommended_action"] = "block_and_notify"
    elif result["risk_score"] > 0.7:
        result["recommended_action"] = "step_up_auth"
    elif result["risk_score"] > 0.5:
        result["recommended_action"] = "passive_monitoring"
    else:
        result["recommended_action"] = "allow"
    
    return result