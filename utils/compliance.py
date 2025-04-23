import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

# Simulated sanctions list
SANCTIONS_LIST = {
    "entities": [
        "Restricted Trading Ltd",
        "Blocked Finance Corp",
        "Sanctioned Holdings LLC",
        "Embargoed Investments",
        "Restricted Bank of Nation X"
    ],
    "individuals": [
        "John Sanctioned",
        "Jane Restricted",
        "Bob Embargoed",
        "Alice Blocked",
        "Sam Listed"
    ],
    "countries": [
        "Sanctioned Country",
        "Restricted Nation",
        "Embargoed Territory",
        "High Risk Country"
    ]
}

def check_sanctions(entity_name):
    """
    Check if an entity is on the sanctions list
    
    Parameters:
    - entity_name: Name of the entity to check
    
    Returns:
    - Dict with match status and details
    """
    # Check for exact matches in each category
    for category, names in SANCTIONS_LIST.items():
        if entity_name in names:
            return {
                "match": True,
                "match_type": "exact",
                "category": category,
                "risk_score": 1.0,
                "details": f"Exact match found in {category} sanctions list"
            }
    
    # Check for partial matches
    for category, names in SANCTIONS_LIST.items():
        for name in names:
            if name.lower() in entity_name.lower() or entity_name.lower() in name.lower():
                similarity = 0.8  # High similarity for partial match
                return {
                    "match": True,
                    "match_type": "partial",
                    "category": category,
                    "risk_score": similarity,
                    "details": f"Partial match found in {category} sanctions list"
                }
    
    # No match found
    return {
        "match": False,
        "match_type": "none",
        "category": "none",
        "risk_score": 0.0,
        "details": "No sanctions match found"
    }

def validate_aml_compliance(transaction):
    """
    Validate a transaction against AML rules
    
    Parameters:
    - transaction: Dict or Series containing transaction data
    
    Returns:
    - Dict with compliance status and details
    """
    flags = []
    risk_score = 0.0
    
    # Check transaction amount for suspicious values
    amount = transaction.get('amount', 0)
    
    # Large transaction check
    if amount > 10000:
        flags.append("Large transaction (>$10,000)")
        risk_score += 0.3
    
    # Structuring check (multiple transactions just below reporting threshold)
    if 9000 <= amount < 10000:
        flags.append("Potential structuring (transaction amount just below reporting threshold)")
        risk_score += 0.5
    
    # Round number check (often suspicious)
    if amount % 1000 == 0 and amount > 5000:
        flags.append("Round amount transaction (multiple of $1,000)")
        risk_score += 0.1
    
    # Check for high-risk transaction types
    transaction_type = transaction.get('transaction_type', '')
    high_risk_types = ['Wire Transfer', 'Cash Deposit', 'Crypto', 'Money Order', 'Cash']
    
    if transaction_type in high_risk_types:
        flags.append(f"High-risk transaction type: {transaction_type}")
        risk_score += 0.3
    
    # Check for high-risk countries
    country = transaction.get('country', '')
    if country in SANCTIONS_LIST['countries']:
        flags.append(f"Transaction involving high-risk country: {country}")
        risk_score += 0.7
    
    # Check counterparty
    counterparty = transaction.get('counterparty_name', '')
    sanctions_check = check_sanctions(counterparty)
    
    if sanctions_check['match']:
        flags.append(f"Counterparty sanctions match: {sanctions_check['details']}")
        risk_score += sanctions_check['risk_score']
    
    # Cap risk score at 1.0
    risk_score = min(1.0, risk_score)
    
    # Determine overall status
    if len(flags) == 0:
        status = "Compliant"
    elif risk_score < 0.3:
        status = "Low Risk"
    elif risk_score < 0.6:
        status = "Medium Risk"
    else:
        status = "High Risk"
    
    return {
        "status": status,
        "risk_score": risk_score,
        "flags": flags,
        "details": ", ".join(flags) if flags else "No AML issues detected"
    }

def validate_kyc_compliance(customer_data):
    """
    Validate customer data against KYC requirements
    
    Parameters:
    - customer_data: Dict or Series containing customer information
    
    Returns:
    - Dict with compliance status and details
    """
    issues = []
    risk_score = 0.0
    
    # Check for required fields
    required_fields = ['name', 'country', 'sector', 'last_review_date']
    
    for field in required_fields:
        if field not in customer_data or not customer_data[field]:
            issues.append(f"Missing required field: {field}")
            risk_score += 0.2
    
    # Check last review date
    if 'last_review_date' in customer_data and customer_data['last_review_date']:
        try:
            last_review = datetime.strptime(customer_data['last_review_date'], '%Y-%m-%d')
            days_since_review = (datetime.now() - last_review).days
            
            if days_since_review > 365:
                issues.append(f"KYC review overdue by {days_since_review - 365} days")
                risk_score += min(0.5, 0.1 + (days_since_review - 365) / 1000)  # Cap at 0.5
        except ValueError:
            issues.append("Invalid last review date format")
            risk_score += 0.2
    
    # Check for high-risk countries
    if 'country' in customer_data and customer_data['country'] in SANCTIONS_LIST['countries']:
        issues.append(f"Customer from high-risk country: {customer_data['country']}")
        risk_score += 0.5
    
    # Check for sanctions matches
    if 'name' in customer_data:
        sanctions_check = check_sanctions(customer_data['name'])
        if sanctions_check['match']:
            issues.append(f"Potential sanctions match: {sanctions_check['details']}")
            risk_score += sanctions_check['risk_score']
    
    # Cap risk score at 1.0
    risk_score = min(1.0, risk_score)
    
    # Determine overall status
    if len(issues) == 0:
        status = "Compliant"
    elif risk_score < 0.3:
        status = "Minor Issues"
    elif risk_score < 0.7:
        status = "Significant Concerns"
    else:
        status = "Non-Compliant"
    
    return {
        "status": status,
        "risk_score": risk_score,
        "issues": issues,
        "details": ", ".join(issues) if issues else "No KYC issues detected"
    }

def generate_compliance_report(counterparty_data, transactions_data):
    """
    Generate a comprehensive compliance report
    
    Parameters:
    - counterparty_data: DataFrame containing counterparty information
    - transactions_data: DataFrame containing transaction information
    
    Returns:
    - Dict with compliance report
    """
    # Initialize report structure
    report = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "summary": {
            "total_counterparties": len(counterparty_data),
            "total_transactions": len(transactions_data),
            "high_risk_counterparties": 0,
            "high_risk_transactions": 0,
            "overall_risk_score": 0.0
        },
        "aml_compliance": {
            "status": "",
            "details": [],
            "risk_score": 0.0
        },
        "kyc_compliance": {
            "status": "",
            "details": [],
            "risk_score": 0.0
        },
        "sanctions_compliance": {
            "status": "",
            "details": [],
            "risk_score": 0.0
        }
    }
    
    # AML compliance check
    aml_issues = []
    aml_risk_scores = []
    
    for _, transaction in transactions_data.iterrows():
        aml_check = validate_aml_compliance(transaction)
        if aml_check['risk_score'] >= 0.6:
            report["summary"]["high_risk_transactions"] += 1
            aml_issues.append({
                "transaction_id": transaction.get('id', 'Unknown'),
                "counterparty": transaction.get('counterparty_name', 'Unknown'),
                "amount": transaction.get('amount', 0),
                "currency": transaction.get('currency', 'USD'),
                "issues": aml_check['details'],
                "risk_score": aml_check['risk_score']
            })
        aml_risk_scores.append(aml_check['risk_score'])
    
    # Calculate average AML risk score
    avg_aml_risk = np.mean(aml_risk_scores) if aml_risk_scores else 0.0
    
    report["aml_compliance"] = {
        "status": "High Risk" if avg_aml_risk >= 0.6 else "Medium Risk" if avg_aml_risk >= 0.3 else "Low Risk",
        "details": aml_issues,
        "risk_score": avg_aml_risk
    }
    
    # KYC compliance check
    kyc_issues = []
    kyc_risk_scores = []
    
    for _, counterparty in counterparty_data.iterrows():
        kyc_check = validate_kyc_compliance(counterparty)
        if kyc_check['risk_score'] >= 0.6:
            report["summary"]["high_risk_counterparties"] += 1
            kyc_issues.append({
                "counterparty_id": counterparty.get('id', 'Unknown'),
                "counterparty_name": counterparty.get('name', 'Unknown'),
                "issues": kyc_check['details'],
                "risk_score": kyc_check['risk_score']
            })
        kyc_risk_scores.append(kyc_check['risk_score'])
    
    # Calculate average KYC risk score
    avg_kyc_risk = np.mean(kyc_risk_scores) if kyc_risk_scores else 0.0
    
    report["kyc_compliance"] = {
        "status": "High Risk" if avg_kyc_risk >= 0.6 else "Medium Risk" if avg_kyc_risk >= 0.3 else "Low Risk",
        "details": kyc_issues,
        "risk_score": avg_kyc_risk
    }
    
    # Sanctions compliance check
    sanctions_issues = []
    
    for _, counterparty in counterparty_data.iterrows():
        sanctions_check = check_sanctions(counterparty.get('name', ''))
        if sanctions_check['match']:
            sanctions_issues.append({
                "counterparty_id": counterparty.get('id', 'Unknown'),
                "counterparty_name": counterparty.get('name', 'Unknown'),
                "match_type": sanctions_check['match_type'],
                "details": sanctions_check['details'],
                "risk_score": sanctions_check['risk_score']
            })
    
    # Sanctions risk score is based on the number of sanctions matches
    sanctions_risk = min(1.0, len(sanctions_issues) / 10)  # Cap at 1.0
    
    report["sanctions_compliance"] = {
        "status": "High Risk" if sanctions_risk >= 0.3 else "Medium Risk" if sanctions_risk > 0 else "Low Risk",
        "details": sanctions_issues,
        "risk_score": sanctions_risk
    }
    
    # Calculate overall risk score (weighted average)
    weights = {
        "aml": 0.4,
        "kyc": 0.4,
        "sanctions": 0.2
    }
    
    overall_risk = (
        weights["aml"] * avg_aml_risk +
        weights["kyc"] * avg_kyc_risk +
        weights["sanctions"] * sanctions_risk
    )
    
    report["summary"]["overall_risk_score"] = overall_risk
    
    return report
