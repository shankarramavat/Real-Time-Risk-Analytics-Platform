import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_var(exposures, confidence_level=0.95, time_horizon=10):
    """
    Calculate Value at Risk (VaR) based on historical simulation method
    
    Parameters:
    - exposures: List or array of historical exposures
    - confidence_level: Confidence level for VaR calculation (default: 0.95)
    - time_horizon: Time horizon in days (default: 10)
    
    Returns:
    - VaR value
    """
    if len(exposures) < 2:
        return 0
    
    # Calculate daily returns
    returns = np.diff(exposures) / exposures[:-1]
    
    # Sort returns
    sorted_returns = np.sort(returns)
    
    # Find the index for the confidence level
    index = int(np.floor((1 - confidence_level) * len(sorted_returns)))
    
    # Get the return at the confidence level
    var_return = sorted_returns[index]
    
    # Calculate VaR
    var = abs(var_return * exposures[-1] * np.sqrt(time_horizon))
    
    return var

def calculate_expected_shortfall(exposures, confidence_level=0.95):
    """
    Calculate Expected Shortfall (ES) / Conditional VaR based on historical simulation
    
    Parameters:
    - exposures: List or array of historical exposures
    - confidence_level: Confidence level for ES calculation (default: 0.95)
    
    Returns:
    - ES value
    """
    if len(exposures) < 2:
        return 0
    
    # Calculate daily returns
    returns = np.diff(exposures) / exposures[:-1]
    
    # Sort returns
    sorted_returns = np.sort(returns)
    
    # Find the index for the confidence level
    index = int(np.floor((1 - confidence_level) * len(sorted_returns)))
    
    # Get the returns below the confidence level
    tail_returns = sorted_returns[:index+1]
    
    # Calculate Expected Shortfall
    es = abs(np.mean(tail_returns) * exposures[-1])
    
    return es

def calculate_counterparty_risk_score(counterparty_data):
    """
    Calculate a risk score for a counterparty based on multiple factors
    
    Parameters:
    - counterparty_data: DataFrame row or dict containing counterparty information
    
    Returns:
    - Risk score (0-1)
    """
    # Define weights for different factors
    weights = {
        'credit_rating': 0.3,
        'exposure_ratio': 0.3,
        'country_risk': 0.2,
        'sector_risk': 0.2
    }
    
    # Credit rating conversion to numerical score (higher = riskier)
    credit_rating_scores = {
        'AAA': 0.1, 'AA+': 0.15, 'AA': 0.2, 'AA-': 0.25,
        'A+': 0.3, 'A': 0.35, 'A-': 0.4,
        'BBB+': 0.5, 'BBB': 0.55, 'BBB-': 0.6,
        'BB+': 0.7, 'BB': 0.75, 'BB-': 0.8,
        'B+': 0.85, 'B': 0.9, 'B-': 0.95,
        'CCC': 1.0
    }
    
    # Country risk (simplified)
    country_risk_scores = {
        'USA': 0.1, 'UK': 0.15, 'Germany': 0.15, 'France': 0.2,
        'Japan': 0.2, 'Canada': 0.15, 'Australia': 0.2,
        'China': 0.4, 'India': 0.5, 'Brazil': 0.6,
        'Russia': 0.7, 'South Africa': 0.6, 'Mexico': 0.5,
        # Default for any other country
        'default': 0.5
    }
    
    # Sector risk (simplified)
    sector_risk_scores = {
        'Government': 0.1, 'Banking': 0.3, 'Insurance': 0.35,
        'Asset Management': 0.4, 'Pension Fund': 0.3,
        'Hedge Fund': 0.6, 'Corporate': 0.5,
        # Default for any other sector
        'default': 0.5
    }
    
    # Calculate credit rating score
    credit_rating = counterparty_data.get('credit_rating', 'BBB')
    credit_score = credit_rating_scores.get(credit_rating, 0.55)
    
    # Calculate exposure ratio (current exposure vs limit)
    exposure_limit = counterparty_data.get('exposure_limit', 1000000)
    current_exposure = counterparty_data.get('current_exposure', 0)
    exposure_ratio = min(1.0, current_exposure / exposure_limit if exposure_limit > 0 else 1.0)
    
    # Get country and sector risk scores
    country = counterparty_data.get('country', 'default')
    country_risk = country_risk_scores.get(country, country_risk_scores['default'])
    
    sector = counterparty_data.get('sector', 'default')
    sector_risk = sector_risk_scores.get(sector, sector_risk_scores['default'])
    
    # Calculate weighted risk score
    risk_score = (
        weights['credit_rating'] * credit_score +
        weights['exposure_ratio'] * exposure_ratio +
        weights['country_risk'] * country_risk +
        weights['sector_risk'] * sector_risk
    )
    
    return min(1.0, max(0.1, risk_score))

def calculate_stress_test_impact(exposures, stress_factor=0.2):
    """
    Calculate the impact of a stress test on exposures
    
    Parameters:
    - exposures: List or array of exposures
    - stress_factor: Factor to apply for stress test (default: 0.2 for 20% shock)
    
    Returns:
    - Stressed exposures
    """
    return exposures * (1 + stress_factor)

def calculate_portfolio_concentration(exposures_by_category):
    """
    Calculate portfolio concentration using Herfindahl-Hirschman Index (HHI)
    
    Parameters:
    - exposures_by_category: Dict with categories (e.g., sectors) as keys and exposure amounts as values
    
    Returns:
    - HHI value (0-1)
    """
    total_exposure = sum(exposures_by_category.values())
    
    if total_exposure == 0:
        return 0
    
    # Calculate market shares
    shares = [exposure / total_exposure for exposure in exposures_by_category.values()]
    
    # Calculate HHI (sum of squared market shares)
    hhi = sum([share ** 2 for share in shares])
    
    return hhi

def generate_risk_time_series(days=30, base_value=0.5, volatility=0.05):
    """
    Generate a time series of risk values for visualization
    
    Parameters:
    - days: Number of days to generate data for
    - base_value: Base risk value
    - volatility: Volatility of the risk value
    
    Returns:
    - List of dates and list of risk values
    """
    np.random.seed(42)  # For reproducibility
    
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    dates.reverse()  # Oldest first
    
    # Generate random walk for risk values
    risk_values = [base_value]
    for _ in range(days - 1):
        new_value = risk_values[-1] + np.random.normal(0, volatility)
        # Keep within bounds
        new_value = min(1.0, max(0.1, new_value))
        risk_values.append(new_value)
    
    return dates, risk_values
