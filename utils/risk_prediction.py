import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Check if model exists, if not, create a simple model
def get_or_create_model(model_type="risk_classifier"):
    """
    Returns a pre-trained model or creates a simple one if it doesn't exist
    
    Parameters:
    - model_type: Type of model to return ("risk_classifier" or "impact_predictor")
    
    Returns:
    - Trained model instance
    """
    model_path = f"models/{model_type}.pkl"
    
    # Create the models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Check if model exists
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except:
            # If loading fails, we'll create a new model
            pass
    
    # Create a simple model based on the type
    if model_type == "risk_classifier":
        # For risk classification (e.g., high/medium/low risk)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Features: transaction_amount, days_since_last_transaction, total_volume, volatility
        X = np.random.rand(n_samples, 4)
        X[:, 0] *= 1000000  # transaction_amount (up to 1M)
        X[:, 1] *= 30       # days_since_last_transaction (up to 30 days)
        X[:, 2] *= 5000000  # total_volume (up to 5M)
        X[:, 3] *= 0.5      # volatility (up to 0.5)
        
        # Target: risk class (0=low, 1=medium, 2=high)
        # Generate based on simple rules for demo purposes
        y = np.zeros(n_samples, dtype=int)
        
        # High transaction amount, high volatility tends to be higher risk
        risk_score = X[:, 0]/1000000 * 0.3 + X[:, 3] * 0.7
        y[risk_score > 0.6] = 2  # High risk
        y[(risk_score > 0.3) & (risk_score <= 0.6)] = 1  # Medium risk
        
        # Fit the model
        model.fit(X, y)
        
    elif model_type == "impact_predictor":
        # For predicting the financial impact of risks
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Features: exposure_amount, risk_score, market_volatility, days_to_maturity
        X = np.random.rand(n_samples, 4)
        X[:, 0] *= 5000000   # exposure_amount (up to 5M)
        X[:, 1] *= 1.0       # risk_score (0-1)
        X[:, 2] *= 0.5       # market_volatility (0-0.5)
        X[:, 3] *= 365       # days_to_maturity (0-365)
        
        # Target: potential financial impact
        # Simple formula: impact = exposure * risk_score * (1 + market_volatility) / sqrt(days_to_maturity/365)
        y = X[:, 0] * X[:, 1] * (1 + X[:, 2]) / np.sqrt((X[:, 3]/365 + 0.1))
        
        # Fit the model
        model.fit(X, y)
    
    # Save the model
    joblib.dump(model, model_path)
    
    return model

def predict_risk_class(data):
    """
    Predict risk class (low, medium, high) for given data
    
    Parameters:
    - data: DataFrame or dict containing features
    
    Returns:
    - Dictionary with risk class and confidence
    """
    # Get the model
    model = get_or_create_model("risk_classifier")
    
    # Prepare the input data
    if isinstance(data, dict):
        # Convert to a format suitable for prediction
        features = np.array([
            data.get('transaction_amount', 0),
            data.get('days_since_last_transaction', 0),
            data.get('total_volume', 0),
            data.get('volatility', 0)
        ]).reshape(1, -1)
    elif isinstance(data, pd.DataFrame):
        # Assuming the DataFrame already has the correct columns
        required_cols = ['transaction_amount', 'days_since_last_transaction', 
                        'total_volume', 'volatility']
        
        # Check if all required columns exist
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            # Fill missing columns with zeros
            for col in missing_cols:
                data[col] = 0
        
        features = data[required_cols].values
    else:
        raise ValueError("Data must be a dictionary or DataFrame")
    
    # Predict probabilities
    risk_probs = model.predict_proba(features)
    
    # Get the class with highest probability
    predicted_class = model.predict(features)
    
    # Map class indices to labels
    risk_labels = {0: "low", 1: "medium", 2: "high"}
    
    # Get the confidence (probability of the predicted class)
    if isinstance(predicted_class, np.ndarray):
        predicted_class = predicted_class[0]
    
    confidence = risk_probs[0][predicted_class]
    
    return {
        "risk_class": risk_labels.get(predicted_class, "unknown"),
        "confidence": confidence,
        "risk_score": risk_probs[0][1] * 0.5 + risk_probs[0][2]  # weighted score
    }

def predict_financial_impact(data):
    """
    Predict potential financial impact for given risk data
    
    Parameters:
    - data: DataFrame or dict containing features
    
    Returns:
    - Dictionary with predicted impact and risk-adjusted projections
    """
    # Get the model
    model = get_or_create_model("impact_predictor")
    
    # Prepare the input data
    if isinstance(data, dict):
        # Convert to a format suitable for prediction
        features = np.array([
            data.get('exposure_amount', 0),
            data.get('risk_score', 0),
            data.get('market_volatility', 0),
            data.get('days_to_maturity', 30)  # Default to 30 days
        ]).reshape(1, -1)
    elif isinstance(data, pd.DataFrame):
        # Assuming the DataFrame already has the correct columns
        required_cols = ['exposure_amount', 'risk_score', 
                        'market_volatility', 'days_to_maturity']
        
        # Check if all required columns exist
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            # Fill missing columns with reasonable defaults
            defaults = {
                'exposure_amount': 1000000,
                'risk_score': 0.5,
                'market_volatility': 0.2,
                'days_to_maturity': 30
            }
            
            for col in missing_cols:
                data[col] = defaults.get(col, 0)
        
        features = data[required_cols].values
    else:
        raise ValueError("Data must be a dictionary or DataFrame")
    
    # Predict impact
    predicted_impact = model.predict(features)
    
    if isinstance(predicted_impact, np.ndarray):
        predicted_impact = predicted_impact[0]
    
    # Create risk-adjusted projections
    base_impact = predicted_impact
    worst_case = base_impact * 1.5
    best_case = base_impact * 0.5
    
    return {
        "predicted_impact": base_impact,
        "worst_case_impact": worst_case,
        "best_case_impact": best_case,
        "confidence_interval": [base_impact * 0.75, base_impact * 1.25],
        "prediction_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def analyze_mitigation_strategies(current_risk, strategies):
    """
    Analyze the potential impact of mitigation strategies
    
    Parameters:
    - current_risk: Current risk assessment (dict with risk_score, etc.)
    - strategies: List of strategy dictionaries with details
    
    Returns:
    - Dictionary with analysis results
    """
    # Extract current risk score
    risk_score = current_risk.get("risk_score", 0.5)
    
    # Calculate potential risk reduction for each strategy
    for strategy in strategies:
        # Assign impact values based on impact rating
        impact_values = {
            "High": np.random.uniform(0.15, 0.3),
            "Medium": np.random.uniform(0.05, 0.15),
            "Low": np.random.uniform(0.01, 0.05)
        }
        
        # Calculate risk reduction based on impact
        impact_rating = strategy.get("impact", "Medium")
        strategy["risk_reduction"] = impact_values.get(impact_rating, 0.1)
        
        # Calculate implementation difficulty factor
        effort_values = {
            "High": 0.7,
            "Medium": 0.85,
            "Low": 1.0
        }
        
        effort_rating = strategy.get("effort", "Medium")
        difficulty_factor = effort_values.get(effort_rating, 0.85)
        
        # Adjusted reduction based on difficulty
        strategy["adjusted_reduction"] = strategy["risk_reduction"] * difficulty_factor
        
        # Calculate cost-benefit ratio (higher is better)
        # Assume cost is inversely proportional to effort (higher effort = higher cost)
        cost_factor = 1.0 / effort_values.get(effort_rating, 0.85)
        strategy["cost_benefit_ratio"] = strategy["adjusted_reduction"] / cost_factor
    
    # Calculate cumulative risk reduction (not simply additive - diminishing returns)
    sorted_strategies = sorted(strategies, key=lambda x: x.get("cost_benefit_ratio", 0), reverse=True)
    
    cumulative_reduction = 0
    remaining_risk = risk_score
    
    for strategy in sorted_strategies:
        # Apply diminishing returns - each successive strategy has less impact
        strategy_reduction = strategy["adjusted_reduction"] * (1 - cumulative_reduction)
        strategy["marginal_reduction"] = strategy_reduction
        
        cumulative_reduction += strategy_reduction
        remaining_risk = risk_score * (1 - cumulative_reduction)
    
    # Generate the final analysis
    return {
        "original_risk_score": risk_score,
        "strategies_analyzed": len(strategies),
        "optimal_strategies": sorted_strategies[:3],  # Top 3 by cost-benefit
        "projected_risk_score": remaining_risk,
        "total_risk_reduction": cumulative_reduction,
        "risk_reduction_pct": cumulative_reduction / risk_score if risk_score > 0 else 0,
        "analysis_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def generate_risk_scenarios(base_risk, num_scenarios=3):
    """
    Generate risk scenarios for stress testing
    
    Parameters:
    - base_risk: Base risk assessment (dict)
    - num_scenarios: Number of scenarios to generate
    
    Returns:
    - List of scenario dictionaries
    """
    scenarios = []
    
    base_score = base_risk.get("risk_score", 0.5)
    
    # Define scenario types
    scenario_types = [
        {
            "name": "Market Downturn",
            "description": "Simulates a significant market downturn with increased volatility",
            "risk_multiplier": np.random.uniform(1.3, 1.7),
            "factors": ["market volatility", "liquidity", "credit risk"]
        },
        {
            "name": "Regulatory Change",
            "description": "Simulates impact of major regulatory changes in the financial sector",
            "risk_multiplier": np.random.uniform(1.2, 1.5),
            "factors": ["compliance", "operational", "reputational"]
        },
        {
            "name": "Counterparty Default",
            "description": "Simulates the default of significant counterparties",
            "risk_multiplier": np.random.uniform(1.4, 1.8),
            "factors": ["credit risk", "exposure", "contagion"]
        },
        {
            "name": "Operational Disruption",
            "description": "Simulates major operational disruptions (cyber attack, system failure)",
            "risk_multiplier": np.random.uniform(1.25, 1.6),
            "factors": ["operational", "reputational", "recovery"]
        },
        {
            "name": "Liquidity Crisis",
            "description": "Simulates a severe liquidity crunch in the market",
            "risk_multiplier": np.random.uniform(1.35, 1.75),
            "factors": ["liquidity", "funding", "market access"]
        }
    ]
    
    # Select random scenarios if more options than requested
    if len(scenario_types) > num_scenarios:
        selected_scenarios = np.random.choice(scenario_types, num_scenarios, replace=False)
    else:
        selected_scenarios = scenario_types
    
    # Generate the scenarios
    for scenario in selected_scenarios:
        # Calculate the stressed risk score
        stressed_score = min(base_score * scenario["risk_multiplier"], 1.0)
        
        # Generate impact metrics
        financial_impact = stressed_score * np.random.uniform(500000, 5000000)
        
        # Create the scenario object
        scenario_obj = {
            "name": scenario["name"],
            "description": scenario["description"],
            "base_risk_score": base_score,
            "stressed_risk_score": stressed_score,
            "risk_increase": stressed_score - base_score,
            "risk_increase_pct": (stressed_score - base_score) / base_score if base_score > 0 else 0,
            "key_factors": scenario["factors"],
            "financial_impact": financial_impact,
            "confidence_level": np.random.uniform(0.6, 0.9),
            "generated_on": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        scenarios.append(scenario_obj)
    
    return scenarios