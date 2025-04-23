import os
import json
from openai import OpenAI

# The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# Do not change this unless explicitly requested by the user
MODEL = "gpt-4o"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)

def get_risk_insights(data, context=None):
    """
    Get AI-powered insights on risk data
    
    Parameters:
    - data: Dict or JSON string containing risk data
    - context: Additional context for the analysis
    
    Returns:
    - Dict containing insights
    """
    if OPENAI_API_KEY == "":
        # Return dummy insights if no API key is provided
        return {
            "summary": "API key not configured. Please set the OPENAI_API_KEY environment variable.",
            "risk_level": "unknown",
            "recommendations": ["Configure OpenAI API key to get AI-powered insights."],
            "key_factors": []
        }
    
    try:
        # Convert data to string if it's a dict
        if isinstance(data, dict):
            data_str = json.dumps(data)
        else:
            data_str = data
        
        # Build the prompt
        prompt = f"""
You are a financial risk expert. Analyze the following risk data and provide insights:

{data_str}

{context if context else ''}

Provide your analysis in the following JSON format:
{{
    "summary": "A brief summary of the risk analysis",
    "risk_level": "One of: low, moderate, elevated, high, severe",
    "recommendations": ["List of actionable recommendations"],
    "key_factors": ["List of key factors driving the risk assessment"]
}}
"""
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert financial risk analyst."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # Parse and return the response
        insight_data = json.loads(response.choices[0].message.content)
        return insight_data
    
    except Exception as e:
        print(f"Error getting insights from OpenAI: {e}")
        return {
            "summary": f"Error analyzing data: {str(e)}",
            "risk_level": "unknown",
            "recommendations": ["Check system logs for error details"],
            "key_factors": ["API error"]
        }

def analyze_compliance_risk(compliance_data, entity_name=None):
    """
    Analyze compliance data for risk factors
    
    Parameters:
    - compliance_data: Dict or DataFrame containing compliance information
    - entity_name: Name of the entity being analyzed
    
    Returns:
    - Dict containing compliance analysis
    """
    if OPENAI_API_KEY == "":
        # Return dummy insights if no API key is provided
        return {
            "summary": "API key not configured. Please set the OPENAI_API_KEY environment variable.",
            "compliance_status": "unknown",
            "risk_factors": ["API key not configured"],
            "recommendations": ["Configure OpenAI API key to get AI-powered insights."]
        }
    
    try:
        # Convert data to string if it's not already
        if not isinstance(compliance_data, str):
            data_str = json.dumps(compliance_data)
        else:
            data_str = compliance_data
        
        entity_context = f"for {entity_name}" if entity_name else ""
        
        # Build the prompt
        prompt = f"""
Analyze the following compliance data {entity_context} and provide a risk assessment:

{data_str}

Focus on identifying potential compliance risks, regulatory issues, and suggested remediation steps.
Provide your analysis in the following JSON format:
{{
    "summary": "A brief summary of the compliance analysis",
    "compliance_status": "One of: compliant, minor_issues, significant_concerns, non_compliant, critical_violations",
    "risk_factors": ["List of key compliance risk factors identified"],
    "recommendations": ["List of actionable recommendations to address compliance issues"]
}}
"""
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert financial compliance analyst."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # Parse and return the response
        analysis_data = json.loads(response.choices[0].message.content)
        return analysis_data
    
    except Exception as e:
        print(f"Error analyzing compliance with OpenAI: {e}")
        return {
            "summary": f"Error analyzing compliance data: {str(e)}",
            "compliance_status": "unknown",
            "risk_factors": ["API error"],
            "recommendations": ["Check system logs for error details"]
        }

def generate_market_scenario(scenario_type="recession"):
    """
    Generate a market scenario for stress testing
    
    Parameters:
    - scenario_type: Type of scenario to generate (recession, inflation, market_crash, etc.)
    
    Returns:
    - Dict containing scenario parameters
    """
    if OPENAI_API_KEY == "":
        # Return dummy scenario if no API key is provided
        return {
            "scenario_name": "API key not configured",
            "description": "Please set the OPENAI_API_KEY environment variable.",
            "parameters": {
                "market_shock": -0.15,
                "interest_rate_change": 0.02,
                "credit_spread_widening": 0.03,
                "liquidity_reduction": 0.25
            }
        }
    
    try:
        # Build the prompt
        prompt = f"""
Generate a realistic financial stress testing scenario for {scenario_type}.

Include market parameters that would be used in a stress test, such as market shocks, interest rate changes, 
credit spread widening, liquidity reductions, etc.

Provide your scenario in the following JSON format:
{{
    "scenario_name": "A descriptive name for the scenario",
    "description": "A detailed description of the economic/market conditions in this scenario",
    "parameters": {{
        "market_shock": numeric value (e.g., -0.25 for a 25% market decline),
        "interest_rate_change": numeric value (percentage points),
        "credit_spread_widening": numeric value (percentage points),
        "liquidity_reduction": numeric value (e.g., 0.40 for 40% reduction in market liquidity),
        ... add other relevant parameters for a {scenario_type} scenario
    }}
}}
"""
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in financial markets and stress testing."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.5
        )
        
        # Parse and return the response
        scenario_data = json.loads(response.choices[0].message.content)
        return scenario_data
    
    except Exception as e:
        print(f"Error generating scenario with OpenAI: {e}")
        return {
            "scenario_name": f"Error: {str(e)}",
            "description": "An error occurred while generating the scenario.",
            "parameters": {
                "market_shock": -0.15,
                "interest_rate_change": 0.02,
                "credit_spread_widening": 0.03,
                "liquidity_reduction": 0.25
            }
        }
