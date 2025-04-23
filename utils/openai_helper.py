import os
import json
import logging
from openai import OpenAI

# Define error types for different OpenAI versions
# This simpler approach works with all OpenAI package versions
API_ERROR_TYPES = {
    "rate_limit": ["rate limit", "quota", "capacity", "exceeded"],
    "connection": ["connection", "network", "timeout"],
    "auth": ["authentication", "auth", "key", "invalid key"]
}

# The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# Do not change this unless explicitly requested by the user
MODEL = "gpt-4o"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Set up OpenAI client if API key is available
client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logging.error(f"Error initializing OpenAI client: {e}")
        # Continue without OpenAI integration

# Helper function to check if OpenAI integration is available
def is_openai_available():
    """Check if OpenAI integration is available and working"""
    return client is not None and OPENAI_API_KEY != ""

def get_risk_insights(data, context=None):
    """
    Get AI-powered insights on risk data
    
    Parameters:
    - data: Dict or JSON string containing risk data
    - context: Additional context for the analysis
    
    Returns:
    - Dict containing insights
    """
    if not is_openai_available():
        # Return sample insights if OpenAI is not available
        return {
            "summary": "AI insights are not available. Please configure your OpenAI API key.",
            "risk_level": "unknown",
            "recommendations": ["Visit OpenAI to get an API key.", 
                              "Set the OPENAI_API_KEY environment variable.",
                              "Restart the application."],
            "key_factors": ["Missing API key or quota exceeded"]
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
        if client:  # Check that client is initialized
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
        else:
            raise Exception("OpenAI client not initialized")
    
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check if this is a rate limit error
        if any(term in error_msg for term in API_ERROR_TYPES["rate_limit"]):
            print(f"OpenAI rate limit exceeded: {e}")
            return {
                "summary": "API rate limit exceeded. Please try again later or upgrade your OpenAI plan.",
                "risk_level": "unknown",
                "recommendations": ["Try again later", "Upgrade your OpenAI API plan"],
                "key_factors": ["API quota exceeded"]
            }
        # Check if this is a connection error
        elif any(term in error_msg for term in API_ERROR_TYPES["connection"]):
            print(f"OpenAI connection error: {e}")
            return {
                "summary": "Unable to connect to OpenAI API. Please check your internet connection.",
                "risk_level": "unknown",
                "recommendations": ["Check your internet connection", "Verify OpenAI API status"],
                "key_factors": ["API connection error"]
            }
        # Check if this is an authentication error
        elif any(term in error_msg for term in API_ERROR_TYPES["auth"]):
            print(f"OpenAI authentication error: {e}")
            return {
                "summary": "Authentication error with OpenAI API. Please check your API key.",
                "risk_level": "unknown",
                "recommendations": ["Verify your OpenAI API key", "Check account status"],
                "key_factors": ["API authentication error"]
            }
        else:
            # General error handling
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
    if not is_openai_available():
        # Return sample insights if OpenAI is not available
        return {
            "summary": "AI insights are not available. Please configure your OpenAI API key.",
            "compliance_status": "unknown",
            "risk_factors": ["API key not configured or quota exceeded"],
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
        if client:  # Check that client is initialized
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
        else:
            raise Exception("OpenAI client not initialized")
    
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check if this is a rate limit error
        if any(term in error_msg for term in API_ERROR_TYPES["rate_limit"]):
            print(f"OpenAI rate limit exceeded: {e}")
            return {
                "summary": "API rate limit exceeded. Please try again later.",
                "compliance_status": "unknown",
                "risk_factors": ["API quota exceeded"],
                "recommendations": ["Try again later", "Upgrade your OpenAI API plan"]
            }
        # Check if this is a connection error
        elif any(term in error_msg for term in API_ERROR_TYPES["connection"]):
            print(f"OpenAI connection error: {e}")
            return {
                "summary": "Unable to connect to OpenAI API.",
                "compliance_status": "unknown",
                "risk_factors": ["API connection error"],
                "recommendations": ["Check your internet connection", "Verify OpenAI API status"]
            }
        # General error handling
        else:
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
    if not is_openai_available():
        # Return sample scenario if OpenAI is not available
        return {
            "scenario_name": "Sample Recession Scenario (API key not configured)",
            "description": "This is a sample scenario provided when OpenAI integration is not available. Configure your OpenAI API key for AI-generated scenarios.",
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
        if client:  # Check that client is initialized
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
        else:
            raise Exception("OpenAI client not initialized")
    
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check if this is a rate limit error
        if any(term in error_msg for term in API_ERROR_TYPES["rate_limit"]):
            print(f"OpenAI rate limit exceeded: {e}")
            return {
                "scenario_name": "API Rate Limit Exceeded",
                "description": "Unable to generate scenario due to OpenAI API rate limit. Please try again later.",
                "parameters": {
                    "market_shock": -0.15,
                    "interest_rate_change": 0.02,
                    "credit_spread_widening": 0.03,
                    "liquidity_reduction": 0.25
                }
            }
        # Check if this is a connection error
        elif any(term in error_msg for term in API_ERROR_TYPES["connection"]):
            print(f"OpenAI connection error: {e}")
            return {
                "scenario_name": "API Connection Error",
                "description": "Unable to connect to OpenAI API. Please check your internet connection.",
                "parameters": {
                    "market_shock": -0.15,
                    "interest_rate_change": 0.02,
                    "credit_spread_widening": 0.03,
                    "liquidity_reduction": 0.25
                }
            }
        # General error handling
        else:
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
