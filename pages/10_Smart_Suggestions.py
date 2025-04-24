import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import utility functions
from utils.risk_calculations import calculate_var, calculate_expected_shortfall
from utils.data_generator import load_data
from utils.session_helper import initialize_session_state
from utils.openai_helper import get_risk_insights, is_openai_available

# Initialize session state
initialize_session_state()

# Page title
st.title("Smart Risk Mitigation Suggestions")

# Description
st.markdown("""
This page provides AI-powered smart suggestions for mitigating identified risks.
The recommendations are tailored to your specific risk profile and exposure patterns.
""")

# Sidebar - Strategy focus
st.sidebar.header("Mitigation Focus")

# Let user select which risk areas to focus on
risk_areas = st.sidebar.multiselect(
    "Risk Areas",
    options=["Counterparty Risk", "Market Exposure", "Compliance", "Fraud", "Operational"],
    default=["Counterparty Risk", "Market Exposure", "Compliance"]
)

# Risk tolerance setting
risk_tolerance = st.sidebar.select_slider(
    "Risk Tolerance",
    options=["Very Low", "Low", "Moderate", "High", "Very High"],
    value="Moderate"
)

# Time horizon
time_horizon = st.sidebar.slider(
    "Strategy Time Horizon (days)",
    min_value=1,
    max_value=365,
    value=30
)

# Load necessary data
df_counterparties = load_data("counterparties")
df_transactions = load_data("transactions")
df_market = load_data("market_data")

# Create columns for summary metrics
col1, col2, col3 = st.columns(3)

# Calculate key metrics
with col1:
    total_counterparties = len(df_counterparties)
    high_risk_counterparties = len(df_counterparties[df_counterparties["risk_score"] > 0.7])
    st.metric(
        "High Risk Counterparties", 
        f"{high_risk_counterparties}",
        f"{high_risk_counterparties/total_counterparties:.1%}"
    )

with col2:
    if "amount" in df_transactions.columns:
        total_exposure = df_transactions["amount"].sum()
        high_risk_exposure = df_transactions[
            df_transactions["counterparty_id"].isin(
                df_counterparties[df_counterparties["risk_score"] > 0.7]["id"]
            )
        ]["amount"].sum() if "amount" in df_transactions.columns else 0
        
        exposure_pct = high_risk_exposure/total_exposure if total_exposure > 0 else 0
        st.metric(
            "High Risk Exposure", 
            f"${high_risk_exposure/1000000:.2f}M",
            f"{exposure_pct:.1%}"
        )
    else:
        st.metric("High Risk Exposure", "N/A")

with col3:
    # Calculate the number of suggested actions
    num_actions = len(risk_areas) * 3 + (5 - ["Very Low", "Low", "Moderate", "High", "Very High"].index(risk_tolerance))
    st.metric(
        "Suggested Actions", 
        num_actions
    )

# Generate mitigation strategies based on risk areas
st.header("Recommended Risk Mitigation Strategies")

# Tab layout for different risk areas
tabs = st.tabs(risk_areas)

# Counter to track overall suggestion number
suggestion_counter = 1

# Function to generate area-specific suggestions
def generate_suggestions_for_area(area, tolerance, time_horizon):
    global suggestion_counter
    suggestions = []
    
    if area == "Counterparty Risk":
        # Base suggestions
        suggestions = [
            {
                "title": "Counterparty Exposure Limits Review",
                "description": "Review and adjust counterparty exposure limits based on current risk scores and market conditions.",
                "impact": "High",
                "effort": "Medium",
                "timeframe": "2-4 weeks",
                "key_metrics": ["Concentration Risk", "Counterparty Risk Score"]
            },
            {
                "title": "Enhanced Due Diligence for High-Risk Counterparties",
                "description": "Implement additional due diligence steps for counterparties with risk scores above 0.7.",
                "impact": "High",
                "effort": "High",
                "timeframe": "1-3 months",
                "key_metrics": ["Due Diligence Completion Rate", "Risk Score Trend"]
            },
            {
                "title": "Hedging Strategy for Large Exposures",
                "description": "Develop hedging strategies for counterparties representing more than 5% of total exposure.",
                "impact": "High",
                "effort": "Medium",
                "timeframe": "2-6 weeks",
                "key_metrics": ["Net Exposure", "Hedging Cost Ratio"]
            }
        ]
        
        # Add specific suggestions based on tolerance
        if tolerance in ["Very Low", "Low"]:
            suggestions.append({
                "title": "Collateral Requirement Increase",
                "description": "Increase collateral requirements for all counterparties based on their risk profiles.",
                "impact": "High",
                "effort": "Medium",
                "timeframe": "2-4 weeks",
                "key_metrics": ["Collateral Coverage Ratio", "Unsecured Exposure"]
            })
        
        if tolerance == "Very Low":
            suggestions.append({
                "title": "Counterparty Diversification Plan",
                "description": "Develop a structured plan to reduce concentration by diversifying counterparty exposure.",
                "impact": "High",
                "effort": "High",
                "timeframe": "3-6 months",
                "key_metrics": ["Herfindahl-Hirschman Index", "Max Single Counterparty Exposure"]
            })
    
    elif area == "Market Exposure":
        # Base suggestions
        suggestions = [
            {
                "title": "Sector Exposure Rebalancing",
                "description": "Rebalance portfolio to reduce overexposure to volatile market sectors.",
                "impact": "High",
                "effort": "Medium",
                "timeframe": "1-4 weeks",
                "key_metrics": ["Sector Concentration", "VaR by Sector"]
            },
            {
                "title": "Stress Testing Parameters Update",
                "description": "Update stress testing scenarios to include recent market volatility patterns.",
                "impact": "Medium",
                "effort": "Low",
                "timeframe": "1-2 weeks",
                "key_metrics": ["Stress Test Coverage", "Scenario Severity"]
            },
            {
                "title": "Hedging Instruments Review",
                "description": "Review and optimize existing hedging instruments against current market conditions.",
                "impact": "High",
                "effort": "Medium",
                "timeframe": "2-4 weeks",
                "key_metrics": ["Hedge Effectiveness", "Hedging Cost"]
            }
        ]
        
        # Add specific suggestions based on tolerance
        if tolerance in ["Very Low", "Low"]:
            suggestions.append({
                "title": "Stop-Loss Trigger Implementation",
                "description": "Implement automated stop-loss triggers for positions with high volatility.",
                "impact": "High",
                "effort": "Medium",
                "timeframe": "3-6 weeks",
                "key_metrics": ["Downside Protection", "Realized Loss Prevention"]
            })
    
    elif area == "Compliance":
        # Base suggestions
        suggestions = [
            {
                "title": "Regulatory Reporting Enhancement",
                "description": "Enhance regulatory reporting processes to ensure timely and accurate submissions.",
                "impact": "Medium",
                "effort": "Medium",
                "timeframe": "1-3 months",
                "key_metrics": ["Reporting Accuracy", "Submission Timeliness"]
            },
            {
                "title": "AML/KYC Procedure Update",
                "description": "Update AML/KYC procedures to align with the latest regulatory requirements.",
                "impact": "High",
                "effort": "High",
                "timeframe": "2-4 months",
                "key_metrics": ["Compliance Score", "Regulatory Finding Rate"]
            },
            {
                "title": "Sanctions Screening Enhancement",
                "description": "Enhance sanctions screening process with advanced matching algorithms and more frequent updates.",
                "impact": "High",
                "effort": "Medium",
                "timeframe": "1-3 months",
                "key_metrics": ["Screening Accuracy", "False Positive Rate"]
            }
        ]
        
        # Add specific suggestions based on tolerance
        if tolerance in ["Very Low", "Low"]:
            suggestions.append({
                "title": "Compliance Training Program",
                "description": "Implement a comprehensive compliance training program for all staff with regular updates.",
                "impact": "Medium",
                "effort": "Medium",
                "timeframe": "2-4 months",
                "key_metrics": ["Training Completion Rate", "Compliance Awareness"]
            })
    
    elif area == "Fraud":
        # Base suggestions
        suggestions = [
            {
                "title": "Fraud Detection Model Update",
                "description": "Update fraud detection models with the latest patterns and techniques.",
                "impact": "High",
                "effort": "Medium",
                "timeframe": "1-2 months",
                "key_metrics": ["False Positive Rate", "Detection Rate"]
            },
            {
                "title": "Behavioral Analytics Implementation",
                "description": "Implement advanced behavioral analytics to detect unusual patterns indicative of fraud.",
                "impact": "High",
                "effort": "High",
                "timeframe": "2-4 months",
                "key_metrics": ["Anomaly Detection Rate", "Investigation Efficiency"]
            },
            {
                "title": "Multi-Factor Authentication Expansion",
                "description": "Expand multi-factor authentication to all critical systems and transactions.",
                "impact": "High",
                "effort": "Medium",
                "timeframe": "1-3 months",
                "key_metrics": ["Authentication Success Rate", "Security Incident Rate"]
            }
        ]
    
    elif area == "Operational":
        # Base suggestions
        suggestions = [
            {
                "title": "Process Automation Expansion",
                "description": "Expand automation of risk management processes to reduce manual errors.",
                "impact": "Medium",
                "effort": "High",
                "timeframe": "3-6 months",
                "key_metrics": ["Process Efficiency", "Error Rate"]
            },
            {
                "title": "Business Continuity Plan Update",
                "description": "Update business continuity plans to address emerging operational risks.",
                "impact": "Medium",
                "effort": "Medium",
                "timeframe": "1-3 months",
                "key_metrics": ["Recovery Time Objective", "Plan Test Results"]
            },
            {
                "title": "Vendor Risk Management Review",
                "description": "Review and enhance vendor risk management processes, especially for critical service providers.",
                "impact": "Medium",
                "effort": "Medium",
                "timeframe": "2-4 months",
                "key_metrics": ["Vendor Risk Score", "Service Level Compliance"]
            }
        ]
    
    # Adjust suggestions based on time horizon
    for suggestion in suggestions:
        timeframe_weeks = {
            "1-2 weeks": 1.5,
            "2-4 weeks": 3,
            "1-3 months": 8,
            "2-4 months": 12,
            "2-6 weeks": 4,
            "3-6 months": 20,
            "3-6 weeks": 4.5,
        }
        
        avg_weeks = timeframe_weeks.get(suggestion["timeframe"], 4)
        
        # If time horizon is shorter than the suggestion timeframe, mark as "Long-term"
        if time_horizon < avg_weeks * 7:
            suggestion["timeline_fit"] = "Long-term"
        else:
            suggestion["timeline_fit"] = "Within horizon"
    
    # Display suggestions with numbers
    for suggestion in suggestions:
        expander = st.expander(f"{suggestion_counter}. {suggestion['title']} ({suggestion['impact']} Impact)")
        with expander:
            st.markdown(f"**Description:** {suggestion['description']}")
            st.markdown(f"**Impact:** {suggestion['impact']}")
            st.markdown(f"**Effort Required:** {suggestion['effort']}")
            st.markdown(f"**Timeframe:** {suggestion['timeframe']}")
            st.markdown(f"**Timeline Fit:** {suggestion['timeline_fit']}")
            
            st.markdown("**Key Metrics to Track:**")
            for metric in suggestion["key_metrics"]:
                st.markdown(f"- {metric}")
            
            # Add implementation button
            if st.button(f"Add to Implementation Plan #{suggestion_counter}"):
                if "implementation_plan" not in st.session_state:
                    st.session_state.implementation_plan = []
                
                st.session_state.implementation_plan.append({
                    "id": suggestion_counter,
                    "title": suggestion["title"],
                    "area": area,
                    "impact": suggestion["impact"],
                    "added_date": datetime.now().strftime("%Y-%m-%d")
                })
                
                st.success(f"Added '{suggestion['title']}' to implementation plan!")
        
        suggestion_counter += 1
    
    return suggestion_counter - 1

# Generate suggestions for each selected risk area
for i, area in enumerate(risk_areas):
    with tabs[i]:
        st.subheader(f"{area} Mitigation Strategies")
        generate_suggestions_for_area(area, risk_tolerance, time_horizon)

# Display AI-powered custom strategy if OpenAI is available
st.header("AI-Powered Custom Strategy")

# Check if OpenAI is available
if is_openai_available():
    # Prepare data for AI analysis
    if len(df_counterparties) > 0 and len(df_transactions) > 0:
        # Prepare risk profile
        risk_profile = {
            "risk_areas": risk_areas,
            "risk_tolerance": risk_tolerance,
            "time_horizon": time_horizon,
            "high_risk_counterparties_pct": high_risk_counterparties/total_counterparties if total_counterparties > 0 else 0,
            "high_risk_exposure_pct": exposure_pct if 'exposure_pct' in locals() else 0,
            "largest_counterparty": df_counterparties.iloc[0]["name"] if len(df_counterparties) > 0 else "Unknown",
        }
        
        # Get AI insights
        with st.spinner("Generating AI-powered custom strategy..."):
            insights = get_risk_insights(risk_profile, "Generate a custom risk mitigation strategy")
            
            if insights and "strategy" in insights:
                st.markdown(f"### Custom Strategy: {insights.get('strategy_name', 'Tailored Approach')}")
                st.markdown(insights.get("strategy", "Strategy generation failed. Please try again."))
                
                # Display key actions
                if "key_actions" in insights:
                    st.subheader("Key Actions")
                    for i, action in enumerate(insights["key_actions"], 1):
                        st.markdown(f"{i}. {action}")
                
                # Display expected outcomes
                if "expected_outcomes" in insights:
                    st.subheader("Expected Outcomes")
                    for outcome in insights["expected_outcomes"]:
                        st.markdown(f"- {outcome}")
            else:
                st.info("The AI could not generate a complete custom strategy. Using pre-defined recommendations instead.")
    else:
        st.info("Insufficient data available for AI-powered custom strategy generation.")
else:
    st.info("""
    AI-powered custom strategy generation requires an OpenAI API key.
    Configure your OpenAI API key to enable this feature and receive tailored
    risk mitigation strategies based on your specific risk profile.
    """)

# Implementation Plan Section
st.header("Your Implementation Plan")

if "implementation_plan" in st.session_state and st.session_state.implementation_plan:
    plan_df = pd.DataFrame(st.session_state.implementation_plan)
    
    # Allow user to prioritize
    st.subheader("Prioritize Your Plan")
    
    # Create columns for the dataframe
    col1, col2 = st.columns([3, 1])
    
    with col1:
        edited_df = st.data_editor(
            plan_df,
            column_config={
                "id": st.column_config.NumberColumn("ID", disabled=True),
                "title": st.column_config.TextColumn("Strategy Title", disabled=True),
                "area": st.column_config.TextColumn("Risk Area", disabled=True),
                "impact": st.column_config.TextColumn("Impact", disabled=True),
                "added_date": st.column_config.TextColumn("Date Added", disabled=True),
                "priority": st.column_config.SelectboxColumn(
                    "Priority",
                    options=["High", "Medium", "Low"],
                    default="Medium"
                ),
                "target_date": st.column_config.DateColumn("Target Date")
            },
            hide_index=True,
            num_rows="dynamic"
        )
    
    with col2:
        if st.button("Save Plan"):
            st.session_state.implementation_plan = edited_df.to_dict('records')
            st.success("Implementation plan updated successfully!")
    
    # Visualize the implementation plan
    if len(edited_df) > 0 and "priority" in edited_df.columns:
        # Count strategies by area
        area_counts = edited_df["area"].value_counts().reset_index()
        area_counts.columns = ["Area", "Count"]
        
        # Create visualization
        fig = px.pie(
            area_counts, 
            values="Count", 
            names="Area",
            title="Risk Mitigation Strategies by Area",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add option to export plan
        st.download_button(
            "Export Implementation Plan",
            data=edited_df.to_csv().encode('utf-8'),
            file_name=f"risk_mitigation_plan_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
else:
    st.info("Your implementation plan is empty. Add strategies from the suggestions above.")

# Resource Allocation Section
st.header("Resource Allocation Planner")

if "implementation_plan" in st.session_state and st.session_state.implementation_plan:
    # Create resource allocation columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Effort Allocation")
        
        # Create simple resource allocation sliders
        resources = {
            "Technology": st.slider("Technology Resources (%)", 0, 100, 30),
            "Personnel": st.slider("Personnel Resources (%)", 0, 100, 40),
            "External": st.slider("External Resources (%)", 0, 100, 20),
            "Training": st.slider("Training Resources (%)", 0, 100, 10)
        }
        
        # Check if total is 100%
        total = sum(resources.values())
        if total != 100:
            st.warning(f"Resource allocation doesn't add up to 100% (currently {total}%)")
        else:
            st.success("Resource allocation is balanced at 100%")
    
    with col2:
        st.subheader("Resource Allocation")
        
        # Create a pie chart for resource allocation
        resource_df = pd.DataFrame({
            "Resource": list(resources.keys()),
            "Allocation": list(resources.values())
        })
        
        fig = px.pie(
            resource_df,
            values="Allocation",
            names="Resource",
            title="Resource Allocation",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Add timeline visualization
    st.subheader("Implementation Timeline")
    
    # Create a fake timeline if target dates are not set
    timeline_df = pd.DataFrame(st.session_state.implementation_plan)
    
    if "target_date" not in timeline_df.columns or timeline_df["target_date"].isna().all():
        st.info("Set target dates in your implementation plan to visualize the timeline.")
    else:
        # Filter out rows with no target date
        timeline_df = timeline_df.dropna(subset=["target_date"])
        
        if len(timeline_df) > 0:
            # Convert to datetime if not already
            timeline_df["target_date"] = pd.to_datetime(timeline_df["target_date"])
            
            # Sort by target date
            timeline_df = timeline_df.sort_values("target_date")
            
            # Create a Gantt-like chart
            fig = px.timeline(
                timeline_df,
                x_start=datetime.now(),
                x_end="target_date",
                y="title",
                color="area",
                title="Implementation Timeline",
                labels={"title": "Strategy", "target_date": "Target Date"}
            )
            
            fig.update_yaxes(autorange="reversed")
            
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Add strategies to your implementation plan to use the resource allocation planner.")

# Risk Impact Simulation
st.header("Risk Impact Simulation")

# Import our ML-based risk prediction
from utils.risk_prediction import predict_risk_class, predict_financial_impact, analyze_mitigation_strategies, generate_risk_scenarios

# Allow user to simulate risk reduction
st.subheader("Simulate Risk Reduction")

# Risk reduction slider
risk_reduction = st.slider(
    "Estimated Risk Reduction (%)",
    min_value=0,
    max_value=100,
    value=30,
    help="Estimated percentage reduction in risk from implementing the suggested strategies"
)

# Use our ML model to get a more accurate risk score
if len(df_transactions) > 0:
    # Prepare risk data for the model
    risk_data = {
        'transaction_amount': df_transactions['amount'].mean() if 'amount' in df_transactions.columns else 100000,
        'days_since_last_transaction': 3,  # Placeholder
        'total_volume': df_transactions['amount'].sum() if 'amount' in df_transactions.columns else 1000000,
        'volatility': 0.2  # Placeholder
    }
    
    # Get ML-based risk predictions
    risk_prediction = predict_risk_class(risk_data)
    current_risk = risk_prediction['risk_score']
    
    # Display the ML-based risk classification
    st.info(f"ML Model Risk Classification: **{risk_prediction['risk_class'].title()}** (Confidence: {risk_prediction['confidence']:.2f})")
else:
    # Fallback if no transaction data
    current_risk = np.random.uniform(0.4, 0.8)

# Calculate simulated risk after implementing strategies
reduced_risk = current_risk * (1 - risk_reduction/100)

# Create columns for visualization
col1, col2 = st.columns(2)

with col1:
    # Create gauge for current risk
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_risk,
        title={"text": "Current Risk Level"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "firebrick"},
            "steps": [
                {"range": [0, 0.3], "color": "green"},
                {"range": [0.3, 0.7], "color": "yellow"},
                {"range": [0.7, 1], "color": "red"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": current_risk
            }
        },
        domain={"x": [0, 1], "y": [0, 1]}
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Create gauge for reduced risk
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=reduced_risk,
        title={"text": "Projected Risk After Mitigation"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "royalblue"},
            "steps": [
                {"range": [0, 0.3], "color": "green"},
                {"range": [0.3, 0.7], "color": "yellow"},
                {"range": [0.7, 1], "color": "red"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": reduced_risk
            }
        },
        domain={"x": [0, 1], "y": [0, 1]}
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Display projected financial impact
st.subheader("Projected Financial Impact")

# Use our ML model to predict financial impact
total_exposure = df_transactions["amount"].sum() if "amount" in df_transactions.columns else 1000000000

# Prepare impact prediction data
impact_data = {
    'exposure_amount': total_exposure,
    'risk_score': current_risk,
    'market_volatility': 0.2,  # Placeholder
    'days_to_maturity': time_horizon
}

# Get ML-based financial impact prediction
impact_prediction = predict_financial_impact(impact_data)
potential_loss = impact_prediction['predicted_impact']
mitigated_loss = potential_loss * (1 - risk_reduction/100)
savings = potential_loss - mitigated_loss

# Create columns for financial impact
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Potential Loss (Current)",
        f"${potential_loss/1000000:.2f}M",
        delta=f"-${potential_loss/1000000:.2f}M",
        delta_color="inverse"
    )

with col2:
    st.metric(
        "Potential Loss (Mitigated)",
        f"${mitigated_loss/1000000:.2f}M",
        delta=f"-${mitigated_loss/1000000:.2f}M",
        delta_color="inverse"
    )

with col3:
    st.metric(
        "Projected Savings",
        f"${savings/1000000:.2f}M",
        delta=f"+${savings/1000000:.2f}M",
        delta_color="normal"
    )

# Display confidence interval
st.info(f"Model Confidence Interval: ${impact_prediction['confidence_interval'][0]/1000000:.2f}M to ${impact_prediction['confidence_interval'][1]/1000000:.2f}M")

# Add risk scenario analysis
st.subheader("Risk Scenario Analysis")

# Generate risk scenarios
risk_profile = {'risk_score': current_risk}
scenarios = generate_risk_scenarios(risk_profile, num_scenarios=3)

# Create tabs for different scenarios
scenario_tabs = st.tabs([scenario['name'] for scenario in scenarios])

# Display each scenario
for i, (tab, scenario) in enumerate(zip(scenario_tabs, scenarios)):
    with tab:
        st.markdown(f"**Description:** {scenario['description']}")
        
        # Create columns for scenario metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk score comparison
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Base Risk', 'Stressed Risk'],
                y=[scenario['base_risk_score'], scenario['stressed_risk_score']],
                marker_color=['royalblue', 'firebrick']
            ))
            
            fig.update_layout(
                title="Risk Score Comparison",
                yaxis_range=[0, 1],
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key risk metrics
            st.markdown(f"**Risk Increase:** {scenario['risk_increase']:.4f} ({scenario['risk_increase_pct']:.1%})")
            st.markdown(f"**Confidence Level:** {scenario['confidence_level']:.1%}")
        
        with col2:
            # Financial impact
            fig = go.Figure(go.Indicator(
                mode="number+delta",
                value=scenario['financial_impact']/1000000,
                number={'prefix': "$", 'suffix': "M"},
                delta={'position': "top", 'reference': potential_loss/1000000},
                title={"text": "Financial Impact"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            
            fig.update_layout(height=300)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key risk factors
            st.markdown("**Key Risk Factors:**")
            for factor in scenario['key_factors']:
                st.markdown(f"- {factor.title()}")
        
        # Mitigation effectiveness in this scenario
        effectiveness = max(0, min(100, 100 * (1 - (scenario['risk_increase_pct'] * (1 - risk_reduction/100)))))
        
        st.markdown("#### Mitigation Effectiveness in This Scenario")
        
        # Create effectiveness gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=effectiveness,
            title={"text": "Strategy Effectiveness (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkgreen" if effectiveness > 70 else "orange" if effectiveness > 40 else "red"},
                "steps": [
                    {"range": [0, 40], "color": "lightcoral"},
                    {"range": [40, 70], "color": "khaki"},
                    {"range": [70, 100], "color": "lightgreen"}
                ]
            },
            domain={'x': [0.1, 0.9], 'y': [0, 1]}
        ))
        
        fig.update_layout(height=200)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation based on scenario
        if effectiveness < 40:
            st.error("**Recommendation:** Current mitigation strategies are insufficient for this scenario. Additional specialized measures are needed.")
        elif effectiveness < 70:
            st.warning("**Recommendation:** Current mitigation strategies provide partial protection. Consider additional targeted measures.")
        else:
            st.success("**Recommendation:** Current mitigation strategies provide good protection against this scenario.")

# Final action button
st.header("Take Action")

if st.button("Generate Comprehensive Mitigation Report", type="primary"):
    st.success("Report generated successfully!")
    
    # Show a sample report structure
    st.markdown("""
    ## Sample Report Structure:
    
    ### 1. Executive Summary
    - Current risk profile overview
    - Key recommendations
    - Projected impact summary
    
    ### 2. Detailed Risk Analysis
    - Risk area breakdown
    - Critical risk factors
    - Exposure quantification
    
    ### 3. Mitigation Strategy
    - Prioritized action items
    - Resource requirements
    - Implementation timeline
    
    ### 4. Expected Outcomes
    - Risk reduction projections
    - Financial impact analysis
    - Key performance indicators
    
    ### 5. Monitoring Framework
    - Tracking metrics
    - Reporting structure
    - Feedback mechanism
    """)
    
    # Download button
    st.download_button(
        "Download Full Report",
        data="This would be a full report in a real implementation",
        file_name=f"risk_mitigation_report_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf"
    )

# Reference section
st.sidebar.markdown("---")
st.sidebar.markdown("### Reference")
st.sidebar.info("""
The Smart Suggestions Panel uses a combination of rules-based recommendations and AI-powered insights to provide actionable strategies for mitigating identified risks.

The suggestions are tailored based on:
- Selected risk areas
- Risk tolerance setting
- Time horizon preferences
- Current risk exposure profile
""")