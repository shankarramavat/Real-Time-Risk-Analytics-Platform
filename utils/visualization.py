import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt

def plot_risk_heatmap(df, x_column, y_column, value_column, title="Risk Heatmap"):
    """
    Create a heatmap visualization for risk data
    
    Parameters:
    - df: DataFrame containing the data
    - x_column: Column name for x-axis
    - y_column: Column name for y-axis
    - value_column: Column name for values (color intensity)
    - title: Title for the heatmap
    
    Returns:
    - Plotly figure
    """
    # Create pivot table for heatmap
    pivot_df = df.pivot_table(index=y_column, columns=x_column, values=value_column, aggfunc='mean')
    
    # Create heatmap
    fig = px.imshow(
        pivot_df,
        color_continuous_scale='RdYlGn_r',  # Red (high risk) to Green (low risk)
        title=title,
        labels={"color": "Risk Score"}
    )
    
    fig.update_layout(
        height=600,
        xaxis_title=x_column,
        yaxis_title=y_column,
        coloraxis_colorbar=dict(title="Risk Score")
    )
    
    return fig

def plot_time_series(dates, values, title="Risk Over Time", y_label="Risk Score"):
    """
    Create a time series plot
    
    Parameters:
    - dates: List of dates
    - values: List of values
    - title: Title for the plot
    - y_label: Label for y-axis
    
    Returns:
    - Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name=y_label,
            line=dict(width=2, color='#0078D7')
        )
    )
    
    fig.update_layout(
        title=title,
        height=400,
        xaxis_title="Date",
        yaxis_title=y_label,
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        ),
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

def plot_risk_distribution(risk_scores, title="Risk Score Distribution"):
    """
    Create a histogram of risk scores
    
    Parameters:
    - risk_scores: List or array of risk scores
    - title: Title for the histogram
    
    Returns:
    - Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(
            x=risk_scores,
            nbinsx=20,
            marker_color='#0078D7',
            opacity=0.7
        )
    )
    
    # Add a vertical line for the mean
    mean_risk = np.mean(risk_scores)
    
    fig.add_vline(
        x=mean_risk,
        line_color="red",
        line_dash="dash",
        annotation_text=f"Mean: {mean_risk:.2f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=title,
        height=400,
        xaxis_title="Risk Score",
        yaxis_title="Frequency",
        bargap=0.1,
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        ),
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

def plot_radar_chart(categories, values, title="Risk Radar"):
    """
    Create a radar chart for risk categories
    
    Parameters:
    - categories: List of risk categories
    - values: List of risk values corresponding to categories
    - title: Title for the radar chart
    
    Returns:
    - Plotly figure
    """
    # Ensure the radar chart is closed by repeating the first value
    categories = categories + [categories[0]]
    values = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(0, 120, 215, 0.3)',
            line=dict(color='#0078D7', width=2),
            name='Risk Profile'
        )
    )
    
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        height=500,
        showlegend=False
    )
    
    return fig

def plot_counterparty_network(nodes_df, edges_df, highlight_node=None):
    """
    Create a network visualization of counterparty relationships
    
    Parameters:
    - nodes_df: DataFrame with node information (id, name, risk_score, etc.)
    - edges_df: DataFrame with edge information (source_id, target_id, strength, etc.)
    - highlight_node: Node ID to highlight
    
    Returns:
    - Plotly figure
    """
    # Create a figure
    fig = go.Figure()
    
    # Create a colorscale for nodes based on risk score
    colorscale = [
        [0, 'green'],
        [0.4, 'yellow'],
        [0.6, 'orange'],
        [1, 'red']
    ]
    
    # Add edges as lines
    for _, edge in edges_df.iterrows():
        source_id = edge['source_id']
        target_id = edge['target_id']
        
        source_node = nodes_df[nodes_df['id'] == source_id].iloc[0]
        target_node = nodes_df[nodes_df['id'] == target_id].iloc[0]
        
        # Generate random positions if not available
        source_x = source_node.get('x', np.random.uniform(-1, 1))
        source_y = source_node.get('y', np.random.uniform(-1, 1))
        target_x = target_node.get('x', np.random.uniform(-1, 1))
        target_y = target_node.get('y', np.random.uniform(-1, 1))
        
        # Line width based on relationship strength
        width = 1 + 3 * edge.get('strength', 0.5)
        
        fig.add_trace(
            go.Scatter(
                x=[source_x, target_x],
                y=[source_y, target_y],
                mode='lines',
                line=dict(width=width, color='rgba(150, 150, 150, 0.5)'),
                hoverinfo='skip',
                showlegend=False
            )
        )
    
    # Add nodes as scatter points
    for _, node in nodes_df.iterrows():
        node_id = node['id']
        
        # Generate random positions if not available
        x = node.get('x', np.random.uniform(-1, 1))
        y = node.get('y', np.random.uniform(-1, 1))
        
        # Determine size and color based on node properties
        size = 15
        color = node.get('risk_score', 0.5)
        
        # Highlight the selected node if specified
        border_width = 2
        border_color = 'black'
        
        if highlight_node is not None and node_id == highlight_node:
            size = 20
            border_width = 3
            border_color = 'blue'
        
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    size=size,
                    color=color,
                    colorscale=colorscale,
                    line=dict(width=border_width, color=border_color),
                    showscale=True,
                    colorbar=dict(title="Risk Score")
                ),
                text=node.get('name', f"Node {node_id}"),
                hoverinfo='text',
                name=f"Node {node_id}"
            )
        )
    
    # Update layout
    fig.update_layout(
        title="Counterparty Network",
        height=600,
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

def plot_risk_map(df, lat_col, lon_col, color_col, hover_data=None, title="Geographic Risk Distribution"):
    """
    Create a map visualization of risk by geographic location
    
    Parameters:
    - df: DataFrame containing the data
    - lat_col: Column name for latitude
    - lon_col: Column name for longitude
    - color_col: Column name for color (usually risk score)
    - hover_data: List of column names to include in hover information
    - title: Title for the map
    
    Returns:
    - Plotly figure
    """
    if hover_data is None:
        hover_data = []
    
    fig = px.scatter_mapbox(
        df,
        lat=lat_col,
        lon=lon_col,
        color=color_col,
        color_continuous_scale='RdYlGn_r',  # Red (high risk) to Green (low risk)
        hover_name=df.index,
        hover_data=hover_data,
        size_max=15,
        zoom=1,
        title=title
    )
    
    fig.update_layout(
        mapbox_style="carto-positron",
        height=600,
        margin=dict(l=10, r=10, t=50, b=10),
        coloraxis_colorbar=dict(title="Risk Score")
    )
    
    return fig

def plot_compliance_status(df, status_col='status', title="Compliance Status Distribution"):
    """
    Create a pie chart of compliance statuses
    
    Parameters:
    - df: DataFrame containing compliance data
    - status_col: Column name for compliance status
    - title: Title for the pie chart
    
    Returns:
    - Plotly figure
    """
    # Count the number of entries for each status
    status_counts = df[status_col].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    # Define colors for different statuses
    status_colors = {
        'Compliant': 'green',
        'Pending Review': 'yellow',
        'Non-Compliant': 'red',
        'Exempt': 'blue',
        'Under Investigation': 'orange'
    }
    
    # Get colors for each status
    colors = [status_colors.get(status, 'gray') for status in status_counts['Status']]
    
    fig = go.Figure(
        data=[
            go.Pie(
                labels=status_counts['Status'],
                values=status_counts['Count'],
                hole=0.4,
                marker_colors=colors
            )
        ]
    )
    
    fig.update_layout(
        title=title,
        height=400,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

def plot_stress_test_results(base_values, stressed_values, categories, title="Stress Test Impact"):
    """
    Create a bar chart comparing base values to stressed values
    
    Parameters:
    - base_values: List of base values before stress test
    - stressed_values: List of values after applying stress test
    - categories: List of category names
    - title: Title for the chart
    
    Returns:
    - Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=categories,
            y=base_values,
            name='Base Case',
            marker_color='rgb(55, 83, 109)'
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=categories,
            y=stressed_values,
            name='Stressed Case',
            marker_color='rgb(219, 64, 82)'
        )
    )
    
    fig.update_layout(
        title=title,
        height=500,
        xaxis_title="Category",
        yaxis_title="Value",
        barmode='group',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        ),
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig
