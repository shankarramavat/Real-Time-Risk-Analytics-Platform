import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import json

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

def generate_counterparties(n=50):
    """Generate sample counterparty data"""
    np.random.seed(42)  # For reproducibility
    
    countries = ['USA', 'UK', 'Germany', 'France', 'Japan', 'China', 'India', 'Brazil', 'Australia', 'Canada']
    sectors = ['Banking', 'Insurance', 'Asset Management', 'Hedge Fund', 'Pension Fund', 'Corporate', 'Government']
    credit_ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB']
    
    counterparties = []
    
    for i in range(1, n+1):
        counterparty = {
            'id': i,
            'name': f"Counterparty {i}",
            'country': np.random.choice(countries),
            'sector': np.random.choice(sectors),
            'credit_rating': np.random.choice(credit_ratings, p=[0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]),
            'exposure_limit': np.random.randint(1000000, 50000000),
            'current_exposure': np.random.randint(100000, 40000000),
            'risk_score': np.random.uniform(0.1, 0.9),
            'last_review_date': (datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d')
        }
        counterparties.append(counterparty)
    
    return pd.DataFrame(counterparties)

def generate_transactions(n=200, counterparties_df=None):
    """Generate sample transaction data"""
    np.random.seed(43)  # For reproducibility
    
    if counterparties_df is None:
        counterparties_df = generate_counterparties()
    
    counterparty_ids = counterparties_df['id'].tolist()
    transaction_types = ['Swap', 'Forward', 'Option', 'Repo', 'Bond', 'Equity', 'Loan']
    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD']
    
    transactions = []
    
    # Current date and time
    now = datetime.now()
    
    for i in range(1, n+1):
        # Random time in the last 24 hours
        transaction_time = now - timedelta(hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60))
        
        amount = np.random.randint(10000, 5000000)
        counterparty_id = np.random.choice(counterparty_ids)
        
        # Get corresponding counterparty name
        counterparty_name = counterparties_df[counterparties_df['id'] == counterparty_id]['name'].values[0]
        
        transaction = {
            'id': i,
            'counterparty_id': counterparty_id,
            'counterparty_name': counterparty_name,
            'transaction_type': np.random.choice(transaction_types),
            'amount': amount,
            'currency': np.random.choice(currencies),
            'timestamp': transaction_time.strftime('%Y-%m-%d %H:%M:%S'),
            'risk_score': np.random.uniform(0.1, 0.9),
            'flagged': np.random.choice([True, False], p=[0.05, 0.95])
        }
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)

def generate_market_data(n=100):
    """Generate sample market data for different sectors and regions"""
    np.random.seed(44)  # For reproducibility
    
    sectors = ['Banking', 'Insurance', 'Technology', 'Healthcare', 'Energy', 'Utilities', 'Consumer Goods', 'Telecommunications']
    regions = ['North America', 'Europe', 'Asia', 'Latin America', 'Middle East', 'Africa', 'Oceania']
    
    now = datetime.now()
    timestamps = [(now - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
    
    market_data = []
    
    for sector in sectors:
        for region in regions:
            # Base volatility and market risk for this sector/region combination
            base_volatility = np.random.uniform(0.05, 0.2)
            base_market_risk = np.random.uniform(0.2, 0.8)
            base_liquidity = np.random.uniform(0.3, 0.9)
            
            for timestamp in timestamps:
                # Add some random variation day to day
                volatility = max(0, min(1, base_volatility + np.random.uniform(-0.02, 0.02)))
                market_risk = max(0, min(1, base_market_risk + np.random.uniform(-0.05, 0.05)))
                liquidity = max(0, min(1, base_liquidity + np.random.uniform(-0.03, 0.03)))
                
                market_data_point = {
                    'date': timestamp,
                    'sector': sector,
                    'region': region,
                    'volatility': volatility,
                    'market_risk': market_risk,
                    'liquidity': liquidity,
                    'index_value': np.random.uniform(800, 1500),
                    'change_pct': np.random.uniform(-3, 3)
                }
                market_data.append(market_data_point)
    
    return pd.DataFrame(market_data)

def generate_compliance_data(n=50, counterparties_df=None):
    """Generate sample compliance data"""
    np.random.seed(45)  # For reproducibility
    
    if counterparties_df is None:
        counterparties_df = generate_counterparties()
    
    counterparty_ids = counterparties_df['id'].tolist()
    compliance_types = ['KYC', 'AML', 'Sanctions', 'Regulatory', 'Internal Policy']
    status_options = ['Compliant', 'Pending Review', 'Non-Compliant', 'Exempt', 'Under Investigation']
    
    compliance_data = []
    
    now = datetime.now()
    
    for i in range(1, n+1):
        counterparty_id = np.random.choice(counterparty_ids)
        counterparty_name = counterparties_df[counterparties_df['id'] == counterparty_id]['name'].values[0]
        
        # Determine if this should be flagged (5% chance)
        is_flagged = np.random.random() < 0.05
        
        # Status is more likely to be non-compliant if flagged
        if is_flagged:
            status = np.random.choice(status_options, p=[0.1, 0.2, 0.4, 0.1, 0.2])
        else:
            status = np.random.choice(status_options, p=[0.7, 0.2, 0.02, 0.05, 0.03])
        
        next_review_date = (now + timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d')
        
        # Risk score higher for non-compliant statuses
        if status in ['Non-Compliant', 'Under Investigation']:
            risk_score = np.random.uniform(0.7, 0.95)
        elif status == 'Pending Review':
            risk_score = np.random.uniform(0.4, 0.7)
        else:
            risk_score = np.random.uniform(0.1, 0.4)
        
        compliance_entry = {
            'id': i,
            'counterparty_id': counterparty_id,
            'counterparty_name': counterparty_name,
            'compliance_type': np.random.choice(compliance_types),
            'status': status,
            'risk_score': risk_score,
            'last_check_date': (now - timedelta(days=np.random.randint(1, 90))).strftime('%Y-%m-%d'),
            'next_review_date': next_review_date,
            'notes': f"Compliance review for {counterparty_name}",
            'flagged': is_flagged
        }
        compliance_data.append(compliance_entry)
    
    return pd.DataFrame(compliance_data)

def generate_sample_data():
    """Generate all sample data and save to CSV files"""
    # Generate counterparties first so we can use them in other datasets
    counterparties_df = generate_counterparties(50)
    transactions_df = generate_transactions(200, counterparties_df)
    market_data_df = generate_market_data()
    compliance_df = generate_compliance_data(50, counterparties_df)
    
    # Save to CSV files
    counterparties_df.to_csv('data/counterparties.csv', index=False)
    transactions_df.to_csv('data/transactions.csv', index=False)
    market_data_df.to_csv('data/market_data.csv', index=False)
    compliance_df.to_csv('data/compliance.csv', index=False)
    
    # Create some relationships for graph visualization
    relationships = []
    for i in range(1, 40):
        # Each counterparty has 1-3 relationships with other counterparties
        counterparty_id = i
        num_relationships = np.random.randint(1, 4)
        
        for _ in range(num_relationships):
            related_id = np.random.randint(1, 51)
            while related_id == counterparty_id:
                related_id = np.random.randint(1, 51)
            
            relationship = {
                'source_id': counterparty_id,
                'target_id': related_id,
                'relationship_type': np.random.choice(['Trading Partner', 'Credit Provider', 'Subsidiary', 'Parent', 'Joint Venture']),
                'strength': np.random.uniform(0.1, 1.0)
            }
            relationships.append(relationship)
    
    relationships_df = pd.DataFrame(relationships)
    relationships_df.to_csv('data/relationships.csv', index=False)

def load_data(data_type):
    """Load data from CSV files"""
    try:
        if data_type == "counterparties":
            return pd.read_csv('data/counterparties.csv')
        elif data_type == "transactions":
            return pd.read_csv('data/transactions.csv')
        elif data_type == "market_data":
            return pd.read_csv('data/market_data.csv')
        elif data_type == "compliance":
            return pd.read_csv('data/compliance.csv')
        elif data_type == "relationships":
            return pd.read_csv('data/relationships.csv')
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    except FileNotFoundError:
        # If file doesn't exist, generate sample data first
        generate_sample_data()
        return load_data(data_type)

if __name__ == "__main__":
    generate_sample_data()
