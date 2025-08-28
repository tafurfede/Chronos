#!/usr/bin/env python3
"""
Test client for QuantNexus Trading API
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_api():
    print("🔍 Testing QuantNexus Trading API...")
    print("="*50)
    
    # Test root endpoint
    print("\n1. Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test stock analysis
    print("\n2. Testing stock analysis...")
    response = requests.post(
        f"{BASE_URL}/analyze",
        json={"symbol": "AAPL", "period": "1d"}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Symbol: {data['symbol']}")
        print(f"Current Price: ${data['current_price']}")
        print(f"Signal: {data['ml_prediction']['signal']}")
        print(f"Confidence: {data['ml_prediction']['confidence']:.1%}")
        print(f"Recommendation: {data['recommendation']}")
    
    # Test market signals
    print("\n3. Testing market signals...")
    response = requests.get(f"{BASE_URL}/signals?min_confidence=0.6")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Found {data['signal_count']} signals")
        for signal in data['signals'][:3]:
            print(f"  {signal['symbol']}: {signal['signal']} (confidence: {signal['confidence']:.1%})")
    
    # Test portfolio
    print("\n4. Testing portfolio...")
    response = requests.get(f"{BASE_URL}/portfolio")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Cash: ${data['cash']}")
        print(f"Total Value: ${data['total_value']}")
        print(f"Positions: {data['positions_count']}")
    
    # Test trade execution
    print("\n5. Testing trade execution...")
    response = requests.post(
        f"{BASE_URL}/trade",
        json={
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 10
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Executed: {data['action'].upper()} {data['quantity']} {data['symbol']} @ ${data['price']}")
    
    # Test backtest
    print("\n6. Testing backtest...")
    response = requests.post(
        f"{BASE_URL}/backtest",
        json={
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 10000
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Total Return: {data['total_return']}%")
        print(f"Win Rate: {data['win_rate']}%")
        print(f"Sharpe Ratio: {data['sharpe_ratio']}")
    
    print("\n✅ API tests complete!")

if __name__ == "__main__":
    test_api()