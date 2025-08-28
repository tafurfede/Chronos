#!/usr/bin/env python3
"""
Test Alpaca API connection and verify all integrations
"""

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
import requests
import json

# Load environment variables
load_dotenv()

def test_alpaca():
    """Test Alpaca API connection"""
    print("🔍 Testing Alpaca API Connection...")
    
    try:
        # Initialize Alpaca client
        trading_client = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=True
        )
        
        # Get account info
        account = trading_client.get_account()
        print(f"✅ Alpaca Connected!")
        print(f"   Account Status: {account.status}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        
        # Get some tradable assets
        search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
        assets = trading_client.get_all_assets(search_params)
        tradable = [a for a in assets if a.tradable][:5]
        print(f"   Tradable Assets: {len([a for a in assets if a.tradable])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Alpaca Error: {e}")
        return False

def test_openai():
    """Test OpenAI API connection"""
    print("\n🔍 Testing OpenAI API...")
    
    try:
        import openai
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Note: Actual API call would cost money, so we just verify the key format
        if openai.api_key and openai.api_key.startswith('sk-'):
            print("✅ OpenAI API key configured")
            return True
        else:
            print("⚠️  OpenAI API key not configured")
            return False
            
    except Exception as e:
        print(f"⚠️  OpenAI setup issue: {e}")
        return False

def test_news_api():
    """Test News API connection"""
    print("\n🔍 Testing News API...")
    
    try:
        api_key = os.getenv('NEWS_API_KEY')
        url = f"https://newsapi.org/v2/everything?q=AAPL&apiKey={api_key}&pageSize=1"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ News API Connected!")
            print(f"   Total Articles Available: {data.get('totalResults', 0)}")
            return True
        else:
            print(f"⚠️  News API Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"⚠️  News API Error: {e}")
        return False

def test_alpha_vantage():
    """Test Alpha Vantage API"""
    print("\n🔍 Testing Alpha Vantage API...")
    
    try:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'Global Quote' in data:
                print("✅ Alpha Vantage Connected!")
                return True
            else:
                print("⚠️  Alpha Vantage: API limit or error")
                return False
        else:
            print(f"⚠️  Alpha Vantage Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"⚠️  Alpha Vantage Error: {e}")
        return False

def main():
    print("="*60)
    print("🚀 QuantNexus Trading System - API Integration Test")
    print("="*60)
    
    results = {
        "Alpaca": test_alpaca(),
        "OpenAI": test_openai(),
        "News API": test_news_api(),
        "Alpha Vantage": test_alpha_vantage()
    }
    
    print("\n" + "="*60)
    print("📊 Summary:")
    print("="*60)
    
    for api, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {api}: {'Connected' if status else 'Not Connected'}")
    
    if results["Alpaca"]:
        print("\n🎉 Core trading system ready!")
        print("You can now:")
        print("  1. Train models: make train")
        print("  2. Run backtest: make backtest")
        print("  3. Start paper trading: make trade")
    else:
        print("\n⚠️  Please check your Alpaca API credentials in .env file")
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)