#!/usr/bin/env python3
"""
Quick test to verify MCP is working
"""

print("Testing MCP installation...")

try:
    # Test MCP imports
    from mcp import Server, Tool, Resource, Prompt
    print("✅ MCP core imports successful")
    
    # Test FastAPI imports
    from fastapi import FastAPI
    print("✅ FastAPI import successful")
    
    # Test other dependencies
    import uvicorn
    import websockets
    import aiohttp
    print("✅ All networking dependencies imported")
    
    # Test ML libraries
    import numpy as np
    import pandas as pd
    import xgboost
    import lightgbm
    print("✅ ML libraries imported")
    
    # Test trading libraries
    import yfinance as yf
    from alpaca.trading.client import TradingClient
    print("✅ Trading libraries imported")
    
    print("\n🎉 All dependencies are working correctly!")
    print("\nYou can now run:")
    print("  make mcp-server   # Start the MCP server")
    print("  make mcp-client   # Run the example client")
    
except ImportError as e:
    print(f"❌ Error importing: {e}")
    print("\nPlease run: make setup")