#!/bin/bash

echo "🚀 QuantNexus ML Trading System - Quick Start Guide"
echo "=================================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Please configure your .env file first:"
    echo "   1. Copy .env file we just created"
    echo "   2. Add your Alpaca API credentials"
    echo "   3. Get free API keys at: https://alpaca.markets/"
    echo ""
    exit 1
fi

echo "📋 Step-by-step setup:"
echo ""
echo "1️⃣  Configure Alpaca API (Paper Trading):"
echo "   - Sign up at https://alpaca.markets/"
echo "   - Get your API keys from the dashboard"
echo "   - Update .env file with your credentials"
echo ""

echo "2️⃣  Train ML Models:"
echo "   make train"
echo "   This will train XGBoost, LightGBM, and Neural Network models"
echo ""

echo "3️⃣  Run Backtest:"
echo "   make backtest"
echo "   Test your strategy on historical data"
echo ""

echo "4️⃣  Start Paper Trading:"
echo "   make trade"
echo "   Begin live paper trading with Alpaca"
echo ""

echo "5️⃣  Monitor Performance:"
echo "   make performance-report"
echo "   Generate performance analytics"
echo ""

echo "📊 Current Server Status:"
if curl -s http://localhost:8000/health > /dev/null; then
    echo "   ✅ API Server is running at http://localhost:8000"
    echo "   📚 API Docs: http://localhost:8000/docs"
else
    echo "   ❌ API Server is not running"
    echo "   Start it with: make mcp-server"
fi

echo ""
echo "🎯 Quick Test Commands:"
echo "   - Test API: venv/bin/python3 test_api_client.py"
echo "   - Check logs: make logs"
echo "   - Stop server: pkill -f api_trading_server.py"
echo ""

echo "💡 Pro Tips:"
echo "   - Start with paper trading to validate your strategy"
echo "   - Monitor the system for at least 1 week before live trading"
echo "   - The system targets 65% win rate with proper configuration"
echo "   - Use Kelly Criterion for optimal position sizing"
echo ""

echo "📖 Documentation:"
echo "   - Blueprint: ML_TRADING_SYSTEM_COMPLETE_BLUEPRINT.md"
echo "   - README: README.md"
echo ""

echo "Need help? Check the documentation or run: make help"