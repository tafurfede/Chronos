#!/usr/bin/env python3
"""
Quick trading script using your Alpaca account
"""

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import yfinance as yf
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

class QuickTrader:
    def __init__(self):
        self.client = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=True
        )
        
    def analyze_stock(self, symbol):
        """Simple technical analysis"""
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1mo")
        
        # Calculate indicators
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA5'] = df['Close'].rolling(5).mean()
        
        current_price = df['Close'].iloc[-1]
        sma5 = df['SMA5'].iloc[-1]
        sma20 = df['SMA20'].iloc[-1]
        
        # Simple signal
        if sma5 > sma20 and current_price > sma5:
            signal = "BUY"
            confidence = 0.7
        elif sma5 < sma20 and current_price < sma5:
            signal = "SELL"
            confidence = 0.7
        else:
            signal = "HOLD"
            confidence = 0.5
            
        return {
            'symbol': symbol,
            'price': current_price,
            'signal': signal,
            'confidence': confidence,
            'sma5': sma5,
            'sma20': sma20
        }
    
    def execute_trade(self, symbol, qty, side):
        """Execute a trade"""
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        
        order = self.client.submit_order(order_data)
        return order
    
    def get_portfolio(self):
        """Get current portfolio"""
        account = self.client.get_account()
        positions = self.client.get_all_positions()
        
        return {
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'buying_power': float(account.buying_power),
            'positions': [
                {
                    'symbol': p.symbol,
                    'qty': float(p.qty),
                    'value': float(p.market_value),
                    'pnl': float(p.unrealized_pl),
                    'pnl_pct': float(p.unrealized_plpc) * 100
                }
                for p in positions
            ]
        }

def main():
    print("🚀 QuantNexus Quick Trading System")
    print("="*50)
    
    trader = QuickTrader()
    
    # Get portfolio status
    portfolio = trader.get_portfolio()
    print(f"\n💼 Portfolio Status:")
    print(f"   Cash: ${portfolio['cash']:,.2f}")
    print(f"   Total Value: ${portfolio['portfolio_value']:,.2f}")
    print(f"   Positions: {len(portfolio['positions'])}")
    
    if portfolio['positions']:
        print("\n📊 Current Positions:")
        for pos in portfolio['positions'][:5]:
            print(f"   {pos['symbol']}: {pos['qty']} shares, P&L: ${pos['pnl']:,.2f} ({pos['pnl_pct']:.1f}%)")
    
    # Analyze some stocks
    print("\n🔍 Stock Analysis:")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
    
    buy_signals = []
    for symbol in symbols:
        analysis = trader.analyze_stock(symbol)
        print(f"   {analysis['symbol']}: ${analysis['price']:.2f} - {analysis['signal']} (confidence: {analysis['confidence']:.1%})")
        
        if analysis['signal'] == 'BUY' and analysis['confidence'] > 0.65:
            buy_signals.append(analysis)
    
    if buy_signals:
        print(f"\n💡 Found {len(buy_signals)} buy signals!")
        print("   Run 'make trade' to start automated trading")
    else:
        print("\n⏸️  No strong buy signals at the moment")
    
    print("\n📈 Next Steps:")
    print("   1. Review positions in Alpaca dashboard")
    print("   2. Run 'make trade' for automated trading")
    print("   3. Monitor with 'make performance-report'")

if __name__ == "__main__":
    main()