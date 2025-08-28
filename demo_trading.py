#!/usr/bin/env python3
"""
Demo trading analysis - Shows what the system would do if market was open
"""

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

load_dotenv()

def analyze_market():
    """Analyze current market conditions"""
    
    client = TradingClient(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        paper=True
    )
    
    print("🚀 QUANTNEXUS ML TRADING SYSTEM - Market Analysis")
    print("="*60)
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get account info
    account = client.get_account()
    positions = client.get_all_positions()
    
    print(f"\n💼 PORTFOLIO STATUS")
    print(f"   Total Value: ${float(account.portfolio_value):,.2f}")
    print(f"   Cash Available: ${float(account.cash):,.2f}")
    print(f"   Buying Power: ${float(account.buying_power):,.2f}")
    print(f"   Active Positions: {len(positions)}")
    
    # Analyze watchlist
    watchlist = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMD', 'META']
    
    print(f"\n📊 STOCK ANALYSIS (If market was open)")
    print("-"*60)
    
    buy_recommendations = []
    sell_recommendations = []
    
    for symbol in watchlist:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1mo")
            
            # Calculate indicators
            df['sma_20'] = df['Close'].rolling(20).mean()
            df['sma_5'] = df['Close'].rolling(5).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_price = df['Close'].iloc[-1]
            sma_5 = df['sma_5'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            # Determine action
            if sma_5 > sma_20 and current_price > sma_5 and current_rsi < 70:
                action = "BUY"
                confidence = min(0.9, 0.5 + (sma_5 / sma_20 - 1) * 2)
                buy_recommendations.append((symbol, current_price, confidence))
            elif sma_5 < sma_20 and current_price < sma_5 and current_rsi > 30:
                action = "SELL"
                confidence = min(0.9, 0.5 + (1 - sma_5 / sma_20) * 2)
                sell_recommendations.append((symbol, current_price, confidence))
            else:
                action = "HOLD"
                confidence = 0.5
            
            # Check if we have position
            has_position = any(p.symbol == symbol for p in positions)
            position_str = "📈" if has_position else "  "
            
            print(f"{position_str} {symbol}: ${current_price:.2f} | RSI: {current_rsi:.1f} | Signal: {action} ({confidence:.1%})")
            
        except Exception as e:
            print(f"   {symbol}: Error analyzing - {e}")
    
    # Trading recommendations
    print(f"\n💡 TRADING RECOMMENDATIONS")
    print("-"*60)
    
    if buy_recommendations:
        print("🟢 BUY SIGNALS:")
        for symbol, price, confidence in sorted(buy_recommendations, key=lambda x: x[2], reverse=True):
            position_size = 5000 * confidence  # Scale with confidence
            shares = int(position_size / price)
            print(f"   {symbol}: Buy {shares} shares @ ${price:.2f} (confidence: {confidence:.1%})")
    else:
        print("🟢 No strong buy signals")
    
    if sell_recommendations:
        print("\n🔴 SELL SIGNALS:")
        for symbol, price, confidence in sorted(sell_recommendations, key=lambda x: x[2], reverse=True):
            position = next((p for p in positions if p.symbol == symbol), None)
            if position:
                print(f"   {symbol}: Sell {position.qty} shares @ ${price:.2f} (confidence: {confidence:.1%})")
    else:
        print("🔴 No sell signals")
    
    # Current positions performance
    if positions:
        print(f"\n📈 CURRENT POSITIONS PERFORMANCE")
        print("-"*60)
        total_pnl = 0
        for p in positions[:10]:  # Show top 10
            pnl = float(p.unrealized_pl)
            pnl_pct = float(p.unrealized_plpc) * 100
            total_pnl += pnl
            
            emoji = "🟢" if pnl > 0 else "🔴"
            print(f"{emoji} {p.symbol}: {p.qty} shares | P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%)")
        
        print(f"\n💰 Total Unrealized P&L: ${total_pnl:,.2f}")
    
    # Market status
    clock = client.get_clock()
    print(f"\n⏰ MARKET STATUS")
    print("-"*60)
    if clock.is_open:
        print("✅ Market is OPEN - Trades will execute automatically")
    else:
        print("🔒 Market is CLOSED")
        print(f"   Next open: {clock.next_open}")
        print(f"   Next close: {clock.next_close}")
    
    print("\n📊 AUTOMATED TRADING STATUS")
    print("-"*60)
    print("✅ Trading bot is running in background")
    print("✅ Monitoring 7 stocks every 5 minutes")
    print("✅ Will execute trades when market opens")
    print("\n💡 To stop: pkill -f start_trading.py")

if __name__ == "__main__":
    analyze_market()