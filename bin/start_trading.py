#!/usr/bin/env python3
"""
Start automated trading with the QuantNexus system
"""

import os
import sys
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AutoTrader:
    def __init__(self):
        self.client = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=True
        )
        self.positions = {}
        self.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMD', 'META']
        
    def calculate_signal(self, df):
        """Calculate trading signal based on technical indicators"""
        # Calculate moving averages
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['rsi'] = self.calculate_rsi(df['Close'])
        
        # Get latest values
        current_price = df['Close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        
        # Generate signal
        signal_strength = 0
        
        # Trend following
        if current_price > sma_20:
            signal_strength += 0.3
        if sma_20 > sma_50:
            signal_strength += 0.3
        
        # RSI conditions
        if rsi < 30:  # Oversold
            signal_strength += 0.4
        elif rsi > 70:  # Overbought
            signal_strength -= 0.4
            
        # Volume confirmation
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        if df['Volume'].iloc[-1] > avg_volume * 1.5:
            signal_strength *= 1.2
            
        return signal_strength
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def analyze_stocks(self):
        """Analyze all stocks in watchlist"""
        signals = []
        
        for symbol in self.watchlist:
            try:
                # Fetch data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="2mo")
                
                if len(df) < 50:
                    continue
                    
                # Calculate signal
                signal = self.calculate_signal(df)
                current_price = df['Close'].iloc[-1]
                
                signals.append({
                    'symbol': symbol,
                    'price': current_price,
                    'signal': signal,
                    'action': 'BUY' if signal > 0.5 else ('SELL' if signal < -0.3 else 'HOLD')
                })
                
                logger.info(f"{symbol}: Price=${current_price:.2f}, Signal={signal:.2f}, Action={signals[-1]['action']}")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                
        return signals
    
    def execute_trades(self, signals):
        """Execute trades based on signals"""
        account = self.client.get_account()
        buying_power = float(account.buying_power)
        
        # Get current positions
        positions = self.client.get_all_positions()
        current_symbols = [p.symbol for p in positions]
        
        for signal in signals:
            try:
                if signal['action'] == 'BUY' and signal['symbol'] not in current_symbols:
                    # Calculate position size (5% of portfolio per trade)
                    position_size = min(buying_power * 0.05, 5000)  # Max $5000 per trade
                    qty = int(position_size / signal['price'])
                    
                    if qty > 0:
                        order = MarketOrderRequest(
                            symbol=signal['symbol'],
                            qty=qty,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY
                        )
                        
                        result = self.client.submit_order(order)
                        logger.info(f"✅ BUY ORDER: {qty} shares of {signal['symbol']} at ${signal['price']:.2f}")
                        
                elif signal['action'] == 'SELL' and signal['symbol'] in current_symbols:
                    # Find position
                    position = next((p for p in positions if p.symbol == signal['symbol']), None)
                    if position:
                        order = MarketOrderRequest(
                            symbol=signal['symbol'],
                            qty=int(position.qty),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY
                        )
                        
                        result = self.client.submit_order(order)
                        logger.info(f"✅ SELL ORDER: {position.qty} shares of {signal['symbol']}")
                        
            except Exception as e:
                logger.error(f"Trade execution error for {signal['symbol']}: {e}")
    
    async def run(self):
        """Main trading loop"""
        logger.info("🚀 Starting QuantNexus Auto Trading System")
        
        while True:
            try:
                # Check if market is open
                clock = self.client.get_clock()
                
                if clock.is_open:
                    logger.info("📊 Market is OPEN - Analyzing stocks...")
                    
                    # Analyze stocks
                    signals = self.analyze_stocks()
                    
                    # Filter strong signals
                    strong_signals = [s for s in signals if abs(s['signal']) > 0.5]
                    
                    if strong_signals:
                        logger.info(f"💡 Found {len(strong_signals)} trading opportunities")
                        self.execute_trades(strong_signals)
                    else:
                        logger.info("⏸️  No strong signals - holding positions")
                    
                    # Display portfolio status
                    account = self.client.get_account()
                    logger.info(f"💼 Portfolio Value: ${float(account.portfolio_value):,.2f}")
                    logger.info(f"💰 Cash: ${float(account.cash):,.2f}")
                    
                    # Wait 5 minutes before next check
                    await asyncio.sleep(300)
                    
                else:
                    logger.info("🔒 Market is CLOSED - waiting for next session")
                    await asyncio.sleep(600)  # Check every 10 minutes
                    
            except KeyboardInterrupt:
                logger.info("⏹️  Trading stopped by user")
                break
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(60)

def main():
    trader = AutoTrader()
    
    # Print startup info
    print("="*60)
    print("🚀 QUANTNEXUS AUTO TRADING SYSTEM")
    print("="*60)
    
    account = trader.client.get_account()
    print(f"Account Status: {account.status}")
    print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}")
    print(f"Watching: {', '.join(trader.watchlist)}")
    print("="*60)
    print("\nPress Ctrl+C to stop trading\n")
    
    # Run trading loop
    asyncio.run(trader.run())

if __name__ == "__main__":
    main()