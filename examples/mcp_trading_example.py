#!/usr/bin/env python3
"""
Example: Using MCP Trading System
Shows how to use the MCP client for trading operations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from src.mcp_trading.mcp_client import TradingMCPClient, SimpleTradingClient

logging.basicConfig(level=logging.INFO)


async def example_analysis():
    """Example: Stock analysis with ML"""
    
    client = TradingMCPClient()
    await client.connect()
    
    try:
        print("\n🔍 STOCK ANALYSIS EXAMPLE")
        print("="*50)
        
        # Analyze multiple stocks
        symbols = ['AAPL', 'GOOGL', 'TSLA']
        
        for symbol in symbols:
            analysis = await client.analyze_stock(symbol)
            
            print(f"\n{symbol} Analysis:")
            print(f"  Current Price: ${analysis['current_price']:.2f}")
            print(f"  ML Prediction: {analysis['ml_prediction']['expected_return']:.2%}")
            print(f"  Confidence: {analysis['ml_prediction']['confidence']:.1%}")
            print(f"  Signal: {analysis['ml_prediction']['signal']}")
            print(f"  Recommendation: {analysis['recommendation']}")
            print(f"  RSI: {analysis['technical'].get('rsi', 'N/A')}")
            print(f"  Trend: {analysis['technical']['trend']}")
            print(f"  Suggested Stop Loss: ${analysis['risk_metrics']['suggested_stop_loss']:.2f}")
    
    finally:
        await client.disconnect()


async def example_trading():
    """Example: Execute trades with ML optimization"""
    
    client = TradingMCPClient()
    await client.connect()
    
    try:
        print("\n📈 TRADING EXECUTION EXAMPLE")
        print("="*50)
        
        # Get high-confidence signals
        signals = await client.get_market_signals(min_confidence=0.70)
        
        if signals:
            # Execute the top signal
            top_signal = signals[0]
            print(f"\nExecuting trade for {top_signal['symbol']}:")
            print(f"  Signal: {top_signal['signal']}")
            print(f"  Confidence: {top_signal['confidence']:.1%}")
            
            if top_signal['signal'] == 'buy':
                trade = await client.execute_trade(
                    symbol=top_signal['symbol'],
                    action='buy',
                    quantity=0  # Auto-calculate position size
                )
                
                print(f"\n✅ Trade Executed:")
                print(f"  Symbol: {trade['symbol']}")
                print(f"  Action: {trade['action']}")
                print(f"  Quantity: {trade['quantity']}")
                print(f"  Entry Price: ${trade['entry_price']:.2f}")
                print(f"  Stop Loss: ${trade['stop_loss']:.2f}")
                print(f"  Stop Loss %: {trade['stop_loss_pct']:.1%}")
                print(f"  Use Trailing: {trade['use_trailing']}")
        else:
            print("No high-confidence signals found")
    
    finally:
        await client.disconnect()


async def example_portfolio_management():
    """Example: Portfolio management and risk metrics"""
    
    client = TradingMCPClient()
    await client.connect()
    
    try:
        print("\n💼 PORTFOLIO MANAGEMENT EXAMPLE")
        print("="*50)
        
        # Get portfolio status
        portfolio = await client.get_portfolio_status()
        
        print(f"\nPortfolio Summary:")
        print(f"  Total Value: ${portfolio['total_value']:.2f}")
        print(f"  Cash Balance: ${portfolio['cash']:.2f}")
        print(f"  Daily P&L: ${portfolio['daily_pnl']:.2f}")
        print(f"  Position Count: {len(portfolio['positions'])}")
        
        # Display positions
        if portfolio['positions']:
            print("\nOpen Positions:")
            for symbol, position in portfolio['positions'].items():
                pnl = position.get('pnl', 0)
                pnl_pct = position.get('pnl_pct', 0)
                print(f"  {symbol}:")
                print(f"    Quantity: {position['quantity']}")
                print(f"    Entry: ${position['entry_price']:.2f}")
                print(f"    Current: ${position.get('current_price', 0):.2f}")
                print(f"    P&L: ${pnl:.2f} ({pnl_pct:.1%})")
        
        # Calculate risk metrics
        risk_metrics = await client.calculate_risk_metrics()
        
        print(f"\nRisk Metrics:")
        print(f"  Win Rate: {risk_metrics.get('win_rate', 0):.1%}")
        print(f"  Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {risk_metrics.get('max_drawdown', 0):.1%}")
        print(f"  Profit Factor: {risk_metrics.get('profit_factor', 0):.2f}")
    
    finally:
        await client.disconnect()


async def example_ml_models():
    """Example: Train and evaluate ML models"""
    
    client = TradingMCPClient()
    await client.connect()
    
    try:
        print("\n🤖 ML MODEL MANAGEMENT EXAMPLE")
        print("="*50)
        
        # Train models
        print("\nTraining ML models...")
        print("This may take several minutes...")
        
        training_result = await client.train_models(
            symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        )
        
        print(f"\n✅ Training Complete:")
        print(f"  Models Trained: {', '.join(training_result['models_trained'])}")
        print(f"  Feature Count: {training_result['feature_count']}")
        
        # Get model performance
        performance = await client.get_model_performance()
        
        print(f"\nModel Performance:")
        for model_name, metrics in performance.items():
            if isinstance(metrics, dict):
                print(f"  {model_name}:")
                print(f"    MSE: {metrics.get('mse', 0):.6f}")
                print(f"    MAE: {metrics.get('mae', 0):.6f}")
    
    finally:
        await client.disconnect()


async def example_backtesting():
    """Example: Run backtests"""
    
    client = TradingMCPClient()
    await client.connect()
    
    try:
        print("\n📊 BACKTESTING EXAMPLE")
        print("="*50)
        
        # Run backtest
        print("\nRunning backtest...")
        print("Period: 2023-01-01 to 2023-12-31")
        print("Initial Capital: $10,000")
        
        results = await client.run_backtest(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=10000
        )
        
        print(f"\n📈 Backtest Results:")
        print(f"  Total Trades: {results.get('total_trades', 0)}")
        print(f"  Win Rate: {results.get('win_rate', 0):.1%}")
        print(f"  Total Return: {results.get('total_return', 0):.1%}")
        print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {results.get('max_drawdown', 0):.1%}")
        print(f"  Profit Factor: {results.get('profit_factor', 0):.2f}")
        print(f"  Final Capital: ${results.get('final_capital', 0):.2f}")
    
    finally:
        await client.disconnect()


async def example_daily_workflow():
    """Example: Complete daily trading workflow"""
    
    client = TradingMCPClient()
    await client.connect()
    
    try:
        print("\n🚀 DAILY TRADING WORKFLOW EXAMPLE")
        print("="*50)
        
        # Execute complete daily trading
        summary = await client.execute_daily_trading()
        
        print(f"\n📅 Daily Trading Summary - {summary['date']}")
        print(f"\nSignals Generated: {len(summary['signals_generated'])}")
        
        for signal in summary['signals_generated'][:3]:
            print(f"  {signal['symbol']}: {signal['signal']} "
                  f"({signal['confidence']:.1%} confidence)")
        
        print(f"\nTrades Executed: {len(summary['trades_executed'])}")
        for trade in summary['trades_executed']:
            print(f"  {trade['action'].upper()} {trade['quantity']} "
                  f"{trade['symbol']} @ ${trade['entry_price']:.2f}")
        
        print(f"\nPositions Closed: {len(summary['positions_closed'])}")
        for position in summary['positions_closed']:
            print(f"  {position['symbol']}: "
                  f"P&L ${position['pnl']:.2f} ({position['pnl_pct']:.1%}) "
                  f"- {position['reason']}")
        
        print(f"\nPortfolio Status:")
        print(f"  Total Value: ${summary['portfolio_status']['total_value']:.2f}")
        print(f"  Daily P&L: ${summary['daily_pnl']:.2f}")
    
    finally:
        await client.disconnect()


def example_simple_client():
    """Example: Using the simple synchronous client"""
    
    print("\n🎯 SIMPLE CLIENT EXAMPLE")
    print("="*50)
    
    # Use context manager for automatic connection/disconnection
    with SimpleTradingClient() as client:
        
        # Analyze a stock
        analysis = client.analyze("AAPL")
        print(f"\nAAPL Analysis:")
        print(f"  Price: ${analysis['current_price']:.2f}")
        print(f"  Signal: {analysis['ml_prediction']['signal']}")
        print(f"  Confidence: {analysis['ml_prediction']['confidence']:.1%}")
        
        # Get signals
        signals = client.signals()
        print(f"\nTop Signals:")
        for signal in signals[:3]:
            print(f"  {signal['symbol']}: {signal['signal']} "
                  f"({signal['confidence']:.1%})")
        
        # Check portfolio
        portfolio = client.portfolio()
        print(f"\nPortfolio:")
        print(f"  Value: ${portfolio['total_value']:.2f}")
        print(f"  Cash: ${portfolio['cash']:.2f}")
        
        # Run backtest
        backtest = client.backtest("2023-06-01", "2023-12-31")
        print(f"\nBacktest Results:")
        print(f"  Win Rate: {backtest.get('win_rate', 0):.1%}")
        print(f"  Return: {backtest.get('total_return', 0):.1%}")


async def main():
    """Run all examples"""
    
    print("\n" + "="*70)
    print("   QUANTNEXUS MCP TRADING SYSTEM - EXAMPLES")
    print("   ML-Powered Trading with 65%+ Success Rate")
    print("="*70)
    
    # Choose which examples to run
    examples = [
        ("Stock Analysis", example_analysis),
        ("Trading Execution", example_trading),
        ("Portfolio Management", example_portfolio_management),
        ("ML Models", example_ml_models),
        ("Backtesting", example_backtesting),
        ("Daily Workflow", example_daily_workflow),
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  {len(examples)+1}. Simple Client (Synchronous)")
    print(f"  {len(examples)+2}. Run All Examples")
    print("  0. Exit")
    
    choice = input("\nSelect example to run (0-8): ")
    
    if choice == "0":
        print("Exiting...")
        return
    elif choice == str(len(examples)+1):
        example_simple_client()
    elif choice == str(len(examples)+2):
        # Run all async examples
        for name, func in examples:
            print(f"\n{'='*20} Running: {name} {'='*20}")
            await func()
            await asyncio.sleep(1)
        # Run simple client example
        print(f"\n{'='*20} Running: Simple Client {'='*20}")
        example_simple_client()
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                name, func = examples[idx]
                print(f"\nRunning: {name}")
                await func()
            else:
                print("Invalid choice")
        except (ValueError, IndexError):
            print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())