#!/usr/bin/env python3
"""
QuantNexus ML Trading System Launcher
Complete ML-powered trading system for 65%+ success rate
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='QuantNexus ML Trading System')
    parser.add_argument('--mode', choices=['trade', 'train', 'backtest', 'analyze'], 
                       default='trade', help='Operating mode')
    parser.add_argument('--paper', action='store_true', default=True,
                       help='Use paper trading (default: True)')
    parser.add_argument('--config', default='.env',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    print("="*80)
    print("QUANTNEXUS ML TRADING SYSTEM")
    print("Target: $10K → $500K | Success Rate: 65%+")
    print("="*80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Paper Trading: {args.paper}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    if args.mode == 'trade':
        run_trading(args)
    elif args.mode == 'train':
        run_training(args)
    elif args.mode == 'backtest':
        run_backtest(args)
    elif args.mode == 'analyze':
        run_analysis(args)

def run_trading(args):
    """Run live trading"""
    from src.ml_trading.core.ml_trading_system import MLTradingBot
    from dotenv import load_dotenv
    
    # Load environment
    load_dotenv(args.config)
    
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("ERROR: Missing API credentials in .env file")
        sys.exit(1)
    
    # Create bot
    bot = MLTradingBot(api_key, secret_key, paper=args.paper)
    
    print("\n🤖 ML Trading Bot Initialized")
    print("📊 Features: 200+ technical indicators")
    print("🧠 Models: XGBoost + LightGBM + Neural Network + LSTM")
    print("🎯 Strategy: Multi-timeframe ensemble prediction")
    print("💰 Risk Management: Dynamic ML-optimized stops")
    print("\n" + "="*80)
    
    try:
        # Run async trading loop
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n\n⏹️ Trading stopped by user")
        print_summary(bot)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def run_training(args):
    """Train ML models"""
    print("\n🧠 Training ML Models...")
    
    from src.ml_trading.core.model_trainer import ModelTrainer
    
    trainer = ModelTrainer()
    
    # Load historical data
    print("Loading historical data...")
    data = trainer.load_training_data()
    
    # Train models
    print("Training ensemble models...")
    models = trainer.train_all_models(data)
    
    # Evaluate
    print("\nModel Performance:")
    for model_name, metrics in models.items():
        print(f"  {model_name}:")
        print(f"    - Accuracy: {metrics['accuracy']:.2%}")
        print(f"    - Sharpe Ratio: {metrics['sharpe']:.2f}")
        print(f"    - Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    # Save models
    trainer.save_models(models)
    print("\n✅ Models saved successfully")

def run_backtest(args):
    """Run backtesting"""
    print("\n📈 Running Backtest...")
    
    from src.ml_trading.backtesting.backtest_engine import BacktestEngine
    
    engine = BacktestEngine()
    
    # Configuration
    config = {
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 10000,
        'commission': 0.001,  # 0.1% commission
        'slippage': 0.001     # 0.1% slippage
    }
    
    # Run backtest
    results = engine.run_backtest(config)
    
    # Print results
    print("\n📊 Backtest Results:")
    print(f"  Total Return: {results['total_return']:.2%}")
    print(f"  Win Rate: {results['win_rate']:.2%}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Profit Factor: {results['profit_factor']:.2f}")
    
    # Generate report
    engine.generate_report(results)
    print("\n✅ Backtest report saved to reports/backtest_report.html")

def run_analysis(args):
    """Run performance analysis"""
    print("\n📊 Running Performance Analysis...")
    
    from src.ml_trading.analysis.performance_analyzer import PerformanceAnalyzer
    
    analyzer = PerformanceAnalyzer()
    
    # Load recent trades
    trades = analyzer.load_recent_trades()
    
    if not trades:
        print("No trades found for analysis")
        return
    
    # Analyze performance
    metrics = analyzer.calculate_metrics(trades)
    
    print("\n📈 Performance Metrics:")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Average Win: ${metrics['avg_win']:.2f}")
    print(f"  Average Loss: ${metrics['avg_loss']:.2f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Expectancy: ${metrics['expectancy']:.2f}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Generate charts
    analyzer.generate_charts(trades, metrics)
    print("\n✅ Analysis charts saved to reports/analysis_charts.html")

def print_summary(bot):
    """Print trading session summary"""
    print("\n" + "="*80)
    print("TRADING SESSION SUMMARY")
    print("="*80)
    
    if bot.trades:
        print(f"Total Trades: {len(bot.trades)}")
        print(f"Win Rate: {bot.performance_metrics['win_rate']:.2%}")
        print(f"Total P&L: ${bot.performance_metrics['total_pnl']:.2f}")
    else:
        print("No trades executed in this session")
    
    print("\nThank you for using QuantNexus ML Trading System!")

if __name__ == "__main__":
    main()