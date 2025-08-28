#!/usr/bin/env python3
"""
QuantNexus MCP Trading Server
Core MCP server for ML-powered stock market analysis and trading
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# MCP imports
from mcp.server import ServerSession
from mcp.types import Tool, Resource
import mcp

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml_trading.core.ml_trading_system import (
    MLTradingBot, MLFeatureEngine, MLEnsembleModel, DynamicStopLossOptimizer
)
from src.ml_trading.core.model_trainer import ModelTrainer
from src.ml_trading.backtesting.backtest_engine import BacktestEngine
from src.ml_trading.analysis.performance_analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)


class TradingMCPServer:
    """
    MCP Server for ML Trading System
    Provides tools, resources, and prompts for stock market analysis and trading
    """
    
    def __init__(self):
        self.server = Server("quantnexus-trading-mcp")
        self.trading_bot = None
        self.feature_engine = MLFeatureEngine()
        self.model_trainer = ModelTrainer()
        self.backtest_engine = BacktestEngine()
        self.performance_analyzer = PerformanceAnalyzer()
        self.ensemble_model = MLEnsembleModel()
        self.stop_optimizer = DynamicStopLossOptimizer()
        
        # Cache for market data
        self.market_data_cache = {}
        self.signals_cache = []
        self.portfolio_state = {
            'cash': 10000,
            'positions': {},
            'total_value': 10000,
            'daily_pnl': 0
        }
        
        # Register all tools, resources, and prompts
        self._register_tools()
        self._register_resources()
        self._register_prompts()
    
    def _register_tools(self):
        """Register all MCP tools for trading"""
        
        # Market Analysis Tools
        @self.server.tool()
        async def analyze_stock(symbol: str, period: str = "1d") -> ToolResult:
            """
            Analyze a stock with ML models and technical indicators
            
            Args:
                symbol: Stock ticker symbol
                period: Time period (1d, 5d, 1mo, 3mo, 1y)
            
            Returns:
                Comprehensive analysis with ML predictions
            """
            try:
                # Fetch market data
                data = await self._fetch_market_data(symbol, period)
                
                # Generate features
                features = self.feature_engine.generate_features(data, symbol)
                
                # Get ML predictions if model is trained
                prediction = None
                confidence = None
                if self.ensemble_model.is_trained:
                    pred_result = self.ensemble_model.predict(features.iloc[-1:].values)
                    prediction = pred_result['prediction']
                    confidence = pred_result['confidence']
                
                # Calculate technical indicators
                current_price = data['close'].iloc[-1]
                sma_20 = data['close'].rolling(20).mean().iloc[-1]
                sma_50 = data['close'].rolling(50).mean().iloc[-1]
                rsi = features['rsi_14'].iloc[-1] if 'rsi_14' in features else None
                
                # Generate signal
                signal = self._generate_signal(prediction, confidence, current_price)
                
                analysis = {
                    'symbol': symbol,
                    'current_price': current_price,
                    'ml_prediction': {
                        'expected_return': prediction,
                        'confidence': confidence,
                        'signal': signal
                    },
                    'technical': {
                        'sma_20': sma_20,
                        'sma_50': sma_50,
                        'rsi': rsi,
                        'trend': 'bullish' if current_price > sma_20 > sma_50 else 'bearish'
                    },
                    'recommendation': self._get_recommendation(signal, confidence),
                    'risk_metrics': {
                        'volatility': features['volatility_20'].iloc[-1] if 'volatility_20' in features else None,
                        'suggested_stop_loss': self._calculate_stop_loss(current_price, features)
                    }
                }
                
                return ToolResult(
                    content=[TextContent(text=json.dumps(analysis, indent=2))],
                    is_error=False
                )
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                return ToolResult(
                    content=[TextContent(text=f"Error: {str(e)}")],
                    is_error=True
                )
        
        @self.server.tool()
        async def get_market_signals(min_confidence: float = 0.65) -> ToolResult:
            """
            Get trading signals for multiple stocks
            
            Args:
                min_confidence: Minimum ML confidence threshold
            
            Returns:
                List of high-confidence trading signals
            """
            try:
                universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']
                signals = []
                
                for symbol in universe:
                    analysis = await analyze_stock(symbol, "1d")
                    result = json.loads(analysis.content[0].text)
                    
                    if result['ml_prediction']['confidence'] >= min_confidence:
                        signals.append({
                            'symbol': symbol,
                            'signal': result['ml_prediction']['signal'],
                            'confidence': result['ml_prediction']['confidence'],
                            'expected_return': result['ml_prediction']['expected_return'],
                            'current_price': result['current_price']
                        })
                
                # Sort by confidence
                signals.sort(key=lambda x: x['confidence'], reverse=True)
                
                return ToolResult(
                    content=[TextContent(text=json.dumps(signals, indent=2))],
                    is_error=False
                )
                
            except Exception as e:
                return ToolResult(
                    content=[TextContent(text=f"Error: {str(e)}")],
                    is_error=True
                )
        
        # Trading Execution Tools
        @self.server.tool()
        async def execute_trade(
            symbol: str,
            action: str,
            quantity: int,
            order_type: str = "market",
            limit_price: Optional[float] = None
        ) -> ToolResult:
            """
            Execute a trade with ML-optimized parameters
            
            Args:
                symbol: Stock ticker
                action: 'buy' or 'sell'
                quantity: Number of shares
                order_type: 'market' or 'limit'
                limit_price: Price for limit orders
            
            Returns:
                Trade execution details
            """
            try:
                # Validate action
                if action not in ['buy', 'sell']:
                    raise ValueError("Action must be 'buy' or 'sell'")
                
                # Get current market data
                data = await self._fetch_market_data(symbol, "1d")
                current_price = data['close'].iloc[-1]
                
                # Calculate position size if not specified
                if quantity == 0:
                    quantity = self._calculate_position_size(current_price)
                
                # Get ML-optimized stop loss
                features = self.feature_engine.generate_features(data, symbol)
                stop_loss_info = self.stop_optimizer.calculate_optimal_stop(
                    features, current_price
                )
                
                # Simulate trade execution
                trade = {
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'order_type': order_type,
                    'entry_price': limit_price if limit_price else current_price,
                    'stop_loss': stop_loss_info['stop_price'],
                    'stop_loss_pct': stop_loss_info['stop_percent'],
                    'use_trailing': stop_loss_info['use_trailing'],
                    'timestamp': datetime.now().isoformat(),
                    'status': 'executed'
                }
                
                # Update portfolio
                self._update_portfolio(trade)
                
                return ToolResult(
                    content=[TextContent(text=json.dumps(trade, indent=2))],
                    is_error=False
                )
                
            except Exception as e:
                return ToolResult(
                    content=[TextContent(text=f"Error: {str(e)}")],
                    is_error=True
                )
        
        @self.server.tool()
        async def close_position(symbol: str, reason: str = "manual") -> ToolResult:
            """
            Close an existing position
            
            Args:
                symbol: Stock ticker
                reason: Reason for closing
            
            Returns:
                Position closure details with P&L
            """
            try:
                if symbol not in self.portfolio_state['positions']:
                    return ToolResult(
                        content=[TextContent(text=f"No position found for {symbol}")],
                        is_error=True
                    )
                
                position = self.portfolio_state['positions'][symbol]
                data = await self._fetch_market_data(symbol, "1d")
                current_price = data['close'].iloc[-1]
                
                # Calculate P&L
                pnl = (current_price - position['entry_price']) * position['quantity']
                pnl_pct = pnl / (position['entry_price'] * position['quantity'])
                
                result = {
                    'symbol': symbol,
                    'quantity': position['quantity'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'reason': reason,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Update portfolio
                del self.portfolio_state['positions'][symbol]
                self.portfolio_state['cash'] += current_price * position['quantity']
                self.portfolio_state['daily_pnl'] += pnl
                
                return ToolResult(
                    content=[TextContent(text=json.dumps(result, indent=2))],
                    is_error=False
                )
                
            except Exception as e:
                return ToolResult(
                    content=[TextContent(text=f"Error: {str(e)}")],
                    is_error=True
                )
        
        # ML Model Management Tools
        @self.server.tool()
        async def train_models(symbols: List[str] = None) -> ToolResult:
            """
            Train ML models with historical data
            
            Args:
                symbols: List of symbols to train on
            
            Returns:
                Training results and model performance metrics
            """
            try:
                # Load training data
                data = self.model_trainer.load_training_data(symbols)
                
                # Train all models
                results = self.model_trainer.train_all_models(data)
                
                # Update ensemble model
                self.ensemble_model = results['models']
                
                return ToolResult(
                    content=[TextContent(text=json.dumps({
                        'status': 'success',
                        'models_trained': list(results['models'].keys()),
                        'metrics': results['metrics'],
                        'feature_count': len(results['feature_columns'])
                    }, indent=2))],
                    is_error=False
                )
                
            except Exception as e:
                return ToolResult(
                    content=[TextContent(text=f"Error: {str(e)}")],
                    is_error=True
                )
        
        @self.server.tool()
        async def get_model_performance() -> ToolResult:
            """
            Get current ML model performance metrics
            
            Returns:
                Model performance statistics
            """
            try:
                if not self.ensemble_model.is_trained:
                    return ToolResult(
                        content=[TextContent(text="Models not trained yet")],
                        is_error=True
                    )
                
                # Load saved metrics
                metrics_file = self.model_trainer.models_dir / 'training_metrics.json'
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                else:
                    metrics = {'message': 'No metrics available'}
                
                return ToolResult(
                    content=[TextContent(text=json.dumps(metrics, indent=2))],
                    is_error=False
                )
                
            except Exception as e:
                return ToolResult(
                    content=[TextContent(text=f"Error: {str(e)}")],
                    is_error=True
                )
        
        # Portfolio Management Tools
        @self.server.tool()
        async def get_portfolio_status() -> ToolResult:
            """
            Get current portfolio status and positions
            
            Returns:
                Portfolio summary with all positions
            """
            try:
                # Update portfolio values
                total_value = self.portfolio_state['cash']
                
                for symbol, position in self.portfolio_state['positions'].items():
                    data = await self._fetch_market_data(symbol, "1d")
                    current_price = data['close'].iloc[-1]
                    position_value = current_price * position['quantity']
                    total_value += position_value
                    
                    # Update position with current values
                    position['current_price'] = current_price
                    position['current_value'] = position_value
                    position['pnl'] = (current_price - position['entry_price']) * position['quantity']
                    position['pnl_pct'] = position['pnl'] / (position['entry_price'] * position['quantity'])
                
                self.portfolio_state['total_value'] = total_value
                
                return ToolResult(
                    content=[TextContent(text=json.dumps(self.portfolio_state, indent=2))],
                    is_error=False
                )
                
            except Exception as e:
                return ToolResult(
                    content=[TextContent(text=f"Error: {str(e)}")],
                    is_error=True
                )
        
        @self.server.tool()
        async def calculate_risk_metrics() -> ToolResult:
            """
            Calculate portfolio risk metrics
            
            Returns:
                VaR, Sharpe ratio, max drawdown, etc.
            """
            try:
                # Get recent trades from analyzer
                trades = self.performance_analyzer.load_recent_trades()
                
                if not trades:
                    return ToolResult(
                        content=[TextContent(text="No trades to analyze")],
                        is_error=False
                    )
                
                # Calculate metrics
                metrics = self.performance_analyzer.calculate_metrics(trades)
                
                # Add portfolio-specific metrics
                metrics['portfolio_value'] = self.portfolio_state['total_value']
                metrics['cash_balance'] = self.portfolio_state['cash']
                metrics['position_count'] = len(self.portfolio_state['positions'])
                metrics['daily_pnl'] = self.portfolio_state['daily_pnl']
                
                return ToolResult(
                    content=[TextContent(text=json.dumps(metrics, indent=2))],
                    is_error=False
                )
                
            except Exception as e:
                return ToolResult(
                    content=[TextContent(text=f"Error: {str(e)}")],
                    is_error=True
                )
        
        # Backtesting Tools
        @self.server.tool()
        async def run_backtest(
            start_date: str,
            end_date: str,
            initial_capital: float = 10000
        ) -> ToolResult:
            """
            Run backtest on historical data
            
            Args:
                start_date: Start date (YYYY-MM-DD)
                end_date: End date (YYYY-MM-DD)
                initial_capital: Starting capital
            
            Returns:
                Backtest results with performance metrics
            """
            try:
                config = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'initial_capital': initial_capital,
                    'commission': 0.001,
                    'slippage': 0.001
                }
                
                results = self.backtest_engine.run_backtest(config)
                
                return ToolResult(
                    content=[TextContent(text=json.dumps(results, indent=2, default=str))],
                    is_error=False
                )
                
            except Exception as e:
                return ToolResult(
                    content=[TextContent(text=f"Error: {str(e)}")],
                    is_error=True
                )
    
    def _register_resources(self):
        """Register MCP resources for data access"""
        
        @self.server.resource("market_data/{symbol}")
        async def market_data_resource(symbol: str) -> ResourceContent:
            """
            Access real-time market data for a symbol
            """
            try:
                data = await self._fetch_market_data(symbol, "1d")
                return ResourceContent(
                    uri=f"market_data/{symbol}",
                    mimeType="application/json",
                    text=data.to_json()
                )
            except Exception as e:
                return ResourceContent(
                    uri=f"market_data/{symbol}",
                    mimeType="text/plain",
                    text=f"Error: {str(e)}"
                )
        
        @self.server.resource("portfolio/state")
        async def portfolio_state_resource() -> ResourceContent:
            """
            Access current portfolio state
            """
            return ResourceContent(
                uri="portfolio/state",
                mimeType="application/json",
                text=json.dumps(self.portfolio_state, indent=2)
            )
        
        @self.server.resource("signals/latest")
        async def latest_signals_resource() -> ResourceContent:
            """
            Access latest trading signals
            """
            return ResourceContent(
                uri="signals/latest",
                mimeType="application/json",
                text=json.dumps(self.signals_cache, indent=2)
            )
        
        @self.server.resource("models/metrics")
        async def model_metrics_resource() -> ResourceContent:
            """
            Access ML model performance metrics
            """
            try:
                metrics_file = self.model_trainer.models_dir / 'training_metrics.json'
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                else:
                    metrics = {'message': 'No metrics available'}
                
                return ResourceContent(
                    uri="models/metrics",
                    mimeType="application/json",
                    text=json.dumps(metrics, indent=2)
                )
            except Exception as e:
                return ResourceContent(
                    uri="models/metrics",
                    mimeType="text/plain",
                    text=f"Error: {str(e)}"
                )
    
    def _register_prompts(self):
        """Register MCP prompt templates"""
        
        @self.server.prompt("analyze_stock")
        async def analyze_stock_prompt(symbol: str) -> Prompt:
            """
            Prompt template for stock analysis
            """
            return Prompt(
                name="analyze_stock",
                description=f"Analyze {symbol} with ML models",
                arguments={
                    "symbol": symbol,
                    "include_technical": True,
                    "include_ml_prediction": True,
                    "include_risk_metrics": True
                },
                template=f"""
                Analyze {symbol} using:
                1. ML ensemble predictions
                2. Technical indicators
                3. Risk metrics
                4. Trading recommendations
                
                Provide actionable insights based on 65%+ confidence threshold.
                """
            )
        
        @self.server.prompt("execute_trade")
        async def execute_trade_prompt(symbol: str, action: str) -> Prompt:
            """
            Prompt template for trade execution
            """
            return Prompt(
                name="execute_trade",
                description=f"Execute {action} trade for {symbol}",
                arguments={
                    "symbol": symbol,
                    "action": action,
                    "use_ml_sizing": True,
                    "use_dynamic_stops": True
                },
                template=f"""
                Execute {action} order for {symbol}:
                1. Calculate optimal position size using Kelly Criterion
                2. Set ML-optimized stop loss
                3. Monitor for entry signals
                4. Execute with proper risk management
                """
            )
        
        @self.server.prompt("daily_trading")
        async def daily_trading_prompt() -> Prompt:
            """
            Prompt template for daily trading workflow
            """
            return Prompt(
                name="daily_trading",
                description="Daily trading workflow",
                arguments={
                    "scan_universe": True,
                    "generate_signals": True,
                    "manage_positions": True,
                    "report_performance": True
                },
                template="""
                Execute daily trading workflow:
                1. Scan stock universe for opportunities
                2. Generate ML signals (confidence > 65%)
                3. Execute highest confidence trades
                4. Manage existing positions
                5. Generate performance report
                
                Target: 65%+ win rate, 3% profit targets, 1.5% stop losses
                """
            )
    
    # Helper methods
    async def _fetch_market_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch market data with caching"""
        cache_key = f"{symbol}_{period}"
        
        # Check cache
        if cache_key in self.market_data_cache:
            cached_data, timestamp = self.market_data_cache[cache_key]
            if datetime.now() - timestamp < timedelta(minutes=5):
                return cached_data
        
        # Fetch new data
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        data.columns = data.columns.str.lower()
        
        # Update cache
        self.market_data_cache[cache_key] = (data, datetime.now())
        
        return data
    
    def _generate_signal(self, prediction: float, confidence: float, price: float) -> str:
        """Generate trading signal based on ML prediction"""
        if confidence < 0.65:
            return 'hold'
        
        if prediction > 0.002:  # 0.2% expected return
            return 'buy'
        elif prediction < -0.002:
            return 'sell'
        else:
            return 'hold'
    
    def _get_recommendation(self, signal: str, confidence: float) -> str:
        """Get trading recommendation"""
        if signal == 'buy' and confidence >= 0.75:
            return 'STRONG BUY'
        elif signal == 'buy' and confidence >= 0.65:
            return 'BUY'
        elif signal == 'sell' and confidence >= 0.75:
            return 'STRONG SELL'
        elif signal == 'sell' and confidence >= 0.65:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_stop_loss(self, price: float, features: pd.DataFrame) -> float:
        """Calculate stop loss price"""
        stop_info = self.stop_optimizer.calculate_optimal_stop(features, price)
        return stop_info['stop_price']
    
    def _calculate_position_size(self, price: float) -> int:
        """Calculate position size using Kelly Criterion"""
        portfolio_value = self.portfolio_state['total_value']
        position_value = portfolio_value * 0.05  # 5% default
        return max(1, int(position_value / price))
    
    def _update_portfolio(self, trade: Dict):
        """Update portfolio with new trade"""
        symbol = trade['symbol']
        
        if trade['action'] == 'buy':
            if symbol in self.portfolio_state['positions']:
                # Add to existing position
                position = self.portfolio_state['positions'][symbol]
                total_quantity = position['quantity'] + trade['quantity']
                avg_price = ((position['entry_price'] * position['quantity']) + 
                           (trade['entry_price'] * trade['quantity'])) / total_quantity
                
                position['quantity'] = total_quantity
                position['entry_price'] = avg_price
            else:
                # New position
                self.portfolio_state['positions'][symbol] = {
                    'quantity': trade['quantity'],
                    'entry_price': trade['entry_price'],
                    'stop_loss': trade['stop_loss'],
                    'entry_time': trade['timestamp']
                }
            
            # Update cash
            self.portfolio_state['cash'] -= trade['entry_price'] * trade['quantity']
        
        elif trade['action'] == 'sell':
            if symbol in self.portfolio_state['positions']:
                position = self.portfolio_state['positions'][symbol]
                
                if trade['quantity'] >= position['quantity']:
                    # Close entire position
                    del self.portfolio_state['positions'][symbol]
                else:
                    # Partial sell
                    position['quantity'] -= trade['quantity']
                
                # Update cash
                self.portfolio_state['cash'] += trade['entry_price'] * trade['quantity']
    
    async def run(self):
        """Run the MCP server"""
        logger.info("Starting QuantNexus MCP Trading Server...")
        
        # Initialize trading bot if credentials are available
        try:
            from dotenv import load_dotenv
            import os
            
            load_dotenv('config/.env')
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if api_key and secret_key:
                self.trading_bot = MLTradingBot(api_key, secret_key, paper=True)
                logger.info("Trading bot initialized with Alpaca API")
        except Exception as e:
            logger.warning(f"Could not initialize trading bot: {e}")
        
        # Start server
        await self.server.run()


def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    server = TradingMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()