#!/usr/bin/env python3
"""
Backtesting Engine for ML Trading System
Comprehensive backtesting with realistic market conditions
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
from dataclasses import dataclass, asdict

# Visualization
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot

# Performance metrics
import quantstats as qs

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Trade record"""
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    side: str  # 'long' or 'short'
    pnl: Optional[float]
    pnl_pct: Optional[float]
    commission: float
    slippage: float
    status: str  # 'open', 'closed'
    exit_reason: Optional[str]  # 'profit_target', 'stop_loss', 'signal', 'eod'


class BacktestEngine:
    """
    Comprehensive backtesting engine with realistic market conditions
    """
    
    def __init__(self):
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.positions = {}
        self.metrics = {}
        
    def run_backtest(self, config: Dict) -> Dict:
        """
        Run complete backtest with ML strategy
        """
        
        logger.info("Starting backtest...")
        logger.info(f"Period: {config['start_date']} to {config['end_date']}")
        logger.info(f"Initial Capital: ${config['initial_capital']:,}")
        
        # Initialize backtest state
        self.initial_capital = config['initial_capital']
        self.current_capital = config['initial_capital']
        self.commission = config.get('commission', 0.001)
        self.slippage = config.get('slippage', 0.001)
        
        # Load ML models
        from ..core.model_trainer import ModelTrainer
        trainer = ModelTrainer()
        models, scaler = trainer.load_models()
        
        if not models:
            logger.warning("No trained models found. Training new models...")
            data = trainer.load_training_data()
            results = trainer.train_all_models(data)
            models = results['models']
        
        # Load historical data for backtesting
        data = self._load_backtest_data(config['start_date'], config['end_date'])
        
        # Generate features
        from ..core.ml_trading_system import MLFeatureEngine
        feature_engine = MLFeatureEngine()
        
        # Run backtest day by day
        for i in range(100, len(data)):  # Start after warmup period
            
            # Get data up to current point
            current_data = data.iloc[:i+1]
            
            # Generate features
            features = feature_engine.generate_features(current_data)
            
            if features.empty or len(features) < 1:
                continue
            
            # Generate signals from ML models
            signals = self._generate_ml_signals(
                models, features.iloc[-1:], 
                current_data.iloc[-1]
            )
            
            # Execute trades based on signals
            for signal in signals:
                self._execute_backtest_trade(signal, i)
            
            # Update open positions
            self._update_positions(current_data.iloc[-1])
            
            # Record equity
            self.equity_curve.append({
                'date': current_data.index[-1],
                'equity': self.current_capital + self._get_open_pnl(current_data.iloc[-1])
            })
        
        # Close all open positions at end
        self._close_all_positions(data.iloc[-1])
        
        # Calculate metrics
        self.metrics = self._calculate_metrics()
        
        # Generate report
        self._generate_backtest_report()
        
        return self.metrics
    
    def _load_backtest_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical data for backtesting"""
        
        import yfinance as yf
        
        # For demonstration, using a few symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        
        all_data = {}
        
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            all_data[symbol] = df
        
        # For simplicity, returning data for first symbol
        # In production, would handle multiple symbols
        return all_data['AAPL']
    
    def _generate_ml_signals(self, models: Dict, features: pd.DataFrame, 
                           current_bar: pd.Series) -> List[Dict]:
        """Generate trading signals from ML models"""
        
        signals = []
        
        # Get predictions from ensemble
        predictions = []
        for name, model in models.items():
            try:
                if name in ['neural_network', 'lstm']:
                    pred = model.predict(features.values, verbose=0)[0][0]
                else:
                    pred = model.predict(features.values)[0]
                predictions.append(pred)
            except:
                continue
        
        if not predictions:
            return signals
        
        # Calculate ensemble prediction
        ensemble_pred = np.mean(predictions)
        confidence = 1 / (1 + np.std(predictions))
        
        # Generate signal if confidence is high enough
        if confidence > 0.65:
            if ensemble_pred > 0.002:  # 0.2% expected return threshold
                signals.append({
                    'symbol': 'AAPL',  # Placeholder
                    'action': 'buy',
                    'price': current_bar['Close'],
                    'confidence': confidence,
                    'expected_return': ensemble_pred,
                    'quantity': self._calculate_position_size(confidence)
                })
            elif ensemble_pred < -0.002:
                signals.append({
                    'symbol': 'AAPL',
                    'action': 'sell',
                    'price': current_bar['Close'],
                    'confidence': confidence,
                    'expected_return': abs(ensemble_pred),
                    'quantity': self._calculate_position_size(confidence)
                })
        
        return signals
    
    def _calculate_position_size(self, confidence: float) -> int:
        """Calculate position size based on Kelly Criterion"""
        
        # Simplified Kelly sizing
        kelly_fraction = confidence * 0.25  # Conservative Kelly (25% of full Kelly)
        position_value = self.current_capital * min(kelly_fraction, 0.1)  # Max 10% per position
        
        # Assume $100 per share for simplicity
        shares = max(1, int(position_value / 100))
        
        return shares
    
    def _execute_backtest_trade(self, signal: Dict, bar_index: int):
        """Execute trade in backtest"""
        
        symbol = signal['symbol']
        
        # Check if already in position
        if symbol in self.positions:
            # Check if should reverse position
            current_pos = self.positions[symbol]
            if (current_pos.side == 'long' and signal['action'] == 'sell') or \
               (current_pos.side == 'short' and signal['action'] == 'buy'):
                # Close current position
                self._close_position(symbol, signal['price'], 'signal_reverse')
                # Open new position
                self._open_position(signal)
        else:
            # Open new position
            self._open_position(signal)
    
    def _open_position(self, signal: Dict):
        """Open new position"""
        
        # Calculate costs
        entry_price = signal['price'] * (1 + self.slippage)
        commission = signal['quantity'] * entry_price * self.commission
        
        # Check if enough capital
        required_capital = signal['quantity'] * entry_price + commission
        if required_capital > self.current_capital:
            return
        
        # Create trade
        trade = Trade(
            symbol=signal['symbol'],
            entry_time=datetime.now(),
            exit_time=None,
            entry_price=entry_price,
            exit_price=None,
            quantity=signal['quantity'],
            side='long' if signal['action'] == 'buy' else 'short',
            pnl=None,
            pnl_pct=None,
            commission=commission,
            slippage=self.slippage,
            status='open',
            exit_reason=None
        )
        
        # Update capital
        self.current_capital -= required_capital
        
        # Store position
        self.positions[signal['symbol']] = trade
        self.trades.append(trade)
        
        logger.debug(f"Opened {trade.side} position: {trade.symbol} @ ${trade.entry_price:.2f}")
    
    def _close_position(self, symbol: str, exit_price: float, reason: str):
        """Close existing position"""
        
        if symbol not in self.positions:
            return
        
        trade = self.positions[symbol]
        
        # Calculate exit with slippage
        if trade.side == 'long':
            exit_price *= (1 - self.slippage)
        else:
            exit_price *= (1 + self.slippage)
        
        # Calculate P&L
        if trade.side == 'long':
            pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity
        
        # Commission on exit
        exit_commission = trade.quantity * exit_price * self.commission
        pnl -= exit_commission
        
        # Update trade
        trade.exit_time = datetime.now()
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_pct = pnl / (trade.entry_price * trade.quantity)
        trade.status = 'closed'
        trade.exit_reason = reason
        
        # Update capital
        self.current_capital += trade.quantity * exit_price - exit_commission
        
        # Remove from open positions
        del self.positions[symbol]
        
        logger.debug(f"Closed {trade.side} position: {symbol} @ ${exit_price:.2f}, P&L: ${pnl:.2f}")
    
    def _update_positions(self, current_bar: pd.Series):
        """Update open positions with stops and targets"""
        
        for symbol, trade in list(self.positions.items()):
            current_price = current_bar['Close']
            
            # Calculate current P&L
            if trade.side == 'long':
                pnl_pct = (current_price - trade.entry_price) / trade.entry_price
            else:
                pnl_pct = (trade.entry_price - current_price) / trade.entry_price
            
            # Check stop loss (1.5%)
            if pnl_pct <= -0.015:
                self._close_position(symbol, current_price, 'stop_loss')
            
            # Check profit target (3%)
            elif pnl_pct >= 0.03:
                self._close_position(symbol, current_price, 'profit_target')
    
    def _get_open_pnl(self, current_bar: pd.Series) -> float:
        """Calculate unrealized P&L for open positions"""
        
        total_pnl = 0
        current_price = current_bar['Close']
        
        for symbol, trade in self.positions.items():
            if trade.side == 'long':
                pnl = (current_price - trade.entry_price) * trade.quantity
            else:
                pnl = (trade.entry_price - current_price) * trade.quantity
            total_pnl += pnl
        
        return total_pnl
    
    def _close_all_positions(self, final_bar: pd.Series):
        """Close all open positions at end of backtest"""
        
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, final_bar['Close'], 'end_of_backtest')
    
    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive backtest metrics"""
        
        if not self.trades:
            return {'error': 'No trades executed'}
        
        closed_trades = [t for t in self.trades if t.status == 'closed']
        
        if not closed_trades:
            return {'error': 'No closed trades'}
        
        # Win/Loss statistics
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        
        # P&L statistics
        total_pnl = sum(t.pnl for t in closed_trades)
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Returns
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio (simplified)
        if self.equity_curve:
            returns = pd.Series([e['equity'] for e in self.equity_curve]).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        if self.equity_curve:
            equity = pd.Series([e['equity'] for e in self.equity_curve])
            cummax = equity.cummax()
            drawdown = (equity - cummax) / cummax
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': self.current_capital
        }
    
    def generate_report(self, results: Dict):
        """Generate HTML backtest report"""
        
        # Create plots
        fig = sp.make_subplots(
            rows=3, cols=2,
            subplot_titles=('Equity Curve', 'Drawdown', 
                          'Monthly Returns', 'Trade Distribution',
                          'Win/Loss Distribution', 'Trade Duration'),
            vertical_spacing=0.1,
            horizontal_spacing=0.15
        )
        
        if self.equity_curve:
            # Equity curve
            dates = [e['date'] for e in self.equity_curve]
            equity = [e['equity'] for e in self.equity_curve]
            
            fig.add_trace(
                go.Scatter(x=dates, y=equity, name='Equity', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Drawdown
            equity_series = pd.Series(equity, index=dates)
            cummax = equity_series.cummax()
            drawdown = (equity_series - cummax) / cummax * 100
            
            fig.add_trace(
                go.Scatter(x=dates, y=drawdown, name='Drawdown %', 
                         fill='tozeroy', line=dict(color='red')),
                row=1, col=2
            )
        
        # Trade distribution
        if self.trades:
            pnls = [t.pnl for t in self.trades if t.status == 'closed' and t.pnl is not None]
            
            fig.add_trace(
                go.Histogram(x=pnls, name='P&L Distribution', nbinsx=30),
                row=2, col=1
            )
            
            # Win/Loss pie chart
            wins = len([p for p in pnls if p > 0])
            losses = len([p for p in pnls if p <= 0])
            
            fig.add_trace(
                go.Pie(labels=['Wins', 'Losses'], values=[wins, losses],
                      marker=dict(colors=['green', 'red'])),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Backtest Report - Final Capital: ${results.get('final_capital', 0):,.2f}",
            showlegend=False,
            height=1000
        )
        
        # Save report
        report_path = Path('reports') / 'backtest_report.html'
        report_path.parent.mkdir(exist_ok=True)
        
        # Create HTML with metrics
        html_content = f"""
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
                .metric {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
            </style>
        </head>
        <body>
            <h1>ML Trading System Backtest Report</h1>
            
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{results.get('win_rate', 0):.1%}</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results.get('total_return', 0):.1%}</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results.get('sharpe_ratio', 0):.2f}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results.get('profit_factor', 0):.2f}</div>
                    <div class="metric-label">Profit Factor</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results.get('max_drawdown', 0):.1%}</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results.get('total_trades', 0)}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
            </div>
            
            {fig.to_html(include_plotlyjs='cdn')}
            
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Backtest report saved to {report_path}")
    
    def _generate_backtest_report(self):
        """Generate comprehensive backtest report"""
        self.generate_report(self.metrics)


if __name__ == "__main__":
    # Test backtesting
    engine = BacktestEngine()
    
    config = {
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 10000,
        'commission': 0.001,
        'slippage': 0.001
    }
    
    results = engine.run_backtest(config)
    
    print("\nBacktest Results:")
    for key, value in results.items():
        if isinstance(value, float):
            if 'rate' in key or 'return' in key or 'drawdown' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")