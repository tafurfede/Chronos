#!/usr/bin/env python3
"""
Performance Analysis Module for ML Trading System
Real-time and historical performance tracking
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from pathlib import Path

# Visualization
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
import matplotlib.pyplot as plt
import seaborn as sns

# Performance metrics
import quantstats as qs

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for ML trading system
    """
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        self.trades_file = self.data_dir / "trades.json"
        self.metrics_file = self.data_dir / "metrics.json"
        
    def load_recent_trades(self, days: int = 30) -> List[Dict]:
        """Load recent trades from storage"""
        
        if not self.trades_file.exists():
            logger.warning("No trades file found")
            return []
        
        with open(self.trades_file, 'r') as f:
            all_trades = json.load(f)
        
        # Filter recent trades
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = [
            t for t in all_trades 
            if datetime.fromisoformat(t['timestamp']) > cutoff_date
        ]
        
        return recent_trades
    
    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate comprehensive trading metrics"""
        
        if not trades:
            return {'error': 'No trades to analyze'}
        
        df = pd.DataFrame(trades)
        
        # Basic statistics
        total_trades = len(df)
        
        # P&L analysis
        if 'pnl' in df.columns:
            winning_trades = df[df['pnl'] > 0]
            losing_trades = df[df['pnl'] <= 0]
            
            win_rate = len(winning_trades) / total_trades
            avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
            
            # Profit factor
            gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
            gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Expectancy
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
            
            # Calculate returns
            if 'capital' in df.columns:
                df['returns'] = df['pnl'] / df['capital']
                
                # Sharpe ratio
                sharpe_ratio = self._calculate_sharpe(df['returns'])
                
                # Sortino ratio
                sortino_ratio = self._calculate_sortino(df['returns'])
                
                # Maximum drawdown
                max_drawdown = self._calculate_max_drawdown(df['returns'])
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
                max_drawdown = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            expectancy = 0
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
        
        # Time-based analysis
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Best trading hours
            hourly_performance = df.groupby('hour')['pnl'].mean() if 'pnl' in df.columns else None
            best_hour = hourly_performance.idxmax() if hourly_performance is not None else None
            
            # Best trading days
            daily_performance = df.groupby('day_of_week')['pnl'].mean() if 'pnl' in df.columns else None
            best_day = daily_performance.idxmax() if daily_performance is not None else None
        else:
            best_hour = None
            best_day = None
        
        # Symbol analysis
        if 'symbol' in df.columns and 'pnl' in df.columns:
            symbol_performance = df.groupby('symbol').agg({
                'pnl': ['sum', 'mean', 'count']
            })
            best_symbol = symbol_performance[('pnl', 'sum')].idxmax()
            worst_symbol = symbol_performance[('pnl', 'sum')].idxmin()
        else:
            best_symbol = None
            worst_symbol = None
        
        # ML model performance
        if 'confidence' in df.columns and 'pnl' in df.columns:
            # Correlation between confidence and returns
            confidence_correlation = df['confidence'].corr(df['pnl'])
            
            # Performance by confidence buckets
            df['confidence_bucket'] = pd.cut(df['confidence'], bins=5)
            confidence_performance = df.groupby('confidence_bucket')['pnl'].mean()
        else:
            confidence_correlation = 0
            confidence_performance = None
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'best_hour': best_hour,
            'best_day': best_day,
            'best_symbol': best_symbol,
            'worst_symbol': worst_symbol,
            'confidence_correlation': confidence_correlation
        }
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        
        if returns.empty or returns.std() == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)
        
        return sharpe
    
    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        
        if returns.empty:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if downside_returns.empty:
            return 0
        
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0
        
        sortino = excess_returns.mean() / downside_std * np.sqrt(252)
        
        return sortino
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        
        if returns.empty:
            return 0
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return drawdown.min()
    
    def generate_charts(self, trades: List[Dict], metrics: Dict):
        """Generate performance analysis charts"""
        
        if not trades:
            logger.warning("No trades to visualize")
            return
        
        df = pd.DataFrame(trades)
        
        # Create subplots
        fig = sp.make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Cumulative P&L', 'Win Rate Over Time', 'P&L Distribution',
                'Performance by Hour', 'Performance by Symbol', 'Confidence vs Returns',
                'Trade Duration Distribution', 'Monthly Returns', 'Risk-Adjusted Returns'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'histogram'}],
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'histogram'}, {'type': 'bar'}, {'type': 'scatter'}]
            ]
        )
        
        # 1. Cumulative P&L
        if 'pnl' in df.columns and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            df['cumulative_pnl'] = df['pnl'].cumsum()
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['cumulative_pnl'],
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # 2. Win Rate Over Time (Rolling)
        if 'pnl' in df.columns:
            df['win'] = (df['pnl'] > 0).astype(int)
            df['rolling_win_rate'] = df['win'].rolling(window=20, min_periods=1).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['rolling_win_rate'],
                    mode='lines',
                    name='Win Rate',
                    line=dict(color='green', width=2)
                ),
                row=1, col=2
            )
            
            # Add 65% target line
            fig.add_hline(y=0.65, line_dash="dash", line_color="red", row=1, col=2)
        
        # 3. P&L Distribution
        if 'pnl' in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df['pnl'],
                    nbinsx=30,
                    name='P&L Distribution',
                    marker_color='lightblue'
                ),
                row=1, col=3
            )
        
        # 4. Performance by Hour
        if 'timestamp' in df.columns and 'pnl' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            hourly_perf = df.groupby('hour')['pnl'].mean()
            
            fig.add_trace(
                go.Bar(
                    x=hourly_perf.index,
                    y=hourly_perf.values,
                    name='Hourly Performance',
                    marker_color='orange'
                ),
                row=2, col=1
            )
        
        # 5. Performance by Symbol
        if 'symbol' in df.columns and 'pnl' in df.columns:
            symbol_perf = df.groupby('symbol')['pnl'].sum().sort_values(ascending=False).head(10)
            
            fig.add_trace(
                go.Bar(
                    x=symbol_perf.index,
                    y=symbol_perf.values,
                    name='Symbol Performance',
                    marker_color='purple'
                ),
                row=2, col=2
            )
        
        # 6. Confidence vs Returns
        if 'confidence' in df.columns and 'pnl' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['confidence'],
                    y=df['pnl'],
                    mode='markers',
                    name='Confidence vs P&L',
                    marker=dict(
                        color=df['pnl'],
                        colorscale='RdYlGn',
                        showscale=True,
                        size=8
                    )
                ),
                row=2, col=3
            )
        
        # 7. Trade Duration Distribution
        if 'duration' in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df['duration'],
                    nbinsx=20,
                    name='Trade Duration',
                    marker_color='cyan'
                ),
                row=3, col=1
            )
        
        # 8. Monthly Returns
        if 'timestamp' in df.columns and 'pnl' in df.columns:
            df['month'] = df['timestamp'].dt.to_period('M')
            monthly_returns = df.groupby('month')['pnl'].sum()
            
            fig.add_trace(
                go.Bar(
                    x=monthly_returns.index.astype(str),
                    y=monthly_returns.values,
                    name='Monthly Returns',
                    marker_color=['red' if x < 0 else 'green' for x in monthly_returns.values]
                ),
                row=3, col=2
            )
        
        # 9. Risk-Adjusted Returns (Sharpe over time)
        if 'pnl' in df.columns and len(df) > 20:
            df['rolling_sharpe'] = df['pnl'].rolling(window=20).apply(
                lambda x: self._calculate_sharpe(pd.Series(x))
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['rolling_sharpe'],
                    mode='lines',
                    name='Rolling Sharpe',
                    line=dict(color='brown', width=2)
                ),
                row=3, col=3
            )
        
        # Update layout
        fig.update_layout(
            title='ML Trading System Performance Analysis',
            showlegend=False,
            height=1200,
            width=1800
        )
        
        # Save chart
        chart_path = self.reports_dir / 'analysis_charts.html'
        fig.write_html(str(chart_path))
        
        logger.info(f"Analysis charts saved to {chart_path}")
        
        # Also generate a performance summary table
        self._generate_summary_table(metrics)
    
    def _generate_summary_table(self, metrics: Dict):
        """Generate performance summary table"""
        
        html_content = """
        <html>
        <head>
            <title>Performance Summary</title>
            <style>
                body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 25px; margin-top: 30px; }
                .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 8px; color: white; }
                .metric-value { font-size: 32px; font-weight: bold; margin: 10px 0; }
                .metric-label { font-size: 14px; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; }
                .good { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
                .warning { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
                .bad { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
                .target-line { margin-top: 30px; padding: 20px; background: #e8f4fd; border-left: 4px solid #3498db; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 ML Trading System Performance</h1>
                
                <div class="target-line">
                    <strong>Target:</strong> 65%+ Win Rate | $10K → $500K in 12 months
                </div>
                
                <div class="metrics-grid">
        """
        
        # Add metric cards
        metric_cards = [
            ('Win Rate', metrics.get('win_rate', 0), 'percentage', 'good' if metrics.get('win_rate', 0) >= 0.65 else 'warning'),
            ('Total Trades', metrics.get('total_trades', 0), 'number', 'good'),
            ('Profit Factor', metrics.get('profit_factor', 0), 'decimal', 'good' if metrics.get('profit_factor', 0) > 1.5 else 'warning'),
            ('Sharpe Ratio', metrics.get('sharpe_ratio', 0), 'decimal', 'good' if metrics.get('sharpe_ratio', 0) > 2 else 'warning'),
            ('Max Drawdown', metrics.get('max_drawdown', 0), 'percentage', 'good' if abs(metrics.get('max_drawdown', 0)) < 0.15 else 'warning'),
            ('Expectancy', metrics.get('expectancy', 0), 'currency', 'good' if metrics.get('expectancy', 0) > 0 else 'bad'),
        ]
        
        for label, value, format_type, css_class in metric_cards:
            if format_type == 'percentage':
                formatted_value = f"{value:.1%}"
            elif format_type == 'currency':
                formatted_value = f"${value:.2f}"
            elif format_type == 'decimal':
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = f"{value}"
            
            html_content += f"""
                <div class="metric-card {css_class}">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{formatted_value}</div>
                </div>
            """
        
        html_content += """
                </div>
                
                <div style="margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                    <h3>📊 Key Insights</h3>
                    <ul style="line-height: 1.8;">
        """
        
        # Add insights
        if metrics.get('win_rate', 0) >= 0.65:
            html_content += "<li>✅ <strong>Win rate target achieved!</strong> System is performing above 65% success rate.</li>"
        else:
            html_content += f"<li>⚠️ Win rate at {metrics.get('win_rate', 0):.1%}, needs improvement to reach 65% target.</li>"
        
        if metrics.get('best_hour') is not None:
            html_content += f"<li>🕐 Best trading hour: {metrics.get('best_hour')}:00</li>"
        
        if metrics.get('best_symbol'):
            html_content += f"<li>💎 Best performing symbol: {metrics.get('best_symbol')}</li>"
        
        if metrics.get('confidence_correlation', 0) > 0.5:
            html_content += "<li>🎯 Strong correlation between ML confidence and returns - models are calibrated well.</li>"
        
        html_content += """
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save summary
        summary_path = self.reports_dir / 'performance_summary.html'
        with open(summary_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Performance summary saved to {summary_path}")
    
    def track_live_performance(self, trade: Dict):
        """Track live trading performance"""
        
        # Load existing trades
        if self.trades_file.exists():
            with open(self.trades_file, 'r') as f:
                trades = json.load(f)
        else:
            trades = []
        
        # Add new trade
        trades.append(trade)
        
        # Save updated trades
        with open(self.trades_file, 'w') as f:
            json.dump(trades, f, indent=2, default=str)
        
        # Update metrics
        metrics = self.calculate_metrics(trades)
        
        # Save metrics
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Log performance
        logger.info(f"Trade recorded - Win Rate: {metrics.get('win_rate', 0):.1%}, "
                   f"Total P&L: ${metrics.get('total_pnl', 0):.2f}")
    
    def generate_daily_report(self):
        """Generate daily performance report"""
        
        # Load today's trades
        trades = self.load_recent_trades(days=1)
        
        if not trades:
            logger.info("No trades today")
            return
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades)
        
        # Generate report
        report = f"""
        Daily Trading Report - {datetime.now().strftime('%Y-%m-%d')}
        {'='*50}
        
        Trades Executed: {metrics.get('total_trades', 0)}
        Win Rate: {metrics.get('win_rate', 0):.1%}
        Total P&L: ${metrics.get('total_pnl', 0):.2f}
        Average Win: ${metrics.get('avg_win', 0):.2f}
        Average Loss: ${metrics.get('avg_loss', 0):.2f}
        Profit Factor: {metrics.get('profit_factor', 0):.2f}
        
        Best Performing Symbol: {metrics.get('best_symbol', 'N/A')}
        Worst Performing Symbol: {metrics.get('worst_symbol', 'N/A')}
        
        Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
        Max Drawdown: {metrics.get('max_drawdown', 0):.1%}
        """
        
        # Save report
        report_path = self.reports_dir / f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Daily report saved to {report_path}")
        
        return report


if __name__ == "__main__":
    # Test performance analysis
    analyzer = PerformanceAnalyzer()
    
    # Create sample trades for testing
    sample_trades = [
        {
            'symbol': 'AAPL',
            'timestamp': datetime.now() - timedelta(days=i),
            'pnl': np.random.normal(50, 100),
            'confidence': np.random.uniform(0.5, 0.9),
            'capital': 10000,
            'duration': np.random.randint(1, 48)
        }
        for i in range(100)
    ]
    
    # Calculate metrics
    metrics = analyzer.calculate_metrics(sample_trades)
    
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Generate charts
    analyzer.generate_charts(sample_trades, metrics)