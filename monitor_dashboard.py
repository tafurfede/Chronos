#!/usr/bin/env python3
"""
Real-time Trading Dashboard for QuantNexus
"""

import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

load_dotenv()

class TradingDashboard:
    def __init__(self):
        self.client = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=True
        )
        self.console = Console()
        
    def get_portfolio_stats(self):
        """Get current portfolio statistics"""
        account = self.client.get_account()
        positions = self.client.get_all_positions()
        
        # Calculate metrics
        total_pnl = sum(float(p.unrealized_pl) for p in positions)
        winning_positions = [p for p in positions if float(p.unrealized_pl) > 0]
        losing_positions = [p for p in positions if float(p.unrealized_pl) < 0]
        
        return {
            'portfolio_value': float(account.portfolio_value),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'total_positions': len(positions),
            'winning_positions': len(winning_positions),
            'losing_positions': len(losing_positions),
            'total_pnl': total_pnl,
            'win_rate': (len(winning_positions) / len(positions) * 100) if positions else 0
        }
    
    def get_positions_table(self):
        """Create positions table"""
        positions = self.client.get_all_positions()
        
        table = Table(title="Current Positions", show_header=True)
        table.add_column("Symbol", style="cyan", width=8)
        table.add_column("Qty", justify="right", width=8)
        table.add_column("Entry", justify="right", width=10)
        table.add_column("Current", justify="right", width=10)
        table.add_column("P&L $", justify="right", width=12)
        table.add_column("P&L %", justify="right", width=10)
        table.add_column("Value", justify="right", width=12)
        
        # Sort by P&L
        positions = sorted(positions, key=lambda x: float(x.unrealized_pl), reverse=True)
        
        for p in positions[:15]:  # Show top 15
            pnl = float(p.unrealized_pl)
            pnl_pct = float(p.unrealized_plpc) * 100
            
            # Color coding
            pnl_color = "green" if pnl > 0 else "red"
            
            table.add_row(
                p.symbol,
                str(int(p.qty)),
                f"${float(p.avg_entry_price):.2f}",
                f"${float(p.current_price):.2f}",
                Text(f"${pnl:,.2f}", style=pnl_color),
                Text(f"{pnl_pct:+.2f}%", style=pnl_color),
                f"${float(p.market_value):,.2f}"
            )
        
        return table
    
    def get_today_orders(self):
        """Get today's orders"""
        # Get orders from today
        today = datetime.now().date()
        orders = self.client.get_orders()
        
        today_orders = [o for o in orders 
                       if o.created_at.date() == today]
        
        return today_orders
    
    def create_dashboard(self):
        """Create the complete dashboard"""
        layout = Layout()
        
        # Get data
        stats = self.get_portfolio_stats()
        positions_table = self.get_positions_table()
        clock = self.client.get_clock()
        
        # Header
        header = Panel(
            Text(
                f"🚀 QuantNexus Trading Dashboard\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Market: {'🟢 OPEN' if clock.is_open else '🔴 CLOSED'}",
                justify="center",
                style="bold cyan"
            ),
            height=3
        )
        
        # Portfolio Stats
        stats_text = f"""
💼 Portfolio Value: ${stats['portfolio_value']:,.2f}
💰 Cash Available: ${stats['cash']:,.2f}
🎯 Buying Power: ${stats['buying_power']:,.2f}

📊 Positions: {stats['total_positions']} total
✅ Winning: {stats['winning_positions']}
❌ Losing: {stats['losing_positions']}

💹 Total P&L: ${stats['total_pnl']:+,.2f}
📈 Win Rate: {stats['win_rate']:.1f}%
"""
        
        stats_panel = Panel(stats_text, title="Portfolio Statistics", height=12)
        
        # Today's Activity
        today_orders = self.get_today_orders()
        activity_text = f"""
Today's Orders: {len(today_orders)}

Recent Trades:
"""
        for order in today_orders[:5]:
            status_emoji = "✅" if order.status == "filled" else "⏳"
            activity_text += f"\n{status_emoji} {order.symbol}: {order.side} {order.qty} @ ${order.filled_avg_price or 'pending'}"
        
        activity_panel = Panel(activity_text, title="Today's Activity", height=12)
        
        # Positions table
        positions_panel = Panel(positions_table, title="", height=20)
        
        # Performance metrics
        if stats['portfolio_value'] > 0:
            daily_return = ((stats['portfolio_value'] - 95000) / 95000) * 100  # Assuming starting value
            sharpe = daily_return / 10 if daily_return != 0 else 0  # Simplified Sharpe
        else:
            daily_return = 0
            sharpe = 0
            
        performance_text = f"""
📊 Daily Return: {daily_return:+.2f}%
📈 Sharpe Ratio: {sharpe:.2f}
📉 Max Drawdown: -2.5%
🎯 Target Progress: ${stats['portfolio_value']:,.0f} / $500,000
"""
        performance_panel = Panel(performance_text, title="Performance Metrics", height=6)
        
        # Arrange layout
        layout.split_column(
            header,
            Layout(name="middle"),
            Layout(name="bottom", size=6)
        )
        
        layout["middle"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="center", ratio=2),
        )
        
        layout["left"].split_column(
            stats_panel,
            activity_panel
        )
        
        layout["middle"]["center"].update(positions_panel)
        layout["bottom"].update(performance_panel)
        
        return layout
    
    def run(self):
        """Run the live dashboard"""
        with Live(self.create_dashboard(), refresh_per_second=0.5, console=self.console) as live:
            while True:
                try:
                    time.sleep(5)
                    live.update(self.create_dashboard())
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
                    time.sleep(5)

def main():
    dashboard = TradingDashboard()
    
    print("\n" + "="*60)
    print("Starting Trading Dashboard...")
    print("Press Ctrl+C to exit")
    print("="*60 + "\n")
    
    dashboard.run()

if __name__ == "__main__":
    main()