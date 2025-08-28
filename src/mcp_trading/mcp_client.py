#!/usr/bin/env python3
"""
MCP Client for Trading System
Provides easy interface to interact with MCP Trading Server
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

# MCP client imports
from mcp import Client
from mcp.types import Tool, Resource

logger = logging.getLogger(__name__)


class TradingMCPClient:
    """
    Client interface for MCP Trading Server
    Simplifies interaction with trading tools and resources
    """
    
    def __init__(self, server_url: str = "localhost:8080"):
        self.client = Client()
        self.server_url = server_url
        self.connected = False
        
    async def connect(self):
        """Connect to MCP server"""
        try:
            await self.client.connect(self.server_url)
            self.connected = True
            logger.info(f"Connected to MCP server at {self.server_url}")
            
            # List available tools
            tools = await self.client.list_tools()
            logger.info(f"Available tools: {[t.name for t in tools]}")
            
            # List available resources
            resources = await self.client.list_resources()
            logger.info(f"Available resources: {[r.uri for r in resources]}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.connected:
            await self.client.disconnect()
            self.connected = False
            logger.info("Disconnected from MCP server")
    
    # Market Analysis Methods
    async def analyze_stock(self, symbol: str, period: str = "1d") -> Dict:
        """
        Analyze a stock using ML models
        
        Args:
            symbol: Stock ticker
            period: Time period for analysis
            
        Returns:
            Comprehensive stock analysis
        """
        result = await self.client.call_tool(
            "analyze_stock",
            {"symbol": symbol, "period": period}
        )
        return json.loads(result.content[0].text)
    
    async def get_market_signals(self, min_confidence: float = 0.65) -> List[Dict]:
        """
        Get high-confidence trading signals
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of trading signals
        """
        result = await self.client.call_tool(
            "get_market_signals",
            {"min_confidence": min_confidence}
        )
        return json.loads(result.content[0].text)
    
    async def scan_opportunities(self, symbols: List[str] = None) -> List[Dict]:
        """
        Scan for trading opportunities
        
        Args:
            symbols: List of symbols to scan
            
        Returns:
            Trading opportunities ranked by confidence
        """
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']
        
        opportunities = []
        
        for symbol in symbols:
            try:
                analysis = await self.analyze_stock(symbol)
                
                if analysis['ml_prediction']['confidence'] >= 0.65:
                    opportunities.append({
                        'symbol': symbol,
                        'confidence': analysis['ml_prediction']['confidence'],
                        'expected_return': analysis['ml_prediction']['expected_return'],
                        'signal': analysis['ml_prediction']['signal'],
                        'current_price': analysis['current_price'],
                        'recommendation': analysis['recommendation']
                    })
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        # Sort by confidence
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        return opportunities
    
    # Trading Execution Methods
    async def execute_trade(
        self,
        symbol: str,
        action: str,
        quantity: int = 0,
        order_type: str = "market",
        limit_price: Optional[float] = None
    ) -> Dict:
        """
        Execute a trade
        
        Args:
            symbol: Stock ticker
            action: 'buy' or 'sell'
            quantity: Number of shares (0 for auto-sizing)
            order_type: 'market' or 'limit'
            limit_price: Price for limit orders
            
        Returns:
            Trade execution details
        """
        result = await self.client.call_tool(
            "execute_trade",
            {
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "order_type": order_type,
                "limit_price": limit_price
            }
        )
        return json.loads(result.content[0].text)
    
    async def close_position(self, symbol: str, reason: str = "manual") -> Dict:
        """
        Close a position
        
        Args:
            symbol: Stock ticker
            reason: Reason for closing
            
        Returns:
            Position closure details
        """
        result = await self.client.call_tool(
            "close_position",
            {"symbol": symbol, "reason": reason}
        )
        return json.loads(result.content[0].text)
    
    async def close_all_positions(self, reason: str = "end_of_day") -> List[Dict]:
        """
        Close all open positions
        
        Args:
            reason: Reason for closing
            
        Returns:
            List of closed positions
        """
        portfolio = await self.get_portfolio_status()
        closed_positions = []
        
        for symbol in portfolio['positions'].keys():
            try:
                result = await self.close_position(symbol, reason)
                closed_positions.append(result)
            except Exception as e:
                logger.error(f"Error closing {symbol}: {e}")
        
        return closed_positions
    
    # Portfolio Management Methods
    async def get_portfolio_status(self) -> Dict:
        """
        Get current portfolio status
        
        Returns:
            Portfolio summary
        """
        result = await self.client.call_tool("get_portfolio_status", {})
        return json.loads(result.content[0].text)
    
    async def calculate_risk_metrics(self) -> Dict:
        """
        Calculate portfolio risk metrics
        
        Returns:
            Risk metrics including VaR, Sharpe, etc.
        """
        result = await self.client.call_tool("calculate_risk_metrics", {})
        return json.loads(result.content[0].text)
    
    # ML Model Methods
    async def train_models(self, symbols: List[str] = None) -> Dict:
        """
        Train ML models
        
        Args:
            symbols: Symbols to train on
            
        Returns:
            Training results
        """
        result = await self.client.call_tool(
            "train_models",
            {"symbols": symbols}
        )
        return json.loads(result.content[0].text)
    
    async def get_model_performance(self) -> Dict:
        """
        Get ML model performance metrics
        
        Returns:
            Model performance statistics
        """
        result = await self.client.call_tool("get_model_performance", {})
        return json.loads(result.content[0].text)
    
    # Backtesting Methods
    async def run_backtest(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 10000
    ) -> Dict:
        """
        Run backtest
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
            
        Returns:
            Backtest results
        """
        result = await self.client.call_tool(
            "run_backtest",
            {
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": initial_capital
            }
        )
        return json.loads(result.content[0].text)
    
    # Resource Access Methods
    async def get_market_data(self, symbol: str) -> pd.DataFrame:
        """
        Get market data for a symbol
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Market data DataFrame
        """
        resource = await self.client.get_resource(f"market_data/{symbol}")
        data = json.loads(resource.text)
        return pd.DataFrame(data)
    
    async def get_latest_signals(self) -> List[Dict]:
        """
        Get latest trading signals
        
        Returns:
            List of recent signals
        """
        resource = await self.client.get_resource("signals/latest")
        return json.loads(resource.text)
    
    # High-Level Trading Workflows
    async def execute_daily_trading(self) -> Dict:
        """
        Execute complete daily trading workflow
        
        Returns:
            Daily trading summary
        """
        summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'signals_generated': [],
            'trades_executed': [],
            'positions_closed': [],
            'portfolio_status': None,
            'daily_pnl': 0
        }
        
        try:
            # 1. Get market signals
            logger.info("Scanning for market signals...")
            signals = await self.get_market_signals(min_confidence=0.65)
            summary['signals_generated'] = signals
            
            # 2. Execute top signals
            logger.info("Executing high-confidence trades...")
            for signal in signals[:3]:  # Top 3 signals
                if signal['signal'] == 'buy':
                    try:
                        trade = await self.execute_trade(
                            symbol=signal['symbol'],
                            action='buy',
                            quantity=0  # Auto-size
                        )
                        summary['trades_executed'].append(trade)
                        logger.info(f"Executed BUY {signal['symbol']}")
                    except Exception as e:
                        logger.error(f"Failed to execute trade for {signal['symbol']}: {e}")
            
            # 3. Manage existing positions
            logger.info("Managing existing positions...")
            portfolio = await self.get_portfolio_status()
            
            for symbol, position in portfolio['positions'].items():
                # Check if should close based on P&L
                if 'pnl_pct' in position:
                    if position['pnl_pct'] >= 0.03:  # 3% profit
                        result = await self.close_position(symbol, "profit_target")
                        summary['positions_closed'].append(result)
                        logger.info(f"Closed {symbol} - Profit target reached")
                    elif position['pnl_pct'] <= -0.015:  # 1.5% loss
                        result = await self.close_position(symbol, "stop_loss")
                        summary['positions_closed'].append(result)
                        logger.info(f"Closed {symbol} - Stop loss triggered")
            
            # 4. Get final portfolio status
            summary['portfolio_status'] = await self.get_portfolio_status()
            summary['daily_pnl'] = summary['portfolio_status']['daily_pnl']
            
            logger.info(f"Daily trading complete - P&L: ${summary['daily_pnl']:.2f}")
            
        except Exception as e:
            logger.error(f"Error in daily trading workflow: {e}")
        
        return summary
    
    async def monitor_positions_realtime(self, interval: int = 30):
        """
        Monitor positions in real-time
        
        Args:
            interval: Update interval in seconds
        """
        logger.info("Starting real-time position monitoring...")
        
        while True:
            try:
                # Get portfolio status
                portfolio = await self.get_portfolio_status()
                
                # Display position summary
                print("\n" + "="*50)
                print(f"Portfolio Value: ${portfolio['total_value']:.2f}")
                print(f"Cash: ${portfolio['cash']:.2f}")
                print(f"Daily P&L: ${portfolio['daily_pnl']:.2f}")
                print("\nPositions:")
                
                for symbol, position in portfolio['positions'].items():
                    pnl = position.get('pnl', 0)
                    pnl_pct = position.get('pnl_pct', 0)
                    print(f"  {symbol}: {position['quantity']} shares | "
                          f"P&L: ${pnl:.2f} ({pnl_pct:.1%})")
                
                # Check for alerts
                risk_metrics = await self.calculate_risk_metrics()
                
                if risk_metrics.get('max_drawdown', 0) < -0.10:
                    logger.warning("⚠️ Max drawdown exceeded 10%!")
                
                if risk_metrics.get('win_rate', 0) < 0.60:
                    logger.warning("⚠️ Win rate below 60%!")
                
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(interval)


class SimpleTradingClient:
    """
    Simplified synchronous client for easy scripting
    """
    
    def __init__(self, server_url: str = "localhost:8080"):
        self.async_client = TradingMCPClient(server_url)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def __enter__(self):
        self.loop.run_until_complete(self.async_client.connect())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.loop.run_until_complete(self.async_client.disconnect())
        self.loop.close()
    
    def analyze(self, symbol: str) -> Dict:
        """Analyze a stock"""
        return self.loop.run_until_complete(
            self.async_client.analyze_stock(symbol)
        )
    
    def buy(self, symbol: str, quantity: int = 0) -> Dict:
        """Buy a stock"""
        return self.loop.run_until_complete(
            self.async_client.execute_trade(symbol, "buy", quantity)
        )
    
    def sell(self, symbol: str, quantity: int = 0) -> Dict:
        """Sell a stock"""
        return self.loop.run_until_complete(
            self.async_client.execute_trade(symbol, "sell", quantity)
        )
    
    def close(self, symbol: str) -> Dict:
        """Close a position"""
        return self.loop.run_until_complete(
            self.async_client.close_position(symbol)
        )
    
    def portfolio(self) -> Dict:
        """Get portfolio status"""
        return self.loop.run_until_complete(
            self.async_client.get_portfolio_status()
        )
    
    def signals(self) -> List[Dict]:
        """Get trading signals"""
        return self.loop.run_until_complete(
            self.async_client.get_market_signals()
        )
    
    def backtest(self, start: str, end: str, capital: float = 10000) -> Dict:
        """Run backtest"""
        return self.loop.run_until_complete(
            self.async_client.run_backtest(start, end, capital)
        )


# Example usage
async def main():
    """Example usage of MCP client"""
    
    # Create client
    client = TradingMCPClient()
    
    try:
        # Connect to server
        await client.connect()
        
        # Analyze a stock
        print("\n📊 Analyzing AAPL...")
        analysis = await client.analyze_stock("AAPL")
        print(f"Current Price: ${analysis['current_price']:.2f}")
        print(f"ML Signal: {analysis['ml_prediction']['signal']}")
        print(f"Confidence: {analysis['ml_prediction']['confidence']:.1%}")
        print(f"Recommendation: {analysis['recommendation']}")
        
        # Get market signals
        print("\n🎯 Getting market signals...")
        signals = await client.get_market_signals()
        for signal in signals[:3]:
            print(f"  {signal['symbol']}: {signal['signal']} "
                  f"(confidence: {signal['confidence']:.1%})")
        
        # Check portfolio
        print("\n💼 Portfolio Status...")
        portfolio = await client.get_portfolio_status()
        print(f"Total Value: ${portfolio['total_value']:.2f}")
        print(f"Cash: ${portfolio['cash']:.2f}")
        print(f"Positions: {len(portfolio['positions'])}")
        
        # Run daily trading
        print("\n🤖 Executing daily trading workflow...")
        summary = await client.execute_daily_trading()
        print(f"Trades executed: {len(summary['trades_executed'])}")
        print(f"Daily P&L: ${summary['daily_pnl']:.2f}")
        
    finally:
        await client.disconnect()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(main())