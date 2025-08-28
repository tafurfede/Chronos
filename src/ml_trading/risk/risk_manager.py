"""
Risk Management Module with Circuit Breakers - Critical for safe trading
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import numpy as np
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class TradingState(Enum):
    """Trading system states"""
    NORMAL = "normal"
    WARNING = "warning"
    RESTRICTED = "restricted"
    HALTED = "halted"

class CircuitBreaker:
    """Circuit breaker implementation for risk control"""
    
    def __init__(self, 
                 threshold: float,
                 time_window: int = 300,  # 5 minutes
                 cooldown: int = 900):  # 15 minutes
        self.threshold = threshold
        self.time_window = time_window
        self.cooldown = cooldown
        self.trips = deque()
        self.last_trip = None
        self.is_tripped = False
        self.lock = threading.Lock()
    
    def check(self, value: float) -> bool:
        """Check if circuit breaker should trip"""
        with self.lock:
            if value >= self.threshold:
                self.trip()
                return True
            return False
    
    def trip(self):
        """Trip the circuit breaker"""
        self.is_tripped = True
        self.last_trip = datetime.now()
        self.trips.append(self.last_trip)
        logger.critical(f"Circuit breaker tripped at {self.last_trip}")
    
    def reset(self) -> bool:
        """Reset circuit breaker if cooldown passed"""
        with self.lock:
            if not self.is_tripped:
                return True
            
            if self.last_trip and \
               datetime.now() - self.last_trip > timedelta(seconds=self.cooldown):
                self.is_tripped = False
                logger.info("Circuit breaker reset")
                return True
            return False
    
    def get_status(self) -> Dict:
        """Get circuit breaker status"""
        return {
            'is_tripped': self.is_tripped,
            'last_trip': self.last_trip.isoformat() if self.last_trip else None,
            'trips_count': len(self.trips)
        }

class RiskManager:
    """Enhanced Risk Manager with Circuit Breakers and Advanced Controls"""
    
    def __init__(self, config: Dict):
        # Basic risk parameters
        self.max_position_size = config.get('max_position_size', 0.05)  # 5% max per trade
        self.max_daily_loss = config.get('max_daily_loss', 0.05)  # 5% daily loss limit
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)  # 2% stop loss
        self.take_profit_pct = config.get('take_profit_pct', 0.05)  # 5% take profit
        self.max_positions = config.get('max_positions', 10)
        
        # Enhanced risk controls
        self.max_concentration = config.get('max_concentration', 0.20)  # 20% in single stock
        self.max_sector_exposure = config.get('max_sector_exposure', 0.40)  # 40% per sector
        self.max_correlation = config.get('max_correlation', 0.70)  # Max correlation between positions
        self.max_daily_trades = config.get('max_daily_trades', 50)
        self.min_liquidity_ratio = config.get('min_liquidity_ratio', 0.30)  # 30% cash minimum
        
        # Circuit breakers
        self.loss_circuit_breaker = CircuitBreaker(
            threshold=self.max_daily_loss,
            time_window=300,
            cooldown=900
        )
        
        self.volatility_circuit_breaker = CircuitBreaker(
            threshold=0.03,  # 3% portfolio volatility
            time_window=60,
            cooldown=300
        )
        
        self.trade_frequency_breaker = CircuitBreaker(
            threshold=10,  # 10 trades in 1 minute
            time_window=60,
            cooldown=300
        )
        
        # State tracking
        self.daily_trades = 0
        self.daily_pnl = 0
        self.active_positions = {}
        self.position_correlations = {}
        self.sector_exposures = {}
        self.trading_state = TradingState.NORMAL
        self.alerts = deque(maxlen=100)
        self.trade_history = deque(maxlen=1000)
        
        # VaR calculation
        self.var_confidence = config.get('var_confidence', 0.95)
        self.var_lookback = config.get('var_lookback', 252)
        
        logger.info("Enhanced Risk Manager initialized with circuit breakers")
        
    def calculate_position_size(self, 
                               symbol: str,
                               account_value: float,
                               confidence: float,
                               current_price: float) -> int:
        """Calculate safe position size using Kelly Criterion"""
        
        # Base position size (% of portfolio)
        base_size = self.max_position_size * account_value
        
        # Adjust by confidence (simplified Kelly)
        kelly_fraction = (confidence - 0.5) * 2  # Convert to 0-1 range
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Final position value
        position_value = base_size * kelly_fraction
        
        # Convert to shares
        shares = int(position_value / current_price)
        
        # Safety checks
        max_shares = int((account_value * self.max_position_size) / current_price)
        shares = min(shares, max_shares)
        
        logger.info(f"Position sizing for {symbol}: {shares} shares @ ${current_price:.2f}")
        return shares
    
    def check_risk_limits(self, account_value: float, cash_available: float = None) -> Dict[str, any]:
        """Enhanced risk limit checking with circuit breakers"""
        
        checks = {
            'can_trade': True,
            'daily_loss_exceeded': False,
            'max_positions_reached': False,
            'margin_call_risk': False,
            'circuit_breakers': {},
            'trading_state': self.trading_state.value,
            'alerts': []
        }
        
        # Check circuit breakers
        daily_loss_pct = abs(self.daily_pnl / account_value) if account_value > 0 else 0
        
        if self.loss_circuit_breaker.check(daily_loss_pct):
            checks['daily_loss_exceeded'] = True
            checks['can_trade'] = False
            checks['trading_state'] = TradingState.HALTED
            self.trading_state = TradingState.HALTED
            alert = f"CRITICAL: Daily loss circuit breaker tripped! Loss: {daily_loss_pct:.2%}"
            checks['alerts'].append(alert)
            self.send_alert(alert, severity="critical")
        
        # Check position concentration
        if self.active_positions:
            total_position_value = sum(p.get('value', 0) for p in self.active_positions.values())
            for symbol, position in self.active_positions.items():
                position_pct = position.get('value', 0) / account_value if account_value > 0 else 0
                if position_pct > self.max_concentration:
                    checks['alerts'].append(f"Position concentration limit exceeded for {symbol}: {position_pct:.2%}")
                    checks['trading_state'] = TradingState.WARNING
        
        # Check trade frequency
        recent_trades = [t for t in self.trade_history 
                        if datetime.now() - t['timestamp'] < timedelta(minutes=1)]
        if len(recent_trades) >= 10:
            self.trade_frequency_breaker.trip()
            checks['can_trade'] = False
            checks['alerts'].append("Trade frequency limit exceeded")
        
        # Check liquidity ratio
        if cash_available is not None:
            liquidity_ratio = cash_available / account_value if account_value > 0 else 0
            if liquidity_ratio < self.min_liquidity_ratio:
                checks['trading_state'] = TradingState.RESTRICTED
                checks['alerts'].append(f"Low liquidity: {liquidity_ratio:.2%}")
        
        # Check position limit
        if len(self.active_positions) >= self.max_positions:
            checks['max_positions_reached'] = True
            checks['can_trade'] = False
            logger.warning(f"Max positions reached: {len(self.active_positions)}")
        
        # Check PDT rule
        if account_value < 25000:
            checks['margin_call_risk'] = True
            checks['alerts'].append("Account below $25K - PDT restrictions apply")
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            checks['can_trade'] = False
            checks['alerts'].append(f"Daily trade limit reached: {self.daily_trades}")
        
        # Collect circuit breaker status
        checks['circuit_breakers'] = {
            'loss': self.loss_circuit_breaker.get_status(),
            'volatility': self.volatility_circuit_breaker.get_status(),
            'frequency': self.trade_frequency_breaker.get_status()
        }
        
        return checks
    
    def send_alert(self, message: str, severity: str = "warning"):
        """Send risk alert"""
        alert = {
            'timestamp': datetime.now(),
            'message': message,
            'severity': severity
        }
        self.alerts.append(alert)
        
        # Log based on severity
        if severity == "critical":
            logger.critical(message)
        elif severity == "high":
            logger.error(message)
        elif severity == "medium":
            logger.warning(message)
        else:
            logger.info(message)
    
    def halt_trading(self, reason: str = "Manual halt"):
        """Emergency trading halt"""
        self.trading_state = TradingState.HALTED
        self.send_alert(f"TRADING HALTED: {reason}", severity="critical")
        
        # Trip all circuit breakers
        self.loss_circuit_breaker.trip()
        self.volatility_circuit_breaker.trip()
        self.trade_frequency_breaker.trip()
        
        logger.critical(f"Trading system halted: {reason}")
        
    def resume_trading(self) -> bool:
        """Attempt to resume trading after halt"""
        # Check if all circuit breakers can be reset
        can_resume = all([
            self.loss_circuit_breaker.reset(),
            self.volatility_circuit_breaker.reset(),
            self.trade_frequency_breaker.reset()
        ])
        
        if can_resume:
            self.trading_state = TradingState.NORMAL
            self.send_alert("Trading resumed", severity="info")
            return True
        else:
            self.send_alert("Cannot resume trading - circuit breakers still active", severity="warning")
            return False
    
    def calculate_stop_loss(self, entry_price: float, position_type: str = 'long') -> float:
        """Calculate stop loss price"""
        if position_type == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, position_type: str = 'long') -> float:
        """Calculate take profit price"""
        if position_type == 'long':
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)
    
    def update_position(self, symbol: str, shares: int, entry_price: float):
        """Track position for risk management"""
        self.active_positions[symbol] = {
            'shares': shares,
            'entry_price': entry_price,
            'stop_loss': self.calculate_stop_loss(entry_price),
            'take_profit': self.calculate_take_profit(entry_price)
        }
        self.daily_trades += 1
    
    def should_exit_position(self, symbol: str, current_price: float) -> Optional[str]:
        """Check if position should be exited"""
        if symbol not in self.active_positions:
            return None
        
        position = self.active_positions[symbol]
        
        # Check stop loss
        if current_price <= position['stop_loss']:
            logger.warning(f"Stop loss triggered for {symbol} at ${current_price:.2f}")
            return 'stop_loss'
        
        # Check take profit
        if current_price >= position['take_profit']:
            logger.info(f"Take profit triggered for {symbol} at ${current_price:.2f}")
            return 'take_profit'
        
        return None
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'active_positions': len(self.active_positions),
            'position_details': self.active_positions
        }
    
    def calculate_var(self, returns: pd.Series, confidence: float = None) -> Dict[str, float]:
        """Calculate Value at Risk (VaR) and Conditional VaR"""
        confidence = confidence or self.var_confidence
        
        # Historical VaR
        var_percentile = (1 - confidence) * 100
        historical_var = np.percentile(returns, var_percentile)
        
        # Parametric VaR (assumes normal distribution)
        mean_return = returns.mean()
        std_return = returns.std()
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence)
        parametric_var = mean_return + z_score * std_return
        
        # Conditional VaR (CVaR) - Expected Shortfall
        losses = returns[returns < historical_var]
        cvar = losses.mean() if len(losses) > 0 else historical_var
        
        return {
            'historical_var': historical_var,
            'parametric_var': parametric_var,
            'cvar': cvar,
            'confidence': confidence
        }
    
    def calculate_portfolio_volatility(self, returns: pd.DataFrame) -> float:
        """Calculate portfolio volatility"""
        if returns.empty:
            return 0.0
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Calculate portfolio volatility
        weights = np.array([1/len(returns.columns)] * len(returns.columns))
        portfolio_std = np.sqrt(weights.T @ returns.cov() @ weights)
        
        return portfolio_std
    
    def update_position_tracking(self, symbol: str, shares: int, price: float, action: str):
        """Update position tracking with trade history"""
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'action': action
        }
        self.trade_history.append(trade)
        self.daily_trades += 1
        
        # Check trade frequency
        if self.daily_trades > 0 and self.daily_trades % 10 == 0:
            self.send_alert(f"High trading activity: {self.daily_trades} trades today", severity="medium")
    
    def get_risk_dashboard(self) -> Dict:
        """Get comprehensive risk dashboard"""
        return {
            'trading_state': self.trading_state.value,
            'daily_metrics': {
                'trades': self.daily_trades,
                'pnl': self.daily_pnl,
                'pnl_pct': (self.daily_pnl / 100000) if self.daily_pnl else 0  # Assumes 100k account
            },
            'positions': {
                'count': len(self.active_positions),
                'max_allowed': self.max_positions,
                'details': self.active_positions
            },
            'circuit_breakers': {
                'loss': self.loss_circuit_breaker.get_status(),
                'volatility': self.volatility_circuit_breaker.get_status(),
                'frequency': self.trade_frequency_breaker.get_status()
            },
            'recent_alerts': list(self.alerts)[-10:],  # Last 10 alerts
            'limits': {
                'max_position_size': self.max_position_size,
                'max_daily_loss': self.max_daily_loss,
                'max_concentration': self.max_concentration,
                'max_daily_trades': self.max_daily_trades
            }
        }
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at market open)"""
        self.daily_trades = 0
        self.daily_pnl = 0
        
        # Reset circuit breakers if cooldown passed
        self.loss_circuit_breaker.reset()
        self.volatility_circuit_breaker.reset()
        self.trade_frequency_breaker.reset()
        
        logger.info("Daily risk metrics reset")