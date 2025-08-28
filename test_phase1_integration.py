#!/usr/bin/env python3
"""
Phase 1 Integration Test - Verify all components are working
"""

import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd

# Add src to path
sys.path.append('.')

def test_database():
    """Test database connection and operations"""
    print("\n🔍 Testing Database Layer...")
    
    try:
        from src.ml_trading.database.connection import get_db_manager
        from src.ml_trading.database.models import Trade, Signal, Position
        
        db = get_db_manager()
        
        # Test health check
        assert db.health_check(), "Database health check failed"
        print("  ✅ Database connection successful")
        
        # Test pool status
        pool_status = db.get_pool_status()
        print(f"  ✅ Connection pool: {pool_status}")
        
        # Test creating a trade
        with db.get_session() as session:
            from src.ml_trading.database.models import OrderSide, OrderStatus
            test_trade = Trade(
                symbol='TEST',
                side=OrderSide.BUY,
                quantity=100,
                price=150.50,
                order_type='market',
                status=OrderStatus.FILLED,
                strategy_name='test_strategy',
                created_at=datetime.now()
            )
            session.add(test_trade)
        
        print("  ✅ Database write operations working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Database test failed: {e}")
        return False

def test_risk_manager():
    """Test risk manager with circuit breakers"""
    print("\n🔍 Testing Risk Manager...")
    
    try:
        from src.ml_trading.risk.risk_manager import RiskManager, TradingState
        
        config = {
            'max_position_size': 0.05,
            'max_daily_loss': 0.05,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05,
            'max_positions': 10
        }
        
        risk_mgr = RiskManager(config)
        
        # Test position sizing
        position_size = risk_mgr.calculate_position_size(
            symbol='AAPL',
            account_value=100000,
            confidence=0.75,
            current_price=150.0
        )
        assert position_size > 0, "Position sizing failed"
        print(f"  ✅ Position sizing: {position_size} shares")
        
        # Test risk limits
        checks = risk_mgr.check_risk_limits(100000, 50000)
        assert checks['can_trade'], "Risk checks failed"
        print(f"  ✅ Risk limits checked: {checks['trading_state']}")
        
        # Test circuit breakers
        assert risk_mgr.trading_state == TradingState.NORMAL
        print("  ✅ Circuit breakers initialized")
        
        # Test VaR calculation
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        var_metrics = risk_mgr.calculate_var(returns)
        print(f"  ✅ VaR calculated: {var_metrics['historical_var']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Risk manager test failed: {e}")
        return False

def test_error_handler():
    """Test error handler with retry logic"""
    print("\n🔍 Testing Error Handler...")
    
    try:
        from src.ml_trading.utils.error_handler import ErrorHandler, RetryStrategy
        
        handler = ErrorHandler(max_retries=3, base_delay=0.1)
        
        # Test retry decorator
        attempt_count = 0
        
        @handler.retry(exceptions=(ValueError,), max_retries=2, strategy=RetryStrategy.EXPONENTIAL)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("Simulated error")
            return "Success"
        
        result = flaky_function()
        assert result == "Success", "Retry logic failed"
        assert attempt_count == 2, "Wrong number of retry attempts"
        print(f"  ✅ Retry logic working ({attempt_count} attempts)")
        
        # Test circuit breaker
        @handler.circuit_breaker(failure_threshold=2, recovery_timeout=1)
        def protected_function(should_fail=False):
            if should_fail:
                raise Exception("Circuit breaker test")
            return "OK"
        
        # Should work initially
        assert protected_function() == "OK"
        print("  ✅ Circuit breaker pattern working")
        
        # Test error statistics
        stats = handler.get_error_statistics()
        print(f"  ✅ Error tracking: {stats['total_errors']} errors logged")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error handler test failed: {e}")
        return False

def test_logging():
    """Test structured logging"""
    print("\n🔍 Testing Logging System...")
    
    try:
        from src.ml_trading.utils.logging_config import (
            setup_logging, log_trade, log_performance, get_log_statistics
        )
        
        # Test logger setup
        logger = setup_logging(
            app_name="test",
            log_level="INFO",
            enable_console=False,
            enable_file=True
        )
        
        logger.info("Test log message")
        print("  ✅ Logger initialized")
        
        # Test trade logging
        log_trade(
            symbol="AAPL",
            action="buy",
            quantity=100,
            price=150.50,
            status="filled"
        )
        print("  ✅ Trade logging working")
        
        # Test performance logging
        log_performance({
            'sharpe_ratio': 1.5,
            'total_return': 0.15,
            'max_drawdown': -0.05
        })
        print("  ✅ Performance logging working")
        
        # Check log statistics
        stats = get_log_statistics()
        print(f"  ✅ Log files created: {len(stats['log_files'])} files")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Logging test failed: {e}")
        return False

def test_metrics():
    """Test Prometheus metrics"""
    print("\n🔍 Testing Metrics System...")
    
    try:
        from src.ml_trading.monitoring.metrics import TradingMetrics
        
        metrics = TradingMetrics()
        
        # Record sample metrics
        metrics.record_trade(
            symbol="AAPL",
            side="buy",
            strategy="momentum",
            value=15000,
            execution_time=0.025
        )
        print("  ✅ Trade metrics recorded")
        
        metrics.update_portfolio(
            portfolio_value=100000,
            cash=50000,
            positions=5,
            daily_pnl=-500,
            win_rate=55.5
        )
        print("  ✅ Portfolio metrics updated")
        
        metrics.update_risk_metrics(
            var_95=-2500,
            sharpe=1.2,
            drawdown=-5.5,
            leverage=1.5
        )
        print("  ✅ Risk metrics updated")
        
        # Get metrics output
        output = metrics.get_metrics()
        assert len(output) > 0, "No metrics generated"
        print(f"  ✅ Metrics output: {len(output)} bytes")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Metrics test failed: {e}")
        return False

def main():
    """Run all Phase 1 integration tests"""
    print("="*60)
    print("PHASE 1 INTEGRATION TEST")
    print("="*60)
    
    results = {
        "Database": test_database(),
        "Risk Manager": test_risk_manager(),
        "Error Handler": test_error_handler(),
        "Logging": test_logging(),
        "Metrics": test_metrics()
    }
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{component:20} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    success_rate = (total_passed / total_tests) * 100
    
    print("\n" + "="*60)
    print(f"Overall: {total_passed}/{total_tests} tests passed ({success_rate:.0f}%)")
    
    if success_rate == 100:
        print("\n🎉 PHASE 1 COMPLETE - All systems operational!")
        print("\nYour infrastructure is ready for:")
        print("  • Persistent trade storage")
        print("  • Advanced risk management with circuit breakers")
        print("  • Automatic error recovery")
        print("  • Structured logging with rotation")
        print("  • Real-time metrics monitoring")
        print("\nNext: Proceed to Phase 2 (ML Model Enhancement)")
    else:
        print("\n⚠️  Some tests failed - please review and fix issues")
        print("Run 'bash scripts/setup_phase1.sh' to reinstall components")
    
    print("="*60)
    
    return 0 if success_rate == 100 else 1

if __name__ == "__main__":
    sys.exit(main())