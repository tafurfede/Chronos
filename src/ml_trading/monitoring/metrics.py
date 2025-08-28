"""
Prometheus Metrics for ML Trading System Monitoring
"""

from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    CollectorRegistry, generate_latest, push_to_gateway,
    start_http_server
)
from typing import Dict, Any, Optional
import time
import threading
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TradingMetrics:
    """Prometheus metrics for trading system monitoring"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        # System Info
        self.system_info = Info(
            'trading_system_info',
            'Trading system information',
            registry=self.registry
        )
        
        # Trading Metrics
        self.trades_total = Counter(
            'trades_total',
            'Total number of trades executed',
            ['symbol', 'side', 'strategy'],
            registry=self.registry
        )
        
        self.trade_value = Summary(
            'trade_value_dollars',
            'Trade value in dollars',
            ['symbol', 'side'],
            registry=self.registry
        )
        
        self.trade_latency = Histogram(
            'trade_execution_latency_seconds',
            'Trade execution latency',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        # Portfolio Metrics
        self.portfolio_value = Gauge(
            'portfolio_value_dollars',
            'Total portfolio value',
            registry=self.registry
        )
        
        self.cash_balance = Gauge(
            'cash_balance_dollars',
            'Available cash balance',
            registry=self.registry
        )
        
        self.positions_count = Gauge(
            'positions_count',
            'Number of open positions',
            registry=self.registry
        )
        
        self.daily_pnl = Gauge(
            'daily_pnl_dollars',
            'Daily profit and loss',
            registry=self.registry
        )
        
        self.win_rate = Gauge(
            'win_rate_percentage',
            'Winning trade percentage',
            registry=self.registry
        )
        
        # Risk Metrics
        self.var_95 = Gauge(
            'value_at_risk_95',
            '95% Value at Risk',
            registry=self.registry
        )
        
        self.sharpe_ratio = Gauge(
            'sharpe_ratio',
            'Portfolio Sharpe ratio',
            registry=self.registry
        )
        
        self.max_drawdown = Gauge(
            'max_drawdown_percentage',
            'Maximum drawdown percentage',
            registry=self.registry
        )
        
        self.leverage_ratio = Gauge(
            'leverage_ratio',
            'Current leverage ratio',
            registry=self.registry
        )
        
        # Circuit Breaker Metrics
        self.circuit_breaker_trips = Counter(
            'circuit_breaker_trips_total',
            'Circuit breaker trips',
            ['breaker_type'],
            registry=self.registry
        )
        
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open)',
            ['breaker_type'],
            registry=self.registry
        )
        
        # Model Performance Metrics
        self.model_predictions = Counter(
            'model_predictions_total',
            'Total model predictions',
            ['model_name', 'prediction_type'],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'model_accuracy_percentage',
            'Model accuracy percentage',
            ['model_name'],
            registry=self.registry
        )
        
        self.model_inference_time = Histogram(
            'model_inference_seconds',
            'Model inference time',
            ['model_name'],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0],
            registry=self.registry
        )
        
        # Data Pipeline Metrics
        self.data_points_processed = Counter(
            'data_points_processed_total',
            'Total data points processed',
            ['data_source'],
            registry=self.registry
        )
        
        self.data_pipeline_errors = Counter(
            'data_pipeline_errors_total',
            'Data pipeline errors',
            ['error_type', 'data_source'],
            registry=self.registry
        )
        
        self.data_latency = Histogram(
            'data_fetch_latency_seconds',
            'Data fetch latency',
            ['data_source'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        # API Metrics
        self.api_requests = Counter(
            'api_requests_total',
            'API requests',
            ['endpoint', 'method', 'status'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['endpoint', 'method'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        # Error Metrics
        self.errors_total = Counter(
            'errors_total',
            'Total errors',
            ['component', 'severity'],
            registry=self.registry
        )
        
        # System Health
        self.system_health = Gauge(
            'system_health_score',
            'System health score (0-100)',
            registry=self.registry
        )
        
        self.database_connections = Gauge(
            'database_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.memory_usage_mb = Gauge(
            'memory_usage_megabytes',
            'Memory usage in MB',
            registry=self.registry
        )
        
        logger.info("Trading metrics initialized")
    
    def record_trade(self, symbol: str, side: str, strategy: str, 
                    value: float, execution_time: float):
        """Record trade metrics"""
        self.trades_total.labels(symbol=symbol, side=side, strategy=strategy).inc()
        self.trade_value.labels(symbol=symbol, side=side).observe(value)
        self.trade_latency.observe(execution_time)
    
    def update_portfolio(self, portfolio_value: float, cash: float, 
                        positions: int, daily_pnl: float, win_rate: float):
        """Update portfolio metrics"""
        self.portfolio_value.set(portfolio_value)
        self.cash_balance.set(cash)
        self.positions_count.set(positions)
        self.daily_pnl.set(daily_pnl)
        self.win_rate.set(win_rate)
    
    def update_risk_metrics(self, var_95: float, sharpe: float, 
                           drawdown: float, leverage: float):
        """Update risk metrics"""
        self.var_95.set(var_95)
        self.sharpe_ratio.set(sharpe)
        self.max_drawdown.set(drawdown)
        self.leverage_ratio.set(leverage)
    
    def record_circuit_breaker(self, breaker_type: str, is_open: bool):
        """Record circuit breaker event"""
        if is_open:
            self.circuit_breaker_trips.labels(breaker_type=breaker_type).inc()
        self.circuit_breaker_state.labels(breaker_type=breaker_type).set(1 if is_open else 0)
    
    def record_model_prediction(self, model_name: str, prediction_type: str, 
                               inference_time: float, accuracy: Optional[float] = None):
        """Record model prediction metrics"""
        self.model_predictions.labels(
            model_name=model_name, 
            prediction_type=prediction_type
        ).inc()
        self.model_inference_time.labels(model_name=model_name).observe(inference_time)
        
        if accuracy is not None:
            self.model_accuracy.labels(model_name=model_name).set(accuracy)
    
    def record_data_processing(self, source: str, points: int, latency: float):
        """Record data processing metrics"""
        self.data_points_processed.labels(data_source=source).inc(points)
        self.data_latency.labels(data_source=source).observe(latency)
    
    def record_error(self, component: str, severity: str):
        """Record error metrics"""
        self.errors_total.labels(component=component, severity=severity).inc()
    
    def update_system_health(self, health_score: float, db_connections: int, 
                            memory_mb: float):
        """Update system health metrics"""
        self.system_health.set(health_score)
        self.database_connections.set(db_connections)
        self.memory_usage_mb.set(memory_mb)
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry)


class MetricsServer:
    """HTTP server for Prometheus metrics"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.metrics = TradingMetrics()
        self.server_thread = None
        self.running = False
    
    def start(self):
        """Start metrics server"""
        if not self.running:
            start_http_server(self.port, registry=self.metrics.registry)
            self.running = True
            logger.info(f"Metrics server started on port {self.port}")
    
    def stop(self):
        """Stop metrics server"""
        self.running = False
        logger.info("Metrics server stopped")
    
    def get_metrics(self) -> TradingMetrics:
        """Get metrics instance"""
        return self.metrics


class MetricsCollector(threading.Thread):
    """Background thread for collecting system metrics"""
    
    def __init__(self, metrics: TradingMetrics, interval: int = 30):
        super().__init__(daemon=True)
        self.metrics = metrics
        self.interval = interval
        self.running = False
    
    def run(self):
        """Collect metrics periodically"""
        self.running = True
        
        while self.running:
            try:
                # Collect system metrics
                import psutil
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics.memory_usage_mb.set(memory.used / (1024 * 1024))
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Calculate health score
                health_score = 100
                if memory.percent > 80:
                    health_score -= 20
                if cpu_percent > 80:
                    health_score -= 20
                
                self.metrics.system_health.set(health_score)
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            time.sleep(self.interval)
    
    def stop(self):
        """Stop collector"""
        self.running = False


# Global metrics instance
_metrics_server: Optional[MetricsServer] = None

def get_metrics_server(port: int = 8000) -> MetricsServer:
    """Get or create metrics server singleton"""
    global _metrics_server
    if _metrics_server is None:
        _metrics_server = MetricsServer(port)
    return _metrics_server

def start_metrics_server(port: int = 8000):
    """Start the metrics server"""
    server = get_metrics_server(port)
    server.start()
    
    # Start background collector
    collector = MetricsCollector(server.metrics)
    collector.start()
    
    return server

def get_metrics() -> TradingMetrics:
    """Get metrics instance"""
    return get_metrics_server().get_metrics()