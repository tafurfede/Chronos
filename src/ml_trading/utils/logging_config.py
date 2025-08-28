"""
Structured Logging Configuration with Rotation
"""

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from pythonjsonlogger import jsonlogger

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

class ContextFilter(logging.Filter):
    """Add context information to log records"""
    
    def __init__(self, context: Dict[str, Any] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record):
        # Add context to record
        for key, value in self.context.items():
            setattr(record, key, value)
        
        # Add default fields
        record.timestamp = datetime.utcnow().isoformat()
        record.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        record.process_id = os.getpid()
        
        return True

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        # Add custom fields
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)

def setup_logging(
    app_name: str = "quantnexus",
    log_level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = True,
    context: Dict[str, Any] = None
) -> logging.Logger:
    """
    Setup structured logging with rotation
    
    Args:
        app_name: Application name for log files
        log_level: Logging level
        enable_console: Enable console output
        enable_file: Enable file output
        enable_json: Enable JSON structured logging
        context: Additional context to add to all logs
    
    Returns:
        Configured logger
    """
    
    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Add context filter
    if context:
        logger.addFilter(ContextFilter(context))
    
    # Console Handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if enable_json:
            console_formatter = CustomJsonFormatter()
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File Handlers with Rotation
    if enable_file:
        # Main log file (all levels)
        main_log_file = LOGS_DIR / f"{app_name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        if enable_json:
            file_formatter = CustomJsonFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Error log file (errors and above)
        error_log_file = LOGS_DIR / f"{app_name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
        
        # Time-based rotation for daily logs
        daily_log_file = LOGS_DIR / f"{app_name}_daily.log"
        time_handler = logging.handlers.TimedRotatingFileHandler(
            daily_log_file,
            when='midnight',
            interval=1,
            backupCount=30,  # Keep 30 days of logs
            encoding='utf-8'
        )
        time_handler.setLevel(logging.INFO)
        time_handler.setFormatter(file_formatter)
        logger.addHandler(time_handler)
    
    return logger

def setup_trade_logger() -> logging.Logger:
    """Setup specialized logger for trade execution"""
    trade_logger = logging.getLogger("trades")
    trade_logger.setLevel(logging.INFO)
    
    # Trade log file with JSON formatting
    trade_log_file = LOGS_DIR / "trades.jsonl"
    trade_handler = logging.handlers.RotatingFileHandler(
        trade_log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=20,
        encoding='utf-8'
    )
    
    # Custom formatter for trades
    trade_formatter = CustomJsonFormatter(
        '%(timestamp)s %(symbol)s %(action)s %(quantity)s %(price)s %(status)s'
    )
    trade_handler.setFormatter(trade_formatter)
    trade_logger.addHandler(trade_handler)
    
    return trade_logger

def setup_performance_logger() -> logging.Logger:
    """Setup logger for performance metrics"""
    perf_logger = logging.getLogger("performance")
    perf_logger.setLevel(logging.INFO)
    
    # Performance log file
    perf_log_file = LOGS_DIR / "performance.jsonl"
    perf_handler = logging.handlers.RotatingFileHandler(
        perf_log_file,
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=10,
        encoding='utf-8'
    )
    
    perf_formatter = CustomJsonFormatter()
    perf_handler.setFormatter(perf_formatter)
    perf_logger.addHandler(perf_handler)
    
    return perf_logger

def log_trade(symbol: str, action: str, quantity: int, price: float, 
             status: str, metadata: Dict = None) -> None:
    """
    Log trade execution
    
    Args:
        symbol: Stock symbol
        action: Buy/Sell
        quantity: Number of shares
        price: Execution price
        status: Trade status
        metadata: Additional trade metadata
    """
    trade_logger = logging.getLogger("trades")
    
    trade_info = {
        'symbol': symbol,
        'action': action,
        'quantity': quantity,
        'price': price,
        'status': status,
        'value': quantity * price,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    if metadata:
        trade_info.update(metadata)
    
    trade_logger.info("Trade executed", extra=trade_info)

def log_performance(metrics: Dict[str, Any]) -> None:
    """
    Log performance metrics
    
    Args:
        metrics: Dictionary of performance metrics
    """
    perf_logger = logging.getLogger("performance")
    
    metrics['timestamp'] = datetime.utcnow().isoformat()
    perf_logger.info("Performance update", extra=metrics)

def get_log_statistics() -> Dict[str, Any]:
    """Get logging statistics"""
    stats = {}
    
    for log_file in LOGS_DIR.glob("*.log*"):
        if log_file.is_file():
            stats[log_file.name] = {
                'size_mb': log_file.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
            }
    
    return {
        'log_directory': str(LOGS_DIR),
        'total_size_mb': sum(f['size_mb'] for f in stats.values()),
        'log_files': stats
    }

def cleanup_old_logs(days: int = 30) -> int:
    """
    Clean up log files older than specified days
    
    Args:
        days: Number of days to keep logs
        
    Returns:
        Number of files deleted
    """
    deleted = 0
    cutoff_time = datetime.now().timestamp() - (days * 86400)
    
    for log_file in LOGS_DIR.glob("*.log.*"):
        if log_file.stat().st_mtime < cutoff_time:
            log_file.unlink()
            deleted += 1
    
    return deleted

# Initialize default loggers
def initialize_loggers():
    """Initialize all application loggers"""
    # Main application logger
    app_logger = setup_logging(
        app_name="quantnexus",
        log_level="INFO",
        enable_console=True,
        enable_file=True,
        enable_json=True,
        context={
            'environment': os.getenv('ENVIRONMENT', 'production'),
            'version': '1.0.0'
        }
    )
    
    # Trade logger
    trade_logger = setup_trade_logger()
    
    # Performance logger
    perf_logger = setup_performance_logger()
    
    # Set up exception handling
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        app_logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = handle_exception
    
    app_logger.info("Logging system initialized")
    
    return app_logger, trade_logger, perf_logger

# Initialize on module import
if __name__ != "__main__":
    app_logger, trade_logger, perf_logger = initialize_loggers()