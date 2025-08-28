"""
Advanced Error Handler with Retry Logic and Dead Letter Queue
"""

import time
import logging
import traceback
from typing import Any, Callable, Dict, Optional, List
from functools import wraps
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import asyncio
import json

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class RetryStrategy(Enum):
    """Retry strategies"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
    CONSTANT = "constant"

class ErrorHandler:
    """Centralized error handling with retry logic"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.error_history = deque(maxlen=1000)
        self.dead_letter_queue = deque(maxlen=100)
        self.error_counts = {}
        self.circuit_breakers = {}
        
    def calculate_delay(self, attempt: int, strategy: RetryStrategy = RetryStrategy.EXPONENTIAL) -> float:
        """Calculate delay based on retry strategy"""
        if strategy == RetryStrategy.EXPONENTIAL:
            return self.base_delay * (2 ** attempt)
        elif strategy == RetryStrategy.LINEAR:
            return self.base_delay * (attempt + 1)
        elif strategy == RetryStrategy.FIBONACCI:
            if attempt <= 1:
                return self.base_delay
            fib = [1, 1]
            for _ in range(attempt - 1):
                fib.append(fib[-1] + fib[-2])
            return self.base_delay * fib[-1]
        else:  # CONSTANT
            return self.base_delay
    
    def retry(self, 
             exceptions: tuple = (Exception,),
             max_retries: int = None,
             strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
             on_retry: Callable = None,
             on_failure: Callable = None):
        """
        Decorator for automatic retry with exponential backoff
        
        Args:
            exceptions: Tuple of exceptions to catch
            max_retries: Maximum number of retry attempts
            strategy: Retry strategy to use
            on_retry: Callback function called on each retry
            on_failure: Callback function called when all retries fail
        """
        max_retries = max_retries or self.max_retries
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt >= max_retries:
                            # All retries exhausted
                            self.log_error(func.__name__, e, severity=ErrorSeverity.HIGH)
                            self.add_to_dead_letter(func.__name__, args, kwargs, e)
                            
                            if on_failure:
                                on_failure(e, attempt)
                            
                            raise
                        
                        # Calculate retry delay
                        delay = self.calculate_delay(attempt, strategy)
                        
                        # Log retry attempt
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {delay:.2f}s delay. Error: {str(e)}"
                        )
                        
                        if on_retry:
                            on_retry(e, attempt)
                        
                        time.sleep(delay)
                
                if last_exception:
                    raise last_exception
                    
            return wrapper
        return decorator
    
    def async_retry(self,
                   exceptions: tuple = (Exception,),
                   max_retries: int = None,
                   strategy: RetryStrategy = RetryStrategy.EXPONENTIAL):
        """Async version of retry decorator"""
        max_retries = max_retries or self.max_retries
        
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt >= max_retries:
                            self.log_error(func.__name__, e, severity=ErrorSeverity.HIGH)
                            self.add_to_dead_letter(func.__name__, args, kwargs, e)
                            raise
                        
                        delay = self.calculate_delay(attempt, strategy)
                        logger.warning(
                            f"Async retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {delay:.2f}s delay. Error: {str(e)}"
                        )
                        
                        await asyncio.sleep(delay)
                
                if last_exception:
                    raise last_exception
                    
            return wrapper
        return decorator
    
    def circuit_breaker(self,
                       failure_threshold: int = 5,
                       recovery_timeout: int = 60,
                       expected_exception: type = Exception):
        """
        Circuit breaker pattern implementation
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting to close circuit
            expected_exception: Exception type to track
        """
        def decorator(func):
            circuit_state = {
                'failures': 0,
                'last_failure': None,
                'state': 'closed'  # closed, open, half_open
            }
            self.circuit_breakers[func.__name__] = circuit_state
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                state = self.circuit_breakers[func.__name__]
                
                # Check if circuit is open
                if state['state'] == 'open':
                    if datetime.now() - state['last_failure'] > timedelta(seconds=recovery_timeout):
                        state['state'] = 'half_open'
                        logger.info(f"Circuit breaker for {func.__name__} entering half-open state")
                    else:
                        raise Exception(f"Circuit breaker OPEN for {func.__name__}")
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Success - reset failures
                    if state['state'] == 'half_open':
                        state['state'] = 'closed'
                        state['failures'] = 0
                        logger.info(f"Circuit breaker for {func.__name__} CLOSED")
                    
                    return result
                    
                except expected_exception as e:
                    state['failures'] += 1
                    state['last_failure'] = datetime.now()
                    
                    if state['failures'] >= failure_threshold:
                        state['state'] = 'open'
                        logger.error(f"Circuit breaker for {func.__name__} OPEN after {state['failures']} failures")
                    
                    raise
                    
            return wrapper
        return decorator
    
    def log_error(self, context: str, error: Exception, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        """Log error with context and severity"""
        error_info = {
            'timestamp': datetime.now(),
            'context': context,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'severity': severity.name,
            'stack_trace': traceback.format_exc()
        }
        
        self.error_history.append(error_info)
        
        # Track error counts
        error_key = f"{context}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"[{context}] {error}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"[{context}] {error}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"[{context}] {error}")
        else:
            logger.info(f"[{context}] {error}")
    
    def add_to_dead_letter(self, func_name: str, args: tuple, kwargs: dict, error: Exception):
        """Add failed operation to dead letter queue for later processing"""
        dead_letter = {
            'timestamp': datetime.now(),
            'function': func_name,
            'args': str(args)[:500],  # Truncate to avoid memory issues
            'kwargs': str(kwargs)[:500],
            'error': str(error),
            'error_type': type(error).__name__
        }
        self.dead_letter_queue.append(dead_letter)
        logger.error(f"Added to dead letter queue: {func_name}")
    
    def get_error_statistics(self) -> Dict:
        """Get error statistics"""
        recent_errors = list(self.error_history)[-100:]
        
        # Group by error type
        error_types = {}
        for error in recent_errors:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Group by severity
        severity_counts = {s.name: 0 for s in ErrorSeverity}
        for error in recent_errors:
            severity = error.get('severity', 'MEDIUM')
            severity_counts[severity] += 1
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors': len(recent_errors),
            'error_types': error_types,
            'severity_distribution': severity_counts,
            'top_errors': sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'dead_letter_count': len(self.dead_letter_queue),
            'circuit_breakers': {
                name: state['state'] 
                for name, state in self.circuit_breakers.items()
            }
        }
    
    def process_dead_letter_queue(self, processor: Callable) -> int:
        """
        Process items in dead letter queue
        
        Args:
            processor: Function to process dead letter items
            
        Returns:
            Number of items processed
        """
        processed = 0
        while self.dead_letter_queue:
            item = self.dead_letter_queue.popleft()
            try:
                processor(item)
                processed += 1
            except Exception as e:
                logger.error(f"Failed to process dead letter item: {e}")
                # Re-add to queue if processing fails
                self.dead_letter_queue.append(item)
                break
        
        return processed


# Global error handler instance
error_handler = ErrorHandler()

# Convenience decorators
retry = error_handler.retry
async_retry = error_handler.async_retry
circuit_breaker = error_handler.circuit_breaker