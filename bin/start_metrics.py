#!/usr/bin/env python3
"""
Start Prometheus metrics server for QuantNexus
"""

import sys
sys.path.append('.')

from src.ml_trading.monitoring.metrics import start_metrics_server
import time

def main():
    print("Starting QuantNexus Metrics Server...")
    print("Prometheus metrics available at http://localhost:8000/metrics")
    print("Press Ctrl+C to stop")
    
    try:
        server = start_metrics_server(port=8000)
        
        # Keep running
        while True:
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nMetrics server stopped")
        
if __name__ == "__main__":
    main()