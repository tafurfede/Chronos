#!/usr/bin/env python3
"""
Production Readiness Checklist
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import importlib.util

load_dotenv()

class ProductionReadinessChecker:
    def __init__(self):
        self.checks = []
        self.critical_failures = []
        self.warnings = []
        
    def check_file_exists(self, filepath, description, critical=True):
        """Check if a file exists"""
        exists = Path(filepath).exists()
        status = "✅" if exists else ("❌" if critical else "⚠️")
        
        self.checks.append({
            'component': description,
            'status': status,
            'critical': critical
        })
        
        if not exists and critical:
            self.critical_failures.append(description)
        elif not exists:
            self.warnings.append(description)
            
        return exists
    
    def check_module_import(self, module_name, description, critical=True):
        """Check if a module can be imported"""
        try:
            importlib.import_module(module_name)
            status = "✅"
            success = True
        except ImportError:
            status = "❌" if critical else "⚠️"
            success = False
            
            if critical:
                self.critical_failures.append(description)
            else:
                self.warnings.append(description)
        
        self.checks.append({
            'component': description,
            'status': status,
            'critical': critical
        })
        
        return success
    
    def check_env_var(self, var_name, description, critical=True):
        """Check if environment variable is set"""
        value = os.getenv(var_name)
        exists = value is not None and value != ""
        status = "✅" if exists else ("❌" if critical else "⚠️")
        
        self.checks.append({
            'component': description,
            'status': status,
            'critical': critical
        })
        
        if not exists and critical:
            self.critical_failures.append(description)
        elif not exists:
            self.warnings.append(description)
            
        return exists
    
    def run_checks(self):
        """Run all production readiness checks"""
        
        print("🔍 PRODUCTION READINESS CHECK")
        print("="*60)
        
        # Critical Components
        print("\n📌 CRITICAL COMPONENTS:")
        print("-"*40)
        
        # Environment Variables
        self.check_env_var('ALPACA_API_KEY', 'Alpaca API Key')
        self.check_env_var('ALPACA_SECRET_KEY', 'Alpaca Secret Key')
        
        # Core Files
        self.check_file_exists('ml_trading_system.py', 'Main Trading System')
        self.check_file_exists('launch_ml_trading.py', 'Launch Script')
        self.check_file_exists('.env', 'Environment Configuration')
        
        # ML Models
        self.check_file_exists('data/models/best_nn_model.h5', 'Neural Network Model')
        self.check_file_exists('data/models/best_lstm_model.h5', 'LSTM Model')
        self.check_file_exists('data/models/scaler.pkl', 'Data Scaler')
        
        # Risk Management
        self.check_file_exists('src/ml_trading/risk/risk_manager.py', 'Risk Manager')
        
        # Strategies
        self.check_file_exists('src/ml_trading/strategies/base_strategy.py', 'Trading Strategies')
        
        # Core Modules
        print("\n📦 MODULE IMPORTS:")
        print("-"*40)
        
        self.check_module_import('alpaca.trading.client', 'Alpaca Trading Library')
        self.check_module_import('yfinance', 'Market Data (yfinance)')
        self.check_module_import('pandas', 'Pandas')
        self.check_module_import('numpy', 'NumPy')
        self.check_module_import('xgboost', 'XGBoost')
        self.check_module_import('lightgbm', 'LightGBM')
        self.check_module_import('tensorflow', 'TensorFlow')
        
        # Optional Components
        print("\n📋 OPTIONAL COMPONENTS:")
        print("-"*40)
        
        self.check_env_var('OPENAI_API_KEY', 'OpenAI API', critical=False)
        self.check_env_var('NEWS_API_KEY', 'News API', critical=False)
        self.check_file_exists('logs/', 'Logs Directory', critical=False)
        self.check_file_exists('reports/', 'Reports Directory', critical=False)
        
        # Trading Scripts
        print("\n🤖 TRADING SCRIPTS:")
        print("-"*40)
        
        self.check_file_exists('start_trading.py', 'Auto Trading Script')
        self.check_file_exists('demo_trading.py', 'Demo Analysis Script')
        self.check_file_exists('quick_trade.py', 'Quick Trade Script')
        self.check_file_exists('monitor_dashboard.py', 'Monitoring Dashboard')
        
        # Print Summary
        print("\n" + "="*60)
        print("📊 SUMMARY:")
        print("-"*40)
        
        total_checks = len(self.checks)
        passed = len([c for c in self.checks if c['status'] == '✅'])
        critical_passed = len([c for c in self.checks if c['critical'] and c['status'] == '✅'])
        critical_total = len([c for c in self.checks if c['critical']])
        
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed}/{total_checks}")
        print(f"Critical: {critical_passed}/{critical_total}")
        
        # Results
        print("\n📋 DETAILED RESULTS:")
        print("-"*40)
        
        for check in self.checks:
            print(f"{check['status']} {check['component']}")
        
        # Final Verdict
        print("\n" + "="*60)
        print("🎯 PRODUCTION READINESS:")
        print("-"*40)
        
        if not self.critical_failures:
            print("✅ READY FOR PRODUCTION TRADING!")
            print("\nYou can start trading with:")
            print("  1. make trade (full system)")
            print("  2. venv/bin/python3 start_trading.py (simple bot)")
            print("  3. venv/bin/python3 monitor_dashboard.py (monitoring)")
            ready = True
        else:
            print("❌ NOT READY - Critical components missing:")
            for failure in self.critical_failures:
                print(f"  - {failure}")
            print("\nFix these issues before trading!")
            ready = False
        
        if self.warnings:
            print("\n⚠️  WARNINGS (non-critical):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        print("\n" + "="*60)
        
        return ready

def main():
    checker = ProductionReadinessChecker()
    is_ready = checker.run_checks()
    
    if is_ready:
        print("\n🚀 Your system is production ready!")
        print("The market opens at 9:30 AM ET.")
        sys.exit(0)
    else:
        print("\n⚠️  Please fix critical issues before trading.")
        sys.exit(1)

if __name__ == "__main__":
    main()