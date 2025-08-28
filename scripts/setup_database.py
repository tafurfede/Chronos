#!/usr/bin/env python3
"""
Database Setup Script for ML Trading System
Creates tables, indexes, and initial configuration
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_database():
    """Create the database if it doesn't exist"""
    
    # Load environment variables
    load_dotenv('config/.env')
    
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'ml_trading')
    db_user = os.getenv('DB_USER', 'trader')
    db_password = os.getenv('DB_PASSWORD', 'password')
    
    try:
        # Connect to PostgreSQL server
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            user='postgres',
            password=db_password
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_name,)
        )
        exists = cursor.fetchone()
        
        if not exists:
            # Create database
            cursor.execute(f"CREATE DATABASE {db_name}")
            logger.info(f"Database '{db_name}' created successfully")
        else:
            logger.info(f"Database '{db_name}' already exists")
        
        # Create user if not exists
        cursor.execute(
            "SELECT 1 FROM pg_user WHERE usename = %s",
            (db_user,)
        )
        user_exists = cursor.fetchone()
        
        if not user_exists:
            cursor.execute(f"CREATE USER {db_user} WITH PASSWORD '{db_password}'")
            logger.info(f"User '{db_user}' created successfully")
        
        # Grant privileges
        cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {db_user}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False


def setup_tables():
    """Create all necessary tables"""
    
    # Load environment variables
    load_dotenv('config/.env')
    
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'ml_trading')
    db_user = os.getenv('DB_USER', 'trader')
    db_password = os.getenv('DB_PASSWORD', 'password')
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        cursor = conn.cursor()
        
        # Read and execute SQL script
        sql_file = os.path.join(os.path.dirname(__file__), 'init_db.sql')
        
        if os.path.exists(sql_file):
            with open(sql_file, 'r') as f:
                sql_script = f.read()
            
            # Execute the SQL script
            cursor.execute(sql_script)
            conn.commit()
            logger.info("Database tables created successfully")
        else:
            logger.warning(f"SQL script not found at {sql_file}")
            
            # Create basic tables if script not found
            create_basic_tables(cursor)
            conn.commit()
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up tables: {e}")
        return False


def create_basic_tables(cursor):
    """Create basic tables if SQL script is not available"""
    
    # Trades table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            side VARCHAR(10) NOT NULL,
            quantity INTEGER NOT NULL,
            entry_price DECIMAL(10, 2) NOT NULL,
            exit_price DECIMAL(10, 2),
            entry_time TIMESTAMP NOT NULL DEFAULT NOW(),
            exit_time TIMESTAMP,
            pnl DECIMAL(10, 2),
            status VARCHAR(20) NOT NULL DEFAULT 'open',
            confidence DECIMAL(3, 2),
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)
    
    # Signals table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
            signal_type VARCHAR(20) NOT NULL,
            confidence DECIMAL(3, 2) NOT NULL,
            expected_return DECIMAL(5, 2),
            price DECIMAL(10, 2) NOT NULL,
            executed BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)
    
    # Portfolio history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_history (
            timestamp TIMESTAMP NOT NULL PRIMARY KEY,
            total_value DECIMAL(12, 2) NOT NULL,
            cash_balance DECIMAL(12, 2) NOT NULL,
            positions_value DECIMAL(12, 2) NOT NULL,
            daily_pnl DECIMAL(10, 2),
            daily_return DECIMAL(5, 4)
        )
    """)
    
    # Model performance table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(50) NOT NULL,
            timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
            accuracy DECIMAL(5, 4),
            sharpe_ratio DECIMAL(5, 2),
            max_drawdown DECIMAL(5, 4)
        )
    """)
    
    logger.info("Basic tables created successfully")


def insert_test_data():
    """Insert some test data for development"""
    
    # Load environment variables
    load_dotenv('config/.env')
    
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'ml_trading')
    db_user = os.getenv('DB_USER', 'trader')
    db_password = os.getenv('DB_PASSWORD', 'password')
    
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        cursor = conn.cursor()
        
        # Insert test trades
        test_trades = [
            ('AAPL', 'buy', 100, 150.50, 152.75, 225.00, 'closed', 0.72),
            ('MSFT', 'buy', 50, 300.25, 298.50, -87.50, 'closed', 0.65),
            ('GOOGL', 'buy', 25, 2500.00, None, None, 'open', 0.68),
        ]
        
        for trade in test_trades:
            cursor.execute("""
                INSERT INTO trades (symbol, side, quantity, entry_price, exit_price, pnl, status, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, trade)
        
        # Insert test signals
        test_signals = [
            ('AAPL', 'buy', 0.72, 0.025, 151.00),
            ('TSLA', 'sell', 0.68, 0.015, 250.00),
            ('NVDA', 'buy', 0.81, 0.035, 450.00),
        ]
        
        for signal in test_signals:
            cursor.execute("""
                INSERT INTO signals (symbol, signal_type, confidence, expected_return, price)
                VALUES (%s, %s, %s, %s, %s)
            """, signal)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("Test data inserted successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error inserting test data: {e}")
        return False


def verify_setup():
    """Verify that the database is set up correctly"""
    
    # Load environment variables
    load_dotenv('config/.env')
    
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'ml_trading')
    db_user = os.getenv('DB_USER', 'trader')
    db_password = os.getenv('DB_PASSWORD', 'password')
    
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        
        logger.info("Database tables:")
        for table in tables:
            logger.info(f"  - {table[0]}")
        
        # Check row counts
        for table in ['trades', 'signals', 'portfolio_history']:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"  {table}: {count} rows")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error verifying setup: {e}")
        return False


def main():
    """Main setup function"""
    
    logger.info("Starting database setup...")
    
    # Create database
    if not create_database():
        logger.error("Failed to create database")
        sys.exit(1)
    
    # Setup tables
    if not setup_tables():
        logger.error("Failed to setup tables")
        sys.exit(1)
    
    # Insert test data (optional)
    response = input("Insert test data? (y/n): ")
    if response.lower() == 'y':
        insert_test_data()
    
    # Verify setup
    if verify_setup():
        logger.info("Database setup completed successfully!")
    else:
        logger.error("Database verification failed")
        sys.exit(1)


if __name__ == "__main__":
    main()