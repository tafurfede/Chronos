#!/usr/bin/env python3
"""
Setup QuantNexus tables in Supabase cloud database
"""

import os
import sys
from dotenv import load_dotenv

# Load cloud database configuration
load_dotenv('.env.cloud')

# Check if DATABASE_URL is set
if not os.getenv('DATABASE_URL') or '[YOUR-PASSWORD]' in os.getenv('DATABASE_URL', ''):
    print("❌ ERROR: Please update .env.cloud with your Supabase connection string!")
    print("\n1. Go to Supabase Dashboard")
    print("2. Settings → Database → Connection string")
    print("3. Copy the URI and paste it in .env.cloud")
    print("\nExample:")
    print("DATABASE_URL=postgresql://postgres:YourPassword@db.xxxxx.supabase.co:5432/postgres")
    sys.exit(1)

sys.path.append('.')

from src.ml_trading.database.connection import get_db_manager
from src.ml_trading.database.models import Base

print("🌐 Connecting to Supabase cloud database...")
print(f"Database URL: {os.getenv('DATABASE_URL')[:50]}...")

try:
    # Initialize database manager
    db = get_db_manager()
    
    # Test connection
    if db.health_check():
        print("✅ Connected to Supabase successfully!")
    else:
        print("❌ Failed to connect. Check your connection string.")
        sys.exit(1)
    
    # Create all tables
    print("\n📊 Creating tables in cloud database...")
    Base.metadata.create_all(bind=db.engine)
    
    print("\n✅ Tables created successfully:")
    print("  • trades")
    print("  • positions")
    print("  • signals")
    print("  • portfolio_history")
    print("  • market_data")
    print("  • risk_metrics")
    print("  • model_performance")
    print("  • error_logs")
    
    # Test write operation
    print("\n🧪 Testing database write...")
    from src.ml_trading.database.models import Trade, OrderSide, OrderStatus
    from datetime import datetime
    
    with db.get_session() as session:
        test_trade = Trade(
            symbol='CLOUD_TEST',
            side=OrderSide.BUY,
            quantity=1,
            price=100.0,
            order_type='market',
            status=OrderStatus.FILLED,
            strategy_name='cloud_verification',
            created_at=datetime.now()
        )
        session.add(test_trade)
        session.commit()
        
        # Read it back
        trade = session.query(Trade).filter_by(symbol='CLOUD_TEST').first()
        if trade:
            print(f"✅ Test trade created: {trade.symbol} @ ${trade.price}")
            # Clean up
            session.delete(trade)
            session.commit()
        
    print("\n🎉 Cloud database setup complete!")
    print("\nYour Supabase database is ready for trading!")
    print("\nTo use the cloud database, run:")
    print("  source .env.cloud")
    print("  python3 start_trading.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check your internet connection")
    print("2. Verify your Supabase connection string in .env.cloud")
    print("3. Make sure your Supabase project is active")
    sys.exit(1)