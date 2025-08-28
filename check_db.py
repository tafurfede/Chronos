#!/usr/bin/env python3
"""Check database contents"""

import sys
import os

# Set database credentials
os.environ['DB_HOST'] = 'localhost'
os.environ['DB_PORT'] = '5432'
os.environ['DB_NAME'] = 'quantnexus_trading'
os.environ['DB_USER'] = 'quantnexus_app'
os.environ['DB_PASSWORD'] = 'TradingApp2024!'

sys.path.append('.')

from src.ml_trading.database.connection import get_db_manager
from src.ml_trading.database.models import Trade, Position, Signal

db = get_db_manager()

with db.get_session() as session:
    trades_count = session.query(Trade).count()
    positions_count = session.query(Position).count()
    signals_count = session.query(Signal).count()
    
    print('📊 Database Statistics:')
    print(f'  Trades: {trades_count}')
    print(f'  Positions: {positions_count}')
    print(f'  Signals: {signals_count}')
    
    # Get recent trades
    recent_trades = session.query(Trade).order_by(Trade.created_at.desc()).limit(5).all()
    if recent_trades:
        print('\n📈 Recent Trades:')
        for trade in recent_trades:
            print(f'  {trade.symbol}: {trade.side.value if trade.side else "N/A"} {trade.quantity} @ ${trade.price}')
    else:
        print('\n  No trades recorded yet (database is ready for trading)')

print('\n✅ Database is connected and operational!')