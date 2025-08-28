"""
Database connection manager with connection pooling for QuantNexus
"""

import os
import logging
from typing import Optional, Generator, Any
from contextlib import contextmanager
from sqlalchemy import create_engine, pool, event
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, DBAPIError
from dotenv import load_dotenv
import time
from .models import Base

load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages database connections with connection pooling and retry logic
    """
    
    def __init__(self, 
                 db_url: Optional[str] = None,
                 pool_size: int = 20,
                 max_overflow: int = 40,
                 pool_timeout: int = 30,
                 pool_recycle: int = 3600):
        """
        Initialize database connection manager
        
        Args:
            db_url: Database URL (defaults to environment variable)
            pool_size: Number of persistent connections
            max_overflow: Maximum overflow connections above pool_size
            pool_timeout: Timeout for getting connection from pool
            pool_recycle: Recycle connections after this many seconds
        """
        self.db_url = db_url or self._get_db_url()
        
        # Create engine with connection pooling
        self.engine = create_engine(
            self.db_url,
            poolclass=pool.QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            echo=False,  # Set to True for SQL debugging
            connect_args={
                "connect_timeout": 10,
                "application_name": "quantnexus_trading",
                "options": "-c statement_timeout=30000"  # 30 second statement timeout
            }
        )
        
        # Create session factory
        self.SessionFactory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )
        
        # Create scoped session for thread safety
        self.Session = scoped_session(self.SessionFactory)
        
        # Setup event listeners
        self._setup_event_listeners()
        
        # Initialize database schema
        self._init_database()
        
        logger.info("Database connection manager initialized")
    
    def _get_db_url(self) -> str:
        """Construct database URL from environment variables"""
        # First check for DATABASE_URL (cloud database like Supabase)
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            # Replace postgres:// with postgresql:// for SQLAlchemy compatibility
            if database_url.startswith('postgres://'):
                database_url = database_url.replace('postgres://', 'postgresql://', 1)
            # Use psycopg3 driver
            if 'postgresql://' in database_url and '+psycopg' not in database_url:
                database_url = database_url.replace('postgresql://', 'postgresql+psycopg://', 1)
            logger.info("Using cloud database connection")
            return database_url
        
        # Otherwise use local database settings
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'quantnexus_trading')
        db_user = os.getenv('DB_USER', 'quantnexus_app')
        db_password = os.getenv('DB_PASSWORD', 'TradingApp2024!')
        
        logger.info(f"Using local database at {db_host}:{db_port}/{db_name}")
        return f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners for monitoring"""
        
        @event.listens_for(Engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Log new connections"""
            connection_record.info['pid'] = os.getpid()
            logger.debug(f"New database connection established (PID: {os.getpid()})")
        
        @event.listens_for(Engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Log connection checkouts from pool"""
            pid = os.getpid()
            if connection_record.info['pid'] != pid:
                connection_record.connection = connection_proxy.connection = None
                raise DBAPIError("Connection record belongs to different PID", None, None)
        
        @event.listens_for(Engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Reset connection state on checkin"""
            pass
    
    def _init_database(self):
        """Initialize database schema if needed"""
        try:
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions with automatic cleanup
        
        Yields:
            Session object
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error occurred: {e}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error: {e}")
            raise
        finally:
            session.close()
    
    def execute_with_retry(self, func, max_retries: int = 3, backoff_factor: float = 2.0):
        """
        Execute database operation with retry logic
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff factor
        
        Returns:
            Result of function execution
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                with self.get_session() as session:
                    return func(session)
            except (DBAPIError, SQLAlchemyError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries}), "
                                 f"retrying in {wait_time} seconds: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Database operation failed after {max_retries} attempts: {e}")
        
        if last_exception:
            raise last_exception
    
    def bulk_insert(self, model_class: Any, records: list):
        """
        Bulk insert records efficiently
        
        Args:
            model_class: SQLAlchemy model class
            records: List of dictionaries containing record data
        """
        if not records:
            return
        
        try:
            with self.get_session() as session:
                session.bulk_insert_mappings(model_class, records)
                logger.info(f"Bulk inserted {len(records)} {model_class.__name__} records")
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        Check database connection health
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_pool_status(self) -> dict:
        """
        Get connection pool status
        
        Returns:
            Dictionary with pool statistics
        """
        pool = self.engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total": pool.checkedin() + pool.checkedout()
        }
    
    def close(self):
        """Close all database connections"""
        try:
            self.Session.remove()
            self.engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Singleton instance
_db_manager: Optional[DatabaseManager] = None

def get_db_manager() -> DatabaseManager:
    """
    Get singleton database manager instance
    
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def close_db_manager():
    """Close the singleton database manager"""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None


# Convenience function for quick queries
@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Quick access to database session
    
    Yields:
        Database session
    """
    db_manager = get_db_manager()
    with db_manager.get_session() as session:
        yield session