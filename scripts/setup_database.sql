-- QuantNexus ML Trading System Database Schema
-- PostgreSQL with TimescaleDB extension

-- Create database
CREATE DATABASE IF NOT EXISTS quantnexus_trading;
\c quantnexus_trading;

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ==========================================
-- TRADES TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(15, 4) NOT NULL,
    price DECIMAL(15, 4) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    filled_qty DECIMAL(15, 4) DEFAULT 0,
    filled_avg_price DECIMAL(15, 4),
    commission DECIMAL(10, 4) DEFAULT 0,
    strategy_name VARCHAR(50),
    signal_strength DECIMAL(5, 4),
    stop_loss DECIMAL(15, 4),
    take_profit DECIMAL(15, 4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    executed_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    pnl DECIMAL(15, 4),
    pnl_percentage DECIMAL(10, 4),
    metadata JSONB,
    PRIMARY KEY (id, created_at)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('trades', 'created_at', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX idx_trades_symbol_time ON trades (symbol, created_at DESC);
CREATE INDEX idx_trades_status ON trades (status);
CREATE INDEX idx_trades_strategy ON trades (strategy_name);

-- ==========================================
-- MARKET_DATA TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS market_data (
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(15, 4) NOT NULL,
    high DECIMAL(15, 4) NOT NULL,
    low DECIMAL(15, 4) NOT NULL,
    close DECIMAL(15, 4) NOT NULL,
    volume BIGINT NOT NULL,
    vwap DECIMAL(15, 4),
    bid DECIMAL(15, 4),
    ask DECIMAL(15, 4),
    spread DECIMAL(15, 4),
    PRIMARY KEY (symbol, timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);

-- Create continuous aggregate for 5-minute candles
CREATE MATERIALIZED VIEW market_data_5min
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('5 minutes', timestamp) AS bucket,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    avg(vwap) AS vwap
FROM market_data
GROUP BY symbol, bucket
WITH NO DATA;

-- ==========================================
-- SIGNALS TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(10) NOT NULL,
    signal_type VARCHAR(20) NOT NULL CHECK (signal_type IN ('buy', 'sell', 'hold')),
    strength DECIMAL(5, 4) NOT NULL,
    strategy_name VARCHAR(50) NOT NULL,
    model_name VARCHAR(50),
    confidence DECIMAL(5, 4),
    features JSONB,
    indicators JSONB,
    metadata JSONB,
    executed BOOLEAN DEFAULT FALSE,
    trade_id INTEGER,
    PRIMARY KEY (id, timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('signals', 'timestamp', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX idx_signals_symbol_time ON signals (symbol, timestamp DESC);
CREATE INDEX idx_signals_executed ON signals (executed);
CREATE INDEX idx_signals_type ON signals (signal_type);

-- ==========================================
-- PORTFOLIO_HISTORY TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS portfolio_history (
    timestamp TIMESTAMPTZ NOT NULL,
    total_value DECIMAL(20, 4) NOT NULL,
    cash_balance DECIMAL(20, 4) NOT NULL,
    positions_value DECIMAL(20, 4) NOT NULL,
    daily_pnl DECIMAL(15, 4),
    daily_return DECIMAL(10, 6),
    cumulative_return DECIMAL(10, 6),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 6),
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 4),
    metadata JSONB,
    PRIMARY KEY (timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('portfolio_history', 'timestamp', if_not_exists => TRUE);

-- ==========================================
-- POSITIONS TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    quantity DECIMAL(15, 4) NOT NULL,
    avg_entry_price DECIMAL(15, 4) NOT NULL,
    current_price DECIMAL(15, 4),
    market_value DECIMAL(20, 4),
    unrealized_pnl DECIMAL(15, 4),
    unrealized_pnl_pct DECIMAL(10, 4),
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    strategy_name VARCHAR(50),
    metadata JSONB,
    UNIQUE(symbol)
);

-- ==========================================
-- RISK_METRICS TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS risk_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    var_95 DECIMAL(15, 4),
    var_99 DECIMAL(15, 4),
    cvar_95 DECIMAL(15, 4),
    beta DECIMAL(10, 4),
    correlation_spy DECIMAL(10, 4),
    volatility_daily DECIMAL(10, 6),
    volatility_annual DECIMAL(10, 6),
    leverage DECIMAL(10, 4),
    concentration_top5 DECIMAL(10, 4),
    metadata JSONB,
    PRIMARY KEY (timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('risk_metrics', 'timestamp', if_not_exists => TRUE);

-- ==========================================
-- MODEL_PERFORMANCE TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    accuracy DECIMAL(10, 6),
    precision_score DECIMAL(10, 6),
    recall_score DECIMAL(10, 6),
    f1_score DECIMAL(10, 6),
    sharpe_ratio DECIMAL(10, 4),
    total_predictions INTEGER,
    correct_predictions INTEGER,
    training_date TIMESTAMPTZ,
    features_used TEXT[],
    hyperparameters JSONB,
    metadata JSONB,
    PRIMARY KEY (id, timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('model_performance', 'timestamp', if_not_exists => TRUE);

-- ==========================================
-- ERROR_LOGS TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS error_logs (
    id SERIAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error_type VARCHAR(100) NOT NULL,
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    component VARCHAR(100),
    severity VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    resolved BOOLEAN DEFAULT FALSE,
    resolution_notes TEXT,
    metadata JSONB,
    PRIMARY KEY (id, timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('error_logs', 'timestamp', if_not_exists => TRUE);

-- ==========================================
-- DATA RETENTION POLICIES
-- ==========================================

-- Keep detailed market data for 6 months, then downsample
SELECT add_retention_policy('market_data', INTERVAL '6 months', if_not_exists => TRUE);

-- Keep trades forever (they're important for tax/audit)
-- No retention policy for trades table

-- Keep signals for 1 year
SELECT add_retention_policy('signals', INTERVAL '1 year', if_not_exists => TRUE);

-- Keep portfolio history forever (daily snapshots)
-- No retention policy

-- Keep detailed risk metrics for 3 months
SELECT add_retention_policy('risk_metrics', INTERVAL '3 months', if_not_exists => TRUE);

-- Keep error logs for 3 months
SELECT add_retention_policy('error_logs', INTERVAL '3 months', if_not_exists => TRUE);

-- ==========================================
-- CONTINUOUS AGGREGATES REFRESH POLICIES
-- ==========================================

-- Refresh 5-minute market data every 5 minutes
SELECT add_continuous_aggregate_policy('market_data_5min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE);

-- ==========================================
-- FUNCTIONS AND TRIGGERS
-- ==========================================

-- Function to update position timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for positions table
CREATE TRIGGER update_positions_updated_at
    BEFORE UPDATE ON positions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Function to calculate portfolio metrics
CREATE OR REPLACE FUNCTION calculate_portfolio_metrics()
RETURNS TABLE (
    total_value DECIMAL,
    total_pnl DECIMAL,
    win_rate DECIMAL,
    sharpe_ratio DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        SUM(p.market_value) AS total_value,
        SUM(p.unrealized_pnl) AS total_pnl,
        CASE 
            WHEN COUNT(*) > 0 THEN 
                SUM(CASE WHEN p.unrealized_pnl > 0 THEN 1 ELSE 0 END)::DECIMAL / COUNT(*)
            ELSE 0
        END AS win_rate,
        CASE 
            WHEN STDDEV(p.unrealized_pnl_pct) > 0 THEN
                AVG(p.unrealized_pnl_pct) / STDDEV(p.unrealized_pnl_pct)
            ELSE 0
        END AS sharpe_ratio
    FROM positions p;
END;
$$ LANGUAGE plpgsql;

-- ==========================================
-- INITIAL DATA AND PERMISSIONS
-- ==========================================

-- Create read-only user for reporting
CREATE USER quantnexus_reader WITH PASSWORD 'ReadOnly2024!';
GRANT CONNECT ON DATABASE quantnexus_trading TO quantnexus_reader;
GRANT USAGE ON SCHEMA public TO quantnexus_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO quantnexus_reader;

-- Create application user with full permissions
CREATE USER quantnexus_app WITH PASSWORD 'TradingApp2024!';
GRANT CONNECT ON DATABASE quantnexus_trading TO quantnexus_app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO quantnexus_app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO quantnexus_app;

PRINT 'Database setup complete! TimescaleDB enabled with all tables and policies.';