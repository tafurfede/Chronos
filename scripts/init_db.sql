-- ML Trading System Database Schema
-- PostgreSQL with TimescaleDB extensions

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create schema
CREATE SCHEMA IF NOT EXISTS trading;

-- Set default schema
SET search_path TO trading, public;

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    trade_id UUID DEFAULT gen_random_uuid() UNIQUE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell', 'long', 'short')),
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(10, 2) NOT NULL,
    exit_price DECIMAL(10, 2),
    entry_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    exit_time TIMESTAMPTZ,
    stop_loss DECIMAL(10, 2),
    take_profit DECIMAL(10, 2),
    pnl DECIMAL(10, 2),
    pnl_pct DECIMAL(5, 2),
    commission DECIMAL(10, 2),
    slippage DECIMAL(10, 2),
    status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed', 'cancelled')),
    exit_reason VARCHAR(50),
    confidence DECIMAL(3, 2),
    expected_return DECIMAL(5, 2),
    model_version VARCHAR(20),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('trades', 'entry_time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_entry_time ON trades(entry_time DESC);
CREATE INDEX idx_trades_pnl ON trades(pnl);

-- Market data table
CREATE TABLE IF NOT EXISTS market_data (
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(10, 2) NOT NULL,
    high DECIMAL(10, 2) NOT NULL,
    low DECIMAL(10, 2) NOT NULL,
    close DECIMAL(10, 2) NOT NULL,
    volume BIGINT NOT NULL,
    vwap DECIMAL(10, 2),
    PRIMARY KEY (symbol, timestamp)
);

-- Convert to hypertable
SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);

-- Create compression policy for older data
SELECT add_compression_policy('market_data', INTERVAL '7 days');

-- Signals table
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    signal_id UUID DEFAULT gen_random_uuid() UNIQUE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    signal_type VARCHAR(20) NOT NULL CHECK (signal_type IN ('buy', 'sell', 'hold')),
    confidence DECIMAL(3, 2) NOT NULL,
    expected_return DECIMAL(5, 2),
    price DECIMAL(10, 2) NOT NULL,
    features JSONB,
    models_used TEXT[],
    executed BOOLEAN DEFAULT FALSE,
    trade_id UUID REFERENCES trades(trade_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('signals', 'timestamp', if_not_exists => TRUE);

-- Model performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metrics JSONB NOT NULL,
    training_date DATE,
    validation_score DECIMAL(5, 4),
    test_score DECIMAL(5, 4),
    feature_importance JSONB,
    hyperparameters JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Portfolio history table
CREATE TABLE IF NOT EXISTS portfolio_history (
    timestamp TIMESTAMPTZ NOT NULL,
    total_value DECIMAL(12, 2) NOT NULL,
    cash_balance DECIMAL(12, 2) NOT NULL,
    positions_value DECIMAL(12, 2) NOT NULL,
    daily_pnl DECIMAL(10, 2),
    daily_return DECIMAL(5, 4),
    cumulative_return DECIMAL(7, 4),
    sharpe_ratio DECIMAL(5, 2),
    max_drawdown DECIMAL(5, 4),
    win_rate DECIMAL(3, 2),
    positions_count INTEGER,
    PRIMARY KEY (timestamp)
);

-- Convert to hypertable
SELECT create_hypertable('portfolio_history', 'timestamp', if_not_exists => TRUE);

-- Risk metrics table
CREATE TABLE IF NOT EXISTS risk_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    var_95 DECIMAL(10, 2),
    var_99 DECIMAL(10, 2),
    cvar_95 DECIMAL(10, 2),
    beta DECIMAL(5, 3),
    correlation_spy DECIMAL(5, 3),
    volatility DECIMAL(5, 4),
    downside_deviation DECIMAL(5, 4),
    sortino_ratio DECIMAL(5, 2),
    calmar_ratio DECIMAL(5, 2),
    PRIMARY KEY (timestamp)
);

-- Convert to hypertable
SELECT create_hypertable('risk_metrics', 'timestamp', if_not_exists => TRUE);

-- Execution logs table
CREATE TABLE IF NOT EXISTS execution_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    level VARCHAR(20) NOT NULL,
    component VARCHAR(50),
    message TEXT,
    details JSONB,
    error_trace TEXT
);

-- Convert to hypertable
SELECT create_hypertable('execution_logs', 'timestamp', if_not_exists => TRUE);

-- Feature store table
CREATE TABLE IF NOT EXISTS feature_store (
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    features JSONB NOT NULL,
    feature_version VARCHAR(20),
    PRIMARY KEY (symbol, timestamp)
);

-- Convert to hypertable
SELECT create_hypertable('feature_store', 'timestamp', if_not_exists => TRUE);

-- Backtesting results table
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    backtest_id UUID DEFAULT gen_random_uuid() UNIQUE NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(12, 2) NOT NULL,
    final_capital DECIMAL(12, 2) NOT NULL,
    total_return DECIMAL(7, 4),
    sharpe_ratio DECIMAL(5, 2),
    max_drawdown DECIMAL(5, 4),
    win_rate DECIMAL(3, 2),
    profit_factor DECIMAL(5, 2),
    total_trades INTEGER,
    parameters JSONB,
    trades JSONB,
    equity_curve JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create materialized views for performance
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_performance AS
SELECT 
    DATE(entry_time) as trading_date,
    COUNT(*) as trades_count,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as win_rate,
    SUM(pnl) as total_pnl,
    AVG(pnl) as avg_pnl,
    MAX(pnl) as max_win,
    MIN(pnl) as max_loss,
    AVG(confidence) as avg_confidence
FROM trades
WHERE status = 'closed'
GROUP BY DATE(entry_time)
ORDER BY trading_date DESC;

-- Create refresh policy for materialized view
CREATE OR REPLACE FUNCTION refresh_daily_performance()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_performance;
END;
$$ LANGUAGE plpgsql;

-- Schedule refresh every hour
SELECT cron.schedule('refresh-daily-performance', '0 * * * *', 'SELECT refresh_daily_performance();');

-- Create functions for analytics
CREATE OR REPLACE FUNCTION calculate_sharpe_ratio(
    returns DECIMAL[], 
    risk_free_rate DECIMAL DEFAULT 0.02
)
RETURNS DECIMAL AS $$
DECLARE
    avg_return DECIMAL;
    std_dev DECIMAL;
BEGIN
    SELECT AVG(r), STDDEV(r) INTO avg_return, std_dev
    FROM UNNEST(returns) AS r;
    
    IF std_dev = 0 THEN
        RETURN 0;
    END IF;
    
    RETURN (avg_return - risk_free_rate/252) / std_dev * SQRT(252);
END;
$$ LANGUAGE plpgsql;

-- Function to calculate maximum drawdown
CREATE OR REPLACE FUNCTION calculate_max_drawdown(equity_curve DECIMAL[])
RETURNS DECIMAL AS $$
DECLARE
    max_equity DECIMAL := 0;
    max_dd DECIMAL := 0;
    current_dd DECIMAL;
    equity DECIMAL;
BEGIN
    FOREACH equity IN ARRAY equity_curve LOOP
        IF equity > max_equity THEN
            max_equity := equity;
        END IF;
        
        current_dd := (equity - max_equity) / max_equity;
        
        IF current_dd < max_dd THEN
            max_dd := current_dd;
        END IF;
    END LOOP;
    
    RETURN max_dd;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_trades_updated_at
    BEFORE UPDATE ON trades
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create performance summary function
CREATE OR REPLACE FUNCTION get_performance_summary(
    start_date TIMESTAMPTZ DEFAULT NOW() - INTERVAL '30 days',
    end_date TIMESTAMPTZ DEFAULT NOW()
)
RETURNS TABLE(
    total_trades BIGINT,
    win_rate DECIMAL,
    total_pnl DECIMAL,
    avg_pnl DECIMAL,
    sharpe_ratio DECIMAL,
    max_drawdown DECIMAL,
    best_trade DECIMAL,
    worst_trade DECIMAL,
    avg_confidence DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_trades,
        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) as win_rate,
        SUM(pnl) as total_pnl,
        AVG(pnl) as avg_pnl,
        calculate_sharpe_ratio(ARRAY_AGG(pnl ORDER BY entry_time)) as sharpe_ratio,
        calculate_max_drawdown(
            ARRAY_AGG(SUM(pnl) OVER (ORDER BY entry_time) ORDER BY entry_time)
        ) as max_drawdown,
        MAX(pnl) as best_trade,
        MIN(pnl) as worst_trade,
        AVG(confidence) as avg_confidence
    FROM trades
    WHERE entry_time BETWEEN start_date AND end_date
        AND status = 'closed';
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO trader;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA trading TO trader;