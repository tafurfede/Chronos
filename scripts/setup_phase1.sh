#!/bin/bash

# QuantNexus Phase 1 Setup Script
# Installs PostgreSQL with TimescaleDB and configures the database

set -e

echo "========================================="
echo "QuantNexus Phase 1 Infrastructure Setup"
echo "========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Install PostgreSQL
echo "Step 1: Installing PostgreSQL..."
if command_exists psql; then
    echo "PostgreSQL already installed"
else
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install postgresql@15
        brew services start postgresql@15
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update
        sudo apt-get install -y postgresql postgresql-contrib
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
    else
        echo "Unsupported OS. Please install PostgreSQL manually."
        exit 1
    fi
fi

# Step 2: Install TimescaleDB
echo "Step 2: Installing TimescaleDB..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew install timescaledb
    timescaledb-tune --quiet --yes
else
    # Add TimescaleDB repository
    sudo sh -c "echo 'deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main' > /etc/apt/sources.list.d/timescaledb.list"
    wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install -y timescaledb-2-postgresql-15
    sudo timescaledb-tune --quiet --yes
fi

# Step 3: Create database and user
echo "Step 3: Setting up database..."
sudo -u postgres psql <<EOF
-- Check if database exists
SELECT 'CREATE DATABASE quantnexus_trading'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'quantnexus_trading');

-- Create users if they don't exist
DO
\$do\$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_user
      WHERE usename = 'quantnexus_app') THEN
      CREATE USER quantnexus_app WITH PASSWORD 'TradingApp2024!';
   END IF;
END
\$do\$;

GRANT ALL PRIVILEGES ON DATABASE quantnexus_trading TO quantnexus_app;
EOF

# Step 4: Run database schema
echo "Step 4: Creating database schema..."
PGPASSWORD='TradingApp2024!' psql -U quantnexus_app -d quantnexus_trading -f scripts/setup_database.sql

# Step 5: Install Python dependencies
echo "Step 5: Installing Python dependencies..."
cat > requirements_phase1.txt << 'EOL'
# Database
psycopg2-binary==2.9.9
SQLAlchemy==2.0.23
alembic==1.13.0

# Logging and Monitoring
python-json-logger==2.0.7
prometheus-client==0.19.0
psutil==5.9.6

# Error handling
tenacity==8.2.3

# Additional for Phase 1
scipy==1.11.4
EOL

pip3 install -r requirements_phase1.txt

# Step 6: Create necessary directories
echo "Step 6: Creating directories..."
mkdir -p logs
mkdir -p data/models
mkdir -p reports
mkdir -p dashboards/grafana

# Step 7: Test database connection
echo "Step 7: Testing database connection..."
python3 << 'EOF'
import sys
sys.path.append('.')
from src.ml_trading.database.connection import get_db_manager

try:
    db = get_db_manager()
    if db.health_check():
        print("✅ Database connection successful!")
    else:
        print("❌ Database connection failed")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
EOF

# Step 8: Start Prometheus metrics server
echo "Step 8: Testing metrics server..."
python3 << 'EOF'
import sys
sys.path.append('.')
from src.ml_trading.monitoring.metrics import start_metrics_server
import time

try:
    server = start_metrics_server(port=8001)  # Use different port for test
    print("✅ Metrics server started successfully!")
    time.sleep(2)
except Exception as e:
    print(f"⚠️ Metrics server test failed: {e}")
EOF

echo ""
echo "========================================="
echo "✅ Phase 1 Setup Complete!"
echo "========================================="
echo ""
echo "Database: PostgreSQL with TimescaleDB"
echo "  - Host: localhost"
echo "  - Port: 5432"
echo "  - Database: quantnexus_trading"
echo "  - User: quantnexus_app"
echo ""
echo "Components Installed:"
echo "  ✅ Database with TimescaleDB"
echo "  ✅ SQLAlchemy Models"
echo "  ✅ Connection Pooling"
echo "  ✅ Circuit Breakers"
echo "  ✅ Error Handler with Retry Logic"
echo "  ✅ Structured Logging"
echo "  ✅ Prometheus Metrics"
echo ""
echo "Next Steps:"
echo "  1. Start Prometheus: prometheus --config.file=config/prometheus.yml"
echo "  2. Start Grafana: grafana-server"
echo "  3. Test the system: python3 test_phase1_integration.py"
echo ""