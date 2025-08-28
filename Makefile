# QuantNexus ML Trading System Makefile

# Check if virtual environment exists
VENV := venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip3

.PHONY: help setup install train backtest trade analyze clean docker-build docker-run mcp-server mcp-client mcp-docker mcp-example

help:
	@echo "QuantNexus ML Trading System - Available Commands:"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup        - Create virtual environment and install all dependencies"
	@echo ""
	@echo "Standard Commands:"
	@echo "  make install      - Install all dependencies"
	@echo "  make train        - Train ML models"
	@echo "  make backtest     - Run backtesting"
	@echo "  make trade        - Start live trading (paper)"
	@echo "  make analyze      - Analyze performance"
	@echo "  make clean        - Clean temporary files"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run in Docker container"
	@echo ""
	@echo "MCP Commands:"
	@echo "  make mcp-server   - Start MCP trading server"
	@echo "  make mcp-client   - Run MCP client example"
	@echo "  make mcp-docker   - Run MCP with Docker"
	@echo "  make mcp-install  - Install MCP dependencies"
	@echo "  make mcp-dev      - Start MCP in development mode"

# Setup virtual environment
setup:
	@echo "🚀 Setting up QuantNexus ML Trading System..."
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV); \
	fi
	@echo "Installing dependencies..."
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -r requirements_mcp.txt
	@echo "✅ Setup complete! Virtual environment ready."
	@echo ""
	@echo "To activate manually: source venv/bin/activate"

install: $(VENV)
	@echo "Installing dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements_ml.txt
	@echo "✅ Installation complete"

train: $(VENV)
	@echo "Training ML models..."
	@$(PYTHON) launch_ml_trading.py --mode train
	@echo "✅ Training complete"

backtest: $(VENV)
	@echo "Running backtest..."
	@$(PYTHON) launch_ml_trading.py --mode backtest
	@echo "✅ Backtest complete"

trade: $(VENV)
	@echo "Starting ML Trading Bot..."
	@$(PYTHON) launch_ml_trading.py --mode trade --paper
	
trade-live: $(VENV)
	@echo "⚠️  Starting LIVE Trading (Real Money)..."
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ] || exit 1
	@$(PYTHON) launch_ml_trading.py --mode trade --no-paper

analyze: $(VENV)
	@echo "Analyzing performance..."
	@$(PYTHON) launch_ml_trading.py --mode analyze
	@echo "✅ Analysis complete"

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src/ml_trading
	@echo "✅ Tests complete"

clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	@echo "✅ Cleanup complete"

docker-build:
	@echo "Building Docker image..."
	docker build -t quantnexus-ml:latest .
	@echo "✅ Docker image built"

docker-run:
	@echo "Running in Docker..."
	docker-compose up -d
	@echo "✅ Container started"

docker-stop:
	@echo "Stopping Docker containers..."
	docker-compose down
	@echo "✅ Containers stopped"

setup-db: $(VENV)
	@echo "Setting up database..."
	@$(PYTHON) scripts/setup_database.py
	@echo "✅ Database setup complete"

monitor:
	@echo "Opening monitoring dashboard..."
	@$(PYTHON) -m webbrowser http://localhost:3000
	
logs:
	@echo "Tailing logs..."
	tail -f logs/trading.log

performance-report: $(VENV)
	@echo "Generating performance report..."
	@$(PYTHON) scripts/generate_performance_report.py
	@echo "✅ Report generated: reports/performance_report.html"

# MCP Server Commands
mcp-install: $(VENV)
	@echo "Installing MCP dependencies..."
	@$(PIP) install mcp fastapi uvicorn websockets aiohttp
	@echo "✅ MCP dependencies installed"

mcp-server: $(VENV)
	@echo "Starting Trading API Server..."
	@$(PYTHON) api_trading_server.py

mcp-dev: $(VENV)
	@echo "Starting MCP Server in development mode..."
	@MCP_MODE=development $(PYTHON) -m mcp_server.trading_server

mcp-client: $(VENV)
	@echo "Running MCP client example..."
	@$(PYTHON) examples/mcp_trading_example.py

mcp-docker:
	@echo "Starting MCP with Docker..."
	docker-compose -f docker-compose.mcp.yml up -d
	@echo "✅ MCP server running at http://localhost:8080"
	@echo "✅ Web UI available at http://localhost:3001"

mcp-docker-stop:
	@echo "Stopping MCP Docker containers..."
	docker-compose -f docker-compose.mcp.yml down
	@echo "✅ MCP containers stopped"

mcp-test: $(VENV)
	@echo "Testing MCP integration..."
	@$(PYTHON) -m pytest tests/test_mcp_server.py -v
	@echo "✅ MCP tests complete"

# Virtual environment target
$(VENV):
	@echo "Creating virtual environment..."
	@python3 -m venv $(VENV)
	@$(PIP) install --upgrade pip