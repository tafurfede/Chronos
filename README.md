# Nexus Trading System

Minimal, organized, and documented with only the commands we actually use.

### Requirements
- Python 3.10+
- Bash shell (macOS/Linux)
- Optional: Docker

### Quick Start
```bash
# 1) Setup environment and install dependencies
./run.sh setup

# 2) Train models or run backtests
./run.sh train
./run.sh backtest

# 3) Start paper trading
./run.sh trade

# Extras
./run.sh tests         # run all tests
./run.sh mcp-server    # run MCP trading server
./run.sh api           # start API trading server
./run.sh docker-up     # bring up docker services
```

### Commands (./run.sh)
| Command | What it does |
|--------|---------------|
| `setup` | Create venv and install requirements |
| `train` | Run training via `bin/launch_ml_trading.py --mode train` |
| `backtest` | Run backtests via `bin/launch_ml_trading.py --mode backtest` |
| `trade` | Start paper trading via `bin/start_trading.py` |
| `trade-live` | Start live trading (uses same entry; caution) |
| `mcp-server` | Run `mcp_server/trading_server.py` |
| `api` | Start services `services/api_trading_server.py` |
| `monitor` | Launch monitoring dashboard script |
| `metrics` | Start metrics exporter |
| `tests` | Run full test suite with pytest |
| `unit` | Run unit tests only |
| `integration` | Run integration tests only |
| `docker-up` | docker compose up -d |
| `docker-down` | docker compose down |
| `logs` | Tail core application logs |

### Project Structure
```
Nexus/
├── bin/                    # Runtime entry scripts
│   ├── start_trading.py
│   ├── launch_ml_trading.py
│   ├── demo_trading.py
│   ├── quick_trade.py
│   ├── monitor_dashboard.py
│   └── start_metrics.py
├── services/               # Long-running services
│   └── api_trading_server.py
├── src/                    # Application source (libraries)
│   └── ml_trading/
│       ├── core/
│       ├── backtesting/
│       ├── analysis/
│       ├── database/
│       ├── monitoring/
│       ├── strategies/
│       └── utils/
├── mcp_server/             # MCP server and tools
├── config/                 # Configuration and alerting
├── scripts/                # Setup and DB helpers
├── tests/                  # Unit and integration tests
│   └── integration/        # Moved root tests here
├── data/                   # Data and model artifacts
├── docs/                   # Guides and research papers
├── logs/                   # App logs
├── run.sh                  # Main command runner
└── docker-compose.yml      # Docker services
```

### Configuration
- Environment variables: use `config/.env` (copy from your existing env source).
- Trading parameters: `config/trading_config.yaml`.

### Notes
- Python entry points were moved into `bin/` and service into `services/`. The `./run.sh` launcher handles the correct paths so you don’t need to adjust imports.
- Root-level tests were moved into `tests/integration/`.
