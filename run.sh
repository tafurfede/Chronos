#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<EOF
Usage: ./run.sh <command> [args]

Commands:
  setup           Create venv and install dependencies
  train           Train ML models
  backtest        Run backtests
  trade           Start paper trading
  trade-live      Start live trading (CAUTION)
  mcp-server      Run MCP trading server
  api             Start API trading server (services)
  monitor         Launch monitoring dashboard
  metrics         Start metrics exporter
  tests           Run full test suite
  unit            Run unit tests
  integration     Run integration tests
  docker-up       Start docker-compose services
  docker-down     Stop docker-compose services
  logs            Tail core logs
  help            Show this help
EOF
}

ensure_venv() {
  if [ ! -d "$ROOT_DIR/venv" ]; then
    python3 -m venv "$ROOT_DIR/venv"
  fi
  # shellcheck source=/dev/null
  source "$ROOT_DIR/venv/bin/activate"
  pip install -q -U pip
}

case "${1:-help}" in
  setup)
    ensure_venv
    pip install -r "$ROOT_DIR/requirements_ml.txt"
    pip install -r "$ROOT_DIR/requirements_mcp.txt" || true
    ;;

  train)
    ensure_venv
    python "$ROOT_DIR/bin/launch_ml_trading.py" --mode train || python "$ROOT_DIR/launch_ml_trading.py" --mode train
    ;;

  backtest)
    ensure_venv
    python "$ROOT_DIR/bin/launch_ml_trading.py" --mode backtest || python "$ROOT_DIR/launch_ml_trading.py" --mode backtest
    ;;

  trade)
    ensure_venv
    python "$ROOT_DIR/bin/start_trading.py" || python "$ROOT_DIR/start_trading.py"
    ;;

  trade-live)
    ensure_venv
    ALPACA_LIVE=1 python "$ROOT_DIR/bin/start_trading.py" || ALPACA_LIVE=1 python "$ROOT_DIR/start_trading.py"
    ;;

  mcp-server)
    ensure_venv
    python "$ROOT_DIR/mcp_server/trading_server.py"
    ;;

  api)
    ensure_venv
    python "$ROOT_DIR/services/api_trading_server.py"
    ;;

  monitor)
    ensure_venv
    python "$ROOT_DIR/bin/monitor_dashboard.py" || python "$ROOT_DIR/monitor_dashboard.py"
    ;;

  metrics)
    ensure_venv
    python "$ROOT_DIR/bin/start_metrics.py" || python "$ROOT_DIR/start_metrics.py"
    ;;

  tests)
    ensure_venv
    pytest -q "$ROOT_DIR/tests"
    ;;

  unit)
    ensure_venv
    pytest -q "$ROOT_DIR/tests/unit"
    ;;

  integration)
    ensure_venv
    pytest -q "$ROOT_DIR/tests/integration"
    ;;

  docker-up)
    docker compose -f "$ROOT_DIR/docker-compose.yml" up -d
    ;;

  docker-down)
    docker compose -f "$ROOT_DIR/docker-compose.yml" down
    ;;

  logs)
    tail -n 200 -f "$ROOT_DIR/logs/quantnexus.log" "$ROOT_DIR/logs/quantnexus_errors.log" 2>/dev/null || true
    ;;

  help|*)
    usage
    ;;
esac


