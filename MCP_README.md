# 🚀 QuantNexus MCP Trading System

## ML-Powered Trading with Model Context Protocol (MCP)

The QuantNexus Trading System now features **full MCP integration**, making it easier than ever to interact with ML models, execute trades, and manage your portfolio through a unified protocol.

## 🎯 What is MCP Integration?

MCP (Model Context Protocol) provides a standardized way to:
- **Access Trading Tools**: Analyze stocks, execute trades, manage positions
- **Leverage ML Models**: Train, evaluate, and use ensemble models
- **Monitor Performance**: Real-time portfolio tracking and risk metrics
- **Run Backtests**: Test strategies on historical data
- **Automate Workflows**: Execute complex trading strategies

## 📋 MCP Architecture

```
┌─────────────────────────────────────────────────────┐
│                   MCP CLIENTS                        │
│  (Web UI, CLI, Jupyter, VS Code, Custom Apps)       │
└─────────────────────┬───────────────────────────────┘
                      │ MCP Protocol
                      ↓
┌─────────────────────────────────────────────────────┐
│              MCP TRADING SERVER                      │
├─────────────────────────────────────────────────────┤
│  Tools:                                              │
│  • analyze_stock      • execute_trade                │
│  • get_market_signals • close_position               │
│  • train_models       • run_backtest                 │
│  • get_portfolio      • calculate_risk               │
├─────────────────────────────────────────────────────┤
│  Resources:                                          │
│  • market_data/{symbol}  • portfolio/state           │
│  • signals/latest        • models/metrics            │
├─────────────────────────────────────────────────────┤
│  Prompts:                                            │
│  • analyze_stock  • execute_trade  • daily_trading   │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│           ML TRADING SYSTEM CORE                     │
│  (200+ Features, XGBoost, LSTM, Neural Networks)     │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start with MCP

### 1. Install MCP Dependencies

```bash
make mcp-install
```

### 2. Start MCP Server

```bash
# Development mode
make mcp-dev

# Production mode
make mcp-server

# Docker mode (recommended)
make mcp-docker
```

### 3. Use MCP Client

#### Python Client Example:

```python
from src.mcp_trading.mcp_client import SimpleTradingClient

# Connect to MCP server
with SimpleTradingClient() as client:
    # Analyze a stock
    analysis = client.analyze("AAPL")
    print(f"Signal: {analysis['ml_prediction']['signal']}")
    print(f"Confidence: {analysis['ml_prediction']['confidence']:.1%}")
    
    # Execute trade if confidence is high
    if analysis['ml_prediction']['confidence'] > 0.65:
        trade = client.buy("AAPL", quantity=10)
        print(f"Executed: {trade}")
    
    # Check portfolio
    portfolio = client.portfolio()
    print(f"Portfolio Value: ${portfolio['total_value']:.2f}")
```

#### Async Client Example:

```python
import asyncio
from src.mcp_trading.mcp_client import TradingMCPClient

async def trading_workflow():
    client = TradingMCPClient()
    await client.connect()
    
    # Get market signals
    signals = await client.get_market_signals(min_confidence=0.65)
    
    # Execute top signals
    for signal in signals[:3]:
        if signal['signal'] == 'buy':
            await client.execute_trade(
                symbol=signal['symbol'],
                action='buy'
            )
    
    # Monitor portfolio
    portfolio = await client.get_portfolio_status()
    print(f"Daily P&L: ${portfolio['daily_pnl']:.2f}")
    
    await client.disconnect()

asyncio.run(trading_workflow())
```

## 📊 MCP Tools Available

### Market Analysis Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `analyze_stock` | Comprehensive ML analysis | `symbol`, `period` |
| `get_market_signals` | High-confidence signals | `min_confidence` |

### Trading Execution Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `execute_trade` | Execute with ML optimization | `symbol`, `action`, `quantity` |
| `close_position` | Close existing position | `symbol`, `reason` |

### Portfolio Management Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_portfolio_status` | Current portfolio state | None |
| `calculate_risk_metrics` | VaR, Sharpe, drawdown | None |

### ML Model Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `train_models` | Train ML ensemble | `symbols` |
| `get_model_performance` | Model metrics | None |

### Backtesting Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `run_backtest` | Historical simulation | `start_date`, `end_date`, `capital` |

## 📁 MCP Resources

Access real-time data through MCP resources:

```python
# Access market data
data = await client.get_resource("market_data/AAPL")

# Get portfolio state
portfolio = await client.get_resource("portfolio/state")

# Latest signals
signals = await client.get_resource("signals/latest")

# Model metrics
metrics = await client.get_resource("models/metrics")
```

## 🎯 Example Workflows

### Daily Trading Workflow

```python
# Execute complete daily trading
summary = await client.execute_daily_trading()

# Results include:
# - Signals generated
# - Trades executed
# - Positions closed
# - Portfolio status
# - Daily P&L
```

### Real-Time Monitoring

```python
# Monitor positions with alerts
await client.monitor_positions_realtime(interval=30)

# Automatically checks:
# - Position P&L
# - Risk metrics
# - Stop loss levels
# - Portfolio limits
```

### ML Model Training

```python
# Train models on historical data
result = await client.train_models(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN']
)

# Get performance metrics
metrics = await client.get_model_performance()
```

## 🐳 Docker Deployment

### Start MCP with Docker Compose

```bash
# Start all services
make mcp-docker

# Services running:
# - MCP Server: http://localhost:8080
# - Web UI: http://localhost:3001
# - Grafana: http://localhost:3000
# - Jupyter: http://localhost:8888
```

### Docker Services

- **mcp-server**: Core MCP trading server
- **postgres**: TimescaleDB for time-series data
- **redis**: Caching and pub/sub
- **grafana**: Performance monitoring
- **jupyter**: Interactive analysis

## 🔧 Configuration

### MCP Server Config (`mcp_server/mcp_config.json`)

```json
{
  "server": {
    "port": 8080,
    "workers": 4
  },
  "capabilities": {
    "tools": ["market_analysis", "trading", "ml_models"],
    "resources": ["market_data", "portfolio", "signals"],
    "prompts": ["analyze_stock", "execute_trade"]
  },
  "integrations": {
    "alpaca": {
      "paper_trading": true
    },
    "database": {
      "type": "postgresql"
    }
  }
}
```

## 📈 Performance Metrics

The MCP system tracks:
- **Win Rate**: Target 65%+
- **Sharpe Ratio**: Target > 2.5
- **Max Drawdown**: Limit < 15%
- **Daily P&L**: Real-time tracking
- **Model Confidence**: Per-trade ML confidence

## 🛡️ Security

- **Authentication**: API key required
- **Rate Limiting**: 60 requests/minute
- **Encryption**: TLS for all communications
- **Audit Logging**: All trades logged

## 📝 Testing

Run MCP integration tests:

```bash
make mcp-test
```

## 🤝 MCP Client Libraries

### Python
```bash
pip install quantnexus-mcp-client
```

### JavaScript/TypeScript
```bash
npm install @quantnexus/mcp-client
```

### CLI
```bash
# Install CLI tool
pip install quantnexus-cli

# Use CLI
qn-trade analyze AAPL
qn-trade buy AAPL 10
qn-trade portfolio
```

## 📚 Additional Resources

- [MCP Protocol Specification](https://modelcontextprotocol.org)
- [API Documentation](docs/mcp_api.md)
- [Trading Strategies](docs/strategies.md)
- [ML Model Details](docs/models.md)

## 🚨 Important Notes

1. **Start with Paper Trading**: Always test with paper trading first
2. **Monitor Performance**: Use Grafana dashboards for real-time monitoring
3. **Risk Management**: Never exceed position limits
4. **Model Retraining**: Retrain models weekly for best performance

## 📧 Support

- GitHub Issues: [Report Issues](https://github.com/quantnexus/mcp-trading/issues)
- Documentation: [Full Docs](https://docs.quantnexus.ai)

---

**Built with MCP for the future of algorithmic trading** 🚀