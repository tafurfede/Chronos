# QuantNexus ML Trading System 🚀

## Advanced ML-Powered Trading Bot for 65%+ Success Rate
### Target: $10K → $500K in 12 Months

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20LSTM%20%7C%20Neural%20Networks-green)
![Trading](https://img.shields.io/badge/Trading-Alpaca%20API-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 🎯 Overview

QuantNexus is a production-ready ML trading system designed to achieve consistent 65%+ win rates through advanced machine learning, featuring:

- **200+ Technical Features**: Comprehensive feature engineering pipeline
- **Ensemble ML Models**: XGBoost, LightGBM, Neural Networks, LSTM
- **Dynamic Risk Management**: ML-optimized stop losses and position sizing
- **Real-Time Execution**: Sub-second decision making with Alpaca API
- **Complete Infrastructure**: Docker, PostgreSQL, Redis, Monitoring

## 📊 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Win Rate | 65%+ | Training |
| Avg Win/Loss Ratio | 2.5:1 | - |
| Daily Trades | 15-25 | - |
| Annual Return | 50x | - |
| Max Drawdown | < 15% | - |
| Sharpe Ratio | > 2.5 | - |

## 🏗️ System Architecture

```
┌─────────────────────────────────────────┐
│         DATA INGESTION LAYER            │
├─────────────────────────────────────────┤
│  • Real-time price feeds                │
│  • Options flow data                     │
│  • News sentiment analysis              │
│  • Market microstructure                │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      FEATURE ENGINEERING (200+)         │
├─────────────────────────────────────────┤
│  • Price features (40)                  │
│  • Volume features (20)                 │
│  • Technical indicators (60)            │
│  • Market microstructure (30)           │
│  • Time patterns (25)                   │
│  • Volatility features (15)             │
│  • Cross-sectional (10)                 │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         ML PREDICTION ENGINE            │
├─────────────────────────────────────────┤
│  • XGBoost (Tree-based)                 │
│  • LightGBM (Gradient Boosting)         │
│  • Neural Network (Deep Learning)       │
│  • LSTM (Time Series)                   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│       EXECUTION & RISK MANAGEMENT       │
├─────────────────────────────────────────┤
│  • Kelly Criterion position sizing      │
│  • Dynamic ML stop losses               │
│  • Real-time order execution            │
│  • Portfolio optimization               │
└─────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Alpaca Trading Account (Paper or Live)
- PostgreSQL 14+
- 8GB+ RAM recommended

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quantnexus-ml-trading.git
cd quantnexus-ml-trading

# Install dependencies
make install

# Setup configuration
cp config/.env.example config/.env
# Edit config/.env with your API keys

# Setup database
make setup-db
```

### 3. Train Models

```bash
# Train ML models with historical data
make train

# Or manually:
python launch_ml_trading.py --mode train
```

### 4. Run Backtest

```bash
# Run comprehensive backtest
make backtest

# View results in reports/backtest_report.html
```

### 5. Start Trading

```bash
# Paper trading (recommended for testing)
make trade

# Live trading (use with caution!)
make trade-live
```

### 6. Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f trading-bot

# Access monitoring dashboard
open http://localhost:3000  # Grafana
open http://localhost:8888  # Jupyter
```

## 📁 Project Structure

```
quantnexus-ml-trading/
├── src/
│   └── ml_trading/
│       ├── core/              # Core trading components
│       │   ├── ml_trading_system.py
│       │   └── model_trainer.py
│       ├── backtesting/       # Backtesting engine
│       │   └── backtest_engine.py
│       ├── analysis/          # Performance analysis
│       │   └── performance_analyzer.py
│       ├── models/            # Saved ML models
│       └── utils/             # Utility functions
├── config/
│   ├── .env.example          # Environment variables
│   └── trading_config.yaml   # Trading configuration
├── data/
│   ├── raw/                  # Raw market data
│   ├── processed/            # Processed features
│   └── models/               # Trained models
├── scripts/
│   ├── init_db.sql          # Database schema
│   └── setup_database.py     # Database setup
├── reports/                   # Performance reports
├── logs/                      # Application logs
├── tests/                     # Unit & integration tests
├── docker-compose.yml         # Docker orchestration
├── Dockerfile                 # Container definition
├── Makefile                   # Build automation
├── requirements_ml.txt        # Python dependencies
└── launch_ml_trading.py      # Main launcher
```

## 🧠 ML Models

### Ensemble Components

1. **XGBoost**: Gradient boosting for non-linear patterns
2. **LightGBM**: Fast gradient boosting for large datasets
3. **Neural Network**: Deep learning for complex relationships
4. **LSTM**: Sequence modeling for time series patterns

### Feature Engineering

- **Price Features**: Returns, moving averages, support/resistance
- **Volume Features**: OBV, VWAP, volume patterns
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic
- **Market Microstructure**: Spread, order flow, liquidity
- **Time Patterns**: Trends, momentum, seasonality
- **Volatility**: Historical, Parkinson, Garman-Klass

## 📈 Performance Monitoring

### Real-Time Dashboard
- Live P&L tracking
- Win rate monitoring
- Position management
- Risk metrics
- Model confidence scores

### Reports
- Daily performance summary
- Trade analysis
- Model performance metrics
- Risk assessment

## 🔧 Configuration

### Trading Parameters
Edit `config/trading_config.yaml`:
```yaml
strategy:
  signals:
    min_confidence: 0.65    # Minimum ML confidence
    min_expected_return: 0.002  # 0.2% minimum
  risk:
    stop_loss_pct: 0.015    # 1.5% stop loss
    profit_target_pct: 0.03  # 3% profit target
```

### API Configuration
Set in `config/.env`:
```bash
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret
INITIAL_CAPITAL=10000
TARGET_CAPITAL=500000
```

## 📊 Backtesting

Run comprehensive backtests:
```python
from src.ml_trading.backtesting import BacktestEngine

engine = BacktestEngine()
results = engine.run_backtest({
    'start_date': '2023-01-01',
    'end_date': '2023-12-31',
    'initial_capital': 10000
})
```

## 🛡️ Risk Management

- **Position Sizing**: Kelly Criterion with safety factor
- **Stop Losses**: ML-optimized dynamic stops
- **Portfolio Limits**: Max 20 positions, 10% per position
- **Daily Loss Limit**: 5% maximum daily drawdown
- **Correlation Management**: Monitor position correlations

## 📝 Commands

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies |
| `make train` | Train ML models |
| `make backtest` | Run backtest simulation |
| `make trade` | Start paper trading |
| `make trade-live` | Start live trading |
| `make analyze` | Analyze performance |
| `make test` | Run test suite |
| `make clean` | Clean temporary files |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run in Docker |

## 🔬 Testing

```bash
# Run all tests
make test

# Run specific tests
pytest tests/unit/test_feature_engine.py
pytest tests/integration/test_trading_system.py
```

## 📚 Documentation

- [Complete Blueprint](ML_TRADING_SYSTEM_COMPLETE_BLUEPRINT.md)
- [API Documentation](docs/api.md)
- [Model Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)

## ⚠️ Disclaimer

**IMPORTANT**: This software is for educational purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always:

- Start with paper trading
- Never risk more than you can afford to lose
- Understand the system before using real money
- Monitor performance continuously
- Have proper risk management

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📧 Support

- Issues: [GitHub Issues](https://github.com/yourusername/quantnexus-ml-trading/issues)
- Documentation: [Wiki](https://github.com/yourusername/quantnexus-ml-trading/wiki)
- Email: support@quantnexus.ai

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

**Built with ❤️ for the future of algorithmic trading**