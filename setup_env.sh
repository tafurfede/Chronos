#!/bin/bash
# Setup script for QuantNexus ML Trading System

echo "🚀 Setting up QuantNexus ML Trading System..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements_ml.txt

# Install MCP dependencies
echo "Installing MCP dependencies..."
pip install mcp fastapi uvicorn websockets aiohttp

echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Then you can run:"
echo "  make mcp-server   # Start MCP server"
echo "  make mcp-client   # Run examples"