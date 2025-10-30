# Cloud Database Setup Guide

## Recommended Cloud Database Providers:

### 1. **Supabase (FREE Tier Available)**
- Sign up at: https://supabase.com
- Create new project
- Get connection string from Settings → Database
- Connection string format:
  ```
  postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
  ```

### 2. **Neon (FREE Tier with 0.5GB)**
- Sign up at: https://neon.tech
- Create database
- Get connection string from Dashboard
- Connection string format:
  ```
  postgresql://[USER]:[PASSWORD]@[HOST]/[DATABASE]?sslmode=require
  ```

### 3. **Railway (Pay-as-you-go)**
- Sign up at: https://railway.app
- Deploy PostgreSQL
- Get DATABASE_URL from Variables tab

### 4. **AWS RDS (Production-grade)**
- More complex setup
- Better for high-volume trading
- Costs ~$15-50/month minimum

## How to Connect to Cloud Database:

1. **Update your .env file:**
```bash
# Replace local database settings with cloud URL
DATABASE_URL=postgresql://user:password@host:port/dbname
```

2. **Update connection.py to use DATABASE_URL:**
```python
def _get_db_url(self) -> str:
    # First check for DATABASE_URL (cloud database)
    if os.getenv('DATABASE_URL'):
        return os.getenv('DATABASE_URL')
    
    # Otherwise use local database
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'quantnexus_trading')
    db_user = os.getenv('DB_USER', 'quantnexus_app')
    db_password = os.getenv('DB_PASSWORD', 'TradingApp2024!')
    
    return f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
```

3. **Run database setup:**
```bash
python3 -c "
from src.ml_trading.database.connection import get_db_manager
db = get_db_manager()
print('Connected to cloud database!')
"
```

## Security Best Practices:

1. **Never commit .env to git:**
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Use environment variables in production:**
   ```bash
   export DATABASE_URL="your-connection-string"
   ```

3. **Rotate passwords regularly**

4. **Use SSL connections for cloud databases**

5. **Enable IP whitelisting if available**