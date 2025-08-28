# 🚀 Supabase Setup Guide for QuantNexus

## Current Status:
✅ Supabase project created  
✅ Project reference: `jsuwszcwmxbfucvsgseh`  
❌ Need to set correct password in connection string

## What You Need to Do:

### Option 1: If You Remember Your Password
Edit `.env.cloud` and replace `[YOUR-PASSWORD]` with your actual password:

```bash
# If your password is: MySecretPass123
DATABASE_URL=postgresql://postgres:MySecretPass123@db.jsuwszcwmxbfucvsgseh.supabase.co:5432/postgres

# If your password has special characters like: MyPass@123!
# URL-encode them:
DATABASE_URL=postgresql://postgres:MyPass%40123%21@db.jsuwszcwmxbfucvsgseh.supabase.co:5432/postgres
```

### Option 2: Reset Your Password
1. Go to [Supabase Dashboard](https://app.supabase.com)
2. Select your project
3. Settings → Database
4. Click "Reset database password"
5. Set a new password (example: `QuantNexus2024`)
6. Update `.env.cloud`:
   ```
   DATABASE_URL=postgresql://postgres:QuantNexus2024@db.jsuwszcwmxbfucvsgseh.supabase.co:5432/postgres
   ```

### Option 3: Use Connection Pooling (Alternative)
1. In Supabase Dashboard → Settings → Database
2. Enable "Connection pooling"
3. Use the pooled connection string (port 6543):
   ```
   DATABASE_URL=postgresql://postgres:[password]@db.jsuwszcwmxbfucvsgseh.supabase.co:6543/postgres?pgbouncer=true
   ```

## Special Character Encoding:
If your password contains these characters, replace them:
- `@` → `%40`
- `#` → `%23`
- `$` → `%24`
- `!` → `%21`
- `:` → `%3A`
- `/` → `%2F`

## Example Passwords:
| Your Password | Encoded Version |
|--------------|-----------------|
| `Simple123` | `Simple123` (no encoding needed) |
| `Pass@word!` | `Pass%40word%21` |
| `My$ecure#1` | `My%24ecure%231` |
| `Test:Pass/2` | `Test%3APass%2F2` |

## After Updating .env.cloud:
```bash
# Test connection
source venv/bin/activate
python3 test_supabase.py

# If successful, set up database
python3 setup_cloud_db.py
```

## Troubleshooting:
- **Connection refused**: Check password is correct
- **Unknown host**: Check project reference is correct
- **Authentication failed**: Password has special characters that need encoding
- **SSL required**: Add `?sslmode=require` to the end of the URL