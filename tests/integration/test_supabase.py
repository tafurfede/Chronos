#!/usr/bin/env python3
"""
Test Supabase connection and help with setup
"""

import os
import sys
import psycopg
from dotenv import load_dotenv

print("🔍 Supabase Connection Tester")
print("="*50)

# Load cloud configuration
load_dotenv('.env.cloud')

database_url = os.getenv('DATABASE_URL')

if not database_url:
    print("❌ No DATABASE_URL found in .env.cloud")
    sys.exit(1)

print(f"📝 Connection string: {database_url[:30]}...")

# Parse the URL
try:
    # Try to connect directly with psycopg
    print("\n🔗 Attempting to connect to Supabase...")
    
    # Clean up the URL for psycopg
    clean_url = database_url
    if 'postgresql://' in clean_url:
        # psycopg3 can use postgresql:// directly
        pass
    
    with psycopg.connect(clean_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()
            print(f"✅ Connected successfully!")
            print(f"📊 Database version: {version[0][:50]}...")
            
            # Check if we can create tables
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                LIMIT 5
            """)
            tables = cur.fetchall()
            
            print(f"\n📋 Existing tables: {len(tables)}")
            for table in tables:
                print(f"  • {table[0]}")
                
except psycopg.OperationalError as e:
    print(f"\n❌ Connection failed: {e}")
    print("\n🔧 Troubleshooting steps:")
    print("1. Check your Supabase Dashboard for the correct connection string")
    print("2. Make sure your password doesn't have special characters, or URL-encode them:")
    print("   • $ → %24")
    print("   • @ → %40")
    print("   • # → %23")
    print("3. Ensure your Supabase project is active (not paused)")
    print("\n📝 The connection string format should be:")
    print("postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres")
    
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    print("\nPlease check your connection string in .env.cloud")