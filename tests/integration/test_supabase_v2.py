#!/usr/bin/env python3
"""
Test Supabase connection with multiple approaches
"""

import os
import sys
import socket
from dotenv import load_dotenv

print("🔍 Advanced Supabase Connection Test")
print("="*50)

# Load cloud configuration
load_dotenv('.env.cloud')

database_url = os.getenv('DATABASE_URL')

# First, let's resolve the hostname
hostname = "db.attaulxkgvidiaoxptqw.supabase.co"
print(f"\n📡 Resolving {hostname}...")

try:
    # Get IP address
    ip_info = socket.getaddrinfo(hostname, 5432, socket.AF_INET)
    if ip_info:
        ipv4_address = ip_info[0][4][0]
        print(f"✅ Found IPv4: {ipv4_address}")
        
        # Try connection with IP address
        import psycopg
        
        # Method 1: Direct connection with original URL
        print("\n🔗 Method 1: Testing direct connection...")
        try:
            with psycopg.connect(database_url) as conn:
                print("✅ Direct connection successful!")
                with conn.cursor() as cur:
                    cur.execute("SELECT version()")
                    version = cur.fetchone()
                    print(f"📊 Database: {version[0][:30]}...")
        except Exception as e:
            print(f"❌ Direct connection failed: {e}")
            
            # Method 2: Try with IP address
            print("\n🔗 Method 2: Testing with IPv4 address...")
            ip_url = database_url.replace(hostname, ipv4_address)
            try:
                with psycopg.connect(ip_url) as conn:
                    print("✅ IP connection successful!")
            except Exception as e:
                print(f"❌ IP connection failed: {e}")
                
                # Method 3: Try connection pooling
                print("\n🔗 Method 3: Trying connection pooling (port 6543)...")
                pooled_url = database_url.replace(':5432', ':6543') + '?sslmode=require'
                print(f"Pooled URL: {pooled_url[:50]}...")
                try:
                    with psycopg.connect(pooled_url) as conn:
                        print("✅ Pooled connection successful!")
                except Exception as e:
                    print(f"❌ Pooled connection failed: {e}")
                    
except socket.gaierror as e:
    print(f"❌ Cannot resolve hostname: {e}")
    print("\n⚠️  Your Supabase project might be:")
    print("• Still initializing (wait 2-3 minutes)")
    print("• Paused (check Supabase dashboard)")
    print("• In a different region")

except Exception as e:
    print(f"❌ Unexpected error: {e}")

print("\n📝 If all methods fail:")
print("1. Check if your Supabase project is active")
print("2. Try enabling 'Connection pooling' in Supabase settings")
print("3. Make sure you're using the correct project reference")
print("4. Try using the Supabase connection pooler URL (port 6543)")