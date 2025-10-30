#!/usr/bin/env python3
"""
Test final Supabase connection
"""

import os
from dotenv import load_dotenv
import psycopg
import socket

# Load environment
load_dotenv('.env.cloud')

database_url = os.getenv('DATABASE_URL')
print("🔍 Testing Supabase Connection")
print("="*50)
print(f"\nConnection string: {database_url[:50]}...")

# Parse the hostname
hostname = "db.attaulxkgvidiaoxptqw.supabase.co"

print(f"\n1. Testing DNS resolution for {hostname}...")
try:
    ip_info = socket.getaddrinfo(hostname, 5432)
    print(f"✅ DNS resolved successfully")
    for info in ip_info[:2]:
        print(f"   • {info[4]}")
except socket.gaierror as e:
    print(f"❌ Cannot resolve hostname: {e}")
    print("\nPossible issues:")
    print("• The project reference might be incomplete")
    print("• Project might be paused or deleted")
    print("• Try checking your Supabase dashboard")
    
    # Try alternative hostnames
    print("\n2. Trying alternative formats...")
    
    # Check if there's a typo or missing characters
    possible_refs = [
        "attaulxkgvidiaoxptqw",  # Original
        "attaulxkgvidiaoxptqws", # +s
        "attaulxkgvidiaoxptqwh", # +h  
        "attaulxkgvidiaoxptqwr", # +r
    ]
    
    for ref in possible_refs:
        test_host = f"db.{ref}.supabase.co"
        try:
            socket.getaddrinfo(test_host, 5432)
            print(f"✅ Found working hostname: {test_host}")
            print(f"   Update your connection string to use: {ref}")
            break
        except:
            continue
    else:
        print("❌ No valid hostname found")
        print("\n📝 Please double-check your Supabase dashboard for the correct project reference")
    
    exit(1)

print("\n2. Testing database connection...")
try:
    with psycopg.connect(database_url) as conn:
        print("✅ Connection successful!")
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()
            print(f"📊 Database version: {version[0][:50]}...")
            
            cur.execute("SELECT current_database()")
            db = cur.fetchone()
            print(f"📊 Database name: {db[0]}")
            
            cur.execute("SELECT current_user")
            user = cur.fetchone()
            print(f"👤 Connected as: {user[0]}")
            
            print("\n✅ Supabase connection is working!")
            print("Ready to migrate tables to cloud database")
            
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("• Check if password is correct")
    print("• Verify project is active in Supabase dashboard")
    print("• Try connection pooling (port 6543) if direct fails")