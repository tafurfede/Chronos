#!/usr/bin/env python3
"""
Verify and help fix Supabase URL
"""

print("🔍 Checking your Supabase connection string...")
print("="*50)

# The URL you provided
provided_url = "postgresql://postgres:QuantNexus2024@db.attaulxkgvidiaoxptqw.supabase.co:5432/postgre"

print(f"\n📝 URL provided:")
print(provided_url)

print("\n⚠️  Issues found:")
print("1. The URL ends with '/postgre' but should end with '/postgres'")
print("2. Let's verify the project reference is correct")

print("\n✅ Corrected URL should be:")
corrected_url = "postgresql://postgres:QuantNexus2024@db.attaulxkgvidiaoxptqw.supabase.co:5432/postgres"
print(corrected_url)

print("\n📋 Please verify in your Supabase Dashboard:")
print("1. Go to https://app.supabase.com")
print("2. Select your project")
print("3. Go to Settings → Database")
print("4. Find 'Connection string' section")
print("5. Copy the EXACT string from the URI field")
print("\nThe project reference part (after 'db.' and before '.supabase.co')")
print("should be about 20 characters long.")
print("\n🔍 Common issues:")
print("• Make sure you're copying from the right project")
print("• Check if the project is still initializing (can take 2-3 minutes)")
print("• Try the 'Connection pooling' option if direct connection fails")