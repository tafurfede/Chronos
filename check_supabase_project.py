#!/usr/bin/env python3
"""
Check and verify Supabase project details
"""

import sys

print("🔍 Supabase Project Verification")
print("="*50)

print("""
The connection string you provided:
postgresql://postgres:QuantNexus2024@db.attaulxkgvidiaoxptqw.supabase.co:5432/postgres

The project reference 'attaulxkgvidiaoxptqw' cannot be resolved.

✅ Please follow these steps to get the correct connection string:

1. Go to https://app.supabase.com
2. Sign in to your account
3. Click on your project
4. Go to Settings (gear icon)
5. Click on "Database" in the left menu
6. Find the "Connection string" section
7. Look for the "URI" field
8. Copy the ENTIRE string (it will look like):
   postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres

The [PROJECT-REF] part should be about 20 characters long, like:
- jsuwszcwmxbfucvsgseh
- xyzabcdefghijklmnopq
- etc.

⚠️  Common issues:
• Make sure you're copying from the correct project
• The project reference is case-sensitive
• Don't accidentally cut off part of the reference

🔄 Alternative: Connection Pooling
If direct connection doesn't work, try the pooled connection:
1. In Database settings, look for "Connection pooling"
2. Enable it if not already enabled
3. Use the pooled connection string (port 6543 instead of 5432)

Once you have the correct connection string, update .env.cloud with it.
""")

# Let's check what might be the issue
print("\n📋 Quick checklist:")
print("✓ Is your Supabase project active? (not paused)")
print("✓ Did you copy the complete project reference?")
print("✓ Is the project in the correct region?")
print("✓ Has the project finished initializing? (can take 2-3 minutes)")

print("\n💡 Next step:")
print("Get the correct connection string from Supabase dashboard")
print("Then update .env.cloud with the correct DATABASE_URL")