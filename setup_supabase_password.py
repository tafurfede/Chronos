#!/usr/bin/env python3
"""
Help set up Supabase password in connection string
"""

import urllib.parse
import getpass

print("🔐 Supabase Password Setup")
print("="*50)

print("\nYour Supabase project reference is: attaulxkgvidiaoxptqw")
print("\nYou need to replace [YOUR-PASSWORD] with your actual password.")
print("\nThis is the password you set when creating your Supabase project.")

print("\n📝 Options:")
print("1. Enter your password now (will be URL-encoded automatically)")
print("2. Reset your password in Supabase dashboard")
print("3. Use the password 'QuantNexus2024' (if you set it earlier)")

choice = input("\nEnter your choice (1-3): ")

if choice == "1":
    password = getpass.getpass("\n🔑 Enter your Supabase password: ")
    if password:
        # URL-encode special characters
        encoded_password = urllib.parse.quote(password, safe='')
        connection_string = f"postgresql://postgres:{encoded_password}@db.attaulxkgvidiaoxptqw.supabase.co:5432/postgres"
        
        print(f"\n✅ Your connection string is:")
        print(connection_string)
        
        update = input("\nUpdate .env.cloud with this connection string? (y/n): ")
        if update.lower() == 'y':
            # Update .env.cloud
            with open('.env.cloud', 'r') as f:
                lines = f.readlines()
            
            with open('.env.cloud', 'w') as f:
                for line in lines:
                    if line.startswith('DATABASE_URL='):
                        f.write(f'DATABASE_URL={connection_string}\n')
                    else:
                        f.write(line)
            
            print("✅ Updated .env.cloud successfully!")
            print("\nNow testing connection...")
            
            import os
            os.environ['DATABASE_URL'] = connection_string
            
            try:
                import psycopg
                with psycopg.connect(connection_string) as conn:
                    print("✅ Connection successful!")
                    with conn.cursor() as cur:
                        cur.execute("SELECT version()")
                        version = cur.fetchone()
                        print(f"📊 Connected to: {version[0][:50]}...")
            except Exception as e:
                print(f"❌ Connection failed: {e}")
                print("\nPossible issues:")
                print("• Wrong password")
                print("• Project still initializing")
                print("• Project paused in Supabase dashboard")

elif choice == "2":
    print("\n📝 To reset your password:")
    print("1. Go to https://app.supabase.com")
    print("2. Select your project")
    print("3. Settings → Database")
    print("4. Click 'Reset database password'")
    print("5. Set new password (e.g., 'QuantNexus2024')")
    print("6. Run this script again with the new password")

elif choice == "3":
    # Use the password from earlier attempts
    password = "QuantNexus2024"
    connection_string = f"postgresql://postgres:{password}@db.attaulxkgvidiaoxptqw.supabase.co:5432/postgres"
    
    print(f"\n📝 Using password: QuantNexus2024")
    print(f"Connection string: {connection_string}")
    
    # Update .env.cloud
    with open('.env.cloud', 'r') as f:
        lines = f.readlines()
    
    with open('.env.cloud', 'w') as f:
        for line in lines:
            if line.startswith('DATABASE_URL='):
                f.write(f'DATABASE_URL={connection_string}\n')
            else:
                f.write(line)
    
    print("✅ Updated .env.cloud")
    print("\nTesting connection...")
    
    import os
    os.environ['DATABASE_URL'] = connection_string
    
    try:
        import psycopg
        with psycopg.connect(connection_string) as conn:
            print("✅ Connection successful!")
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()
                print(f"📊 Connected to: {version[0][:50]}...")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nThe password might be different. Please check Supabase dashboard.")