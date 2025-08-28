#!/usr/bin/env python3
"""
Help update the cloud database connection string
"""

print("🔐 Supabase Connection Setup")
print("="*50)
print("\nYou need to update .env.cloud with your actual Supabase password.")
print("\nThe connection string from Supabase is:")
print("postgresql://postgres:[YOUR-PASSWORD]@db.jsuwszcwmxbfucvsgseh.supabase.co:5432/postgres")
print("\nYou need to replace [YOUR-PASSWORD] with the password you created")
print("when you set up your Supabase project.")
print("\n📝 Steps:")
print("1. Remember the password you used when creating the Supabase project")
print("2. Replace [YOUR-PASSWORD] in the connection string")
print("3. If your password contains special characters, URL-encode them:")
print("   • @ becomes %40")
print("   • # becomes %23")
print("   • $ becomes %24")
print("   • ! becomes %21")
print("\nFor example, if your password is: MyPass@123!")
print("The connection string would be:")
print("postgresql://postgres:MyPass%40123%21@db.jsuwszcwmxbfucvsgseh.supabase.co:5432/postgres")

password = input("\n🔑 Enter your Supabase password (or 'skip' to do it manually): ")

if password and password.lower() != 'skip':
    # URL-encode special characters
    import urllib.parse
    encoded_password = urllib.parse.quote(password, safe='')
    
    connection_string = f"postgresql://postgres:{encoded_password}@db.jsuwszcwmxbfucvsgseh.supabase.co:5432/postgres"
    
    print(f"\n✅ Your connection string is:")
    print(connection_string)
    
    update = input("\nUpdate .env.cloud with this connection string? (y/n): ")
    
    if update.lower() == 'y':
        with open('.env.cloud', 'r') as f:
            lines = f.readlines()
        
        with open('.env.cloud', 'w') as f:
            for line in lines:
                if line.startswith('DATABASE_URL='):
                    f.write(f'DATABASE_URL={connection_string}\n')
                else:
                    f.write(line)
        
        print("✅ Updated .env.cloud successfully!")
        print("\nNow run: python3 setup_cloud_db.py")
    else:
        print("\n📝 Please manually update .env.cloud with:")
        print(f"DATABASE_URL={connection_string}")
else:
    print("\n📝 Please manually update .env.cloud")
    print("Replace the DATABASE_URL line with your actual connection string")