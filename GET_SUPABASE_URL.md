# 🚨 IMPORTANT: Get Your Correct Supabase Connection String

## The Problem
The project reference `attaulxkgvidiaoxptqw` cannot be resolved. This means it's either:
- Incomplete (missing characters)
- Incorrect (typo)
- From a deleted/paused project

## ✅ Step-by-Step Guide to Get Correct URL

### 1. Go to Supabase Dashboard
- Open: https://app.supabase.com
- Sign in with your account

### 2. Find Your Project
- You should see your project(s) listed
- Click on the project you want to use

### 3. Get Connection String
- Click **Settings** (gear icon) in left sidebar
- Click **Database** in the settings menu
- Scroll to **Connection string** section
- Look for the **URI** field

### 4. Copy the EXACT String
The URI will look like:
```
postgresql://postgres:[YOUR-PASSWORD]@db.[20-CHARACTER-REF].supabase.co:5432/postgres
```

**IMPORTANT**: The project reference should be about 20 characters long, like:
- `jsuwszcwmxbfucvsgseh` (20 chars)
- `xyzabcdefghijklmnopq` (20 chars)

Your reference `attaulxkgvidiaoxptqw` is only 19 characters - it might be missing a character!

### 5. Common Project References Patterns
Supabase project references are usually:
- Exactly 20 characters
- All lowercase letters
- Random sequence

### 6. What to Look For
In the Supabase dashboard, the connection string section shows:
```
URI
postgresql://postgres:[YOUR-PASSWORD]@db.xxxxxxxxxxxxxxxxxxxx.supabase.co:5432/postgres
```

Copy this ENTIRE string, including all 20 characters of the project reference.

### 7. If You Have Multiple Projects
Make sure you're copying from the RIGHT project:
- Check the project name at the top
- Verify it's the project you just created
- Ensure it shows as "Active" not "Paused"

## 🔄 Alternative: Create New Project
If you can't find the correct reference:

1. Create a NEW Supabase project
2. Name it clearly (e.g., "QuantNexus-Trading")
3. Set password to: `QuantNexus2024`
4. Wait for it to initialize (2-3 minutes)
5. Copy the new connection string

## 📋 Quick Check
Your connection string should have:
- ✅ `postgresql://postgres:` (protocol and user)
- ✅ Your password (or [YOUR-PASSWORD] placeholder)
- ✅ `@db.` (separator)
- ✅ **20 characters** for project reference
- ✅ `.supabase.co:5432/postgres` (domain and database)

## Example of Correct Format
```
postgresql://postgres:QuantNexus2024@db.jsuwszcwmxbfucvsgseh.supabase.co:5432/postgres
                                        ^^^^^^^^^^^^^^^^^^^^
                                        20 characters exactly
```

## Your Current String
```
postgresql://postgres:QuantNexus2024@db.attaulxkgvidiaoxptqw.supabase.co:5432/postgres
                                        ^^^^^^^^^^^^^^^^^^^
                                        Only 19 characters - missing one!
```

## Next Steps
1. Get the correct connection string from Supabase
2. Update `.env.cloud` with it
3. Run: `python setup_cloud_db.py`