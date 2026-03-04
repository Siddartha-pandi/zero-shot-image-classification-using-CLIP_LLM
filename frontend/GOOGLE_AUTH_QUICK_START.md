# Quick Start: Google OAuth Setup

## 📋 Checklist

- [ ] Go to https://console.cloud.google.com/
- [ ] Create new project or select existing
- [ ] Enable Google+ API
- [ ] Create OAuth 2.0 credentials (Web application)
- [ ] Add redirect URIs:
  - [ ] `http://localhost:3000/api/auth/callback/google`
  - [ ] `http://localhost:3000` (authorize JavaScript origin)
- [ ] Copy Client ID and Client Secret
- [ ] Generate AUTH_SECRET (see commands below)
- [ ] Update `frontend/.env.local`
- [ ] Restart dev server

## 🔑 Generate AUTH_SECRET

### Windows PowerShell:
```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Linux/Mac:
```bash
openssl rand -base64 32
```

## 📝 Update frontend/.env.local

```
AUTH_SECRET=<paste-generated-secret>
GOOGLE_CLIENT_ID=<paste-your-client-id>
GOOGLE_CLIENT_SECRET=<paste-your-client-secret>
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXTAUTH_URL=http://localhost:3000
```

## 🚀 Start Development

```bash
cd frontend
npm run dev
```

Then visit: http://localhost:3000

## 📍 Sign-In Locations

- **Landing Page**: http://localhost:3000 (main "Sign in with Google" button)
- **Login Page**: http://localhost:3000/auth/login (dedicated login page)

## 🛡️ Protected Routes

After signing in, you can access:
- `/home` - Dashboard
- `/upload` - Image upload
- `/evaluate` - Evaluation tools

## ⚠️ Important

- **DO NOT commit `.env.local`** - it contains secrets
- Use different credentials for production
- Keep `AUTH_SECRET` confidential

## 🆘 Need Help?

See `GOOGLE_AUTH_SETUP.md` for detailed instructions and troubleshooting.
