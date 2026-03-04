# Zero-Shot Image Classification (CLIP + LLM)

Quick start guide for running the full stack locally.

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm
- GNU Make (recommended on Windows via Git Bash / MSYS2 / WSL)

## 1) Install dependencies

From repo root:

```bash
make install
```

If `make` is not available:

```bash
cd backend
python -m pip install -r requirements.txt
cd ../frontend
npm install
```

## 2) Configure environment variables

### Backend (`backend/.env`)

Create `backend/.env` and add:

```env
GEMINI_API_KEY=your_google_gemini_api_key
ENABLE_LLM_AUTO_TUNING=true
ENABLE_SELF_LEARNING=true
ENABLE_AUTO_SELF_LEARNING=true
AUTO_SELF_LEARNING_THRESHOLD=0.80
SELF_LEARNING_MIN_SUPPORT=2
```

### Frontend (`frontend/.env.local`)

Create `frontend/.env.local` and add:

```env
AUTH_SECRET=your_random_secret
GOOGLE_CLIENT_ID=your_google_oauth_client_id
GOOGLE_CLIENT_SECRET=your_google_oauth_client_secret
NEXTAUTH_URL=http://localhost:3000
```

## 3) Start the app

### Recommended (two terminals)

Terminal 1:

```bash
make start-backend
```

Terminal 2:

```bash
make start-frontend
```

### Without `make`

Terminal 1:

```bash
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2:

```bash
cd frontend
npm run dev
```

## 4) Open URLs

- Frontend: http://localhost:3000
- Home (after sign-in): http://localhost:3000/home
- Upload: http://localhost:3000/upload
- Backend health: http://127.0.0.1:8000/health
- Backend docs: http://127.0.0.1:8000/docs

## 5) Quick checks

```bash
make check-all
```

If `make` is unavailable, verify manually:

- Frontend loads at `http://localhost:3000`
- Backend returns JSON at `http://127.0.0.1:8000/health`

## Notes

- Protected routes (`/home`, `/upload`, `/evaluate`) require Google sign-in.
- On first classification request, model loading can take 1–2 minutes.
- If you get auth errors, confirm `NEXTAUTH_URL`, Google OAuth credentials, and callback URL config in Google Cloud Console.

## Self-Learning API

- `POST /api/feedback` records corrected labels so future predictions can auto-adapt.
- Payload example:

```json
{
	"domain": "animal",
	"predicted_label": "wolf",
	"true_label": "dog",
	"caption": "a brown dog running on grass"
}
```

- Learning data is stored in `backend/auto_learning_feedback.json`.
