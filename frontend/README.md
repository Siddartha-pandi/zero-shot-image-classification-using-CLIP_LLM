# Zero-Shot Image Classification - Frontend

Modern Next.js frontend for the Adaptive CLIP-LLM Framework with automatic prompt engineering and domain adaptation.

## 🚀 Features

- **Upload & Classify**: Drag-and-drop image upload with real-time classification
- **Class Manager**: Add custom classes with domain-specific tuning
- **Domain Hints**: Optional text hints for improved domain detection
- **Rich Results**: 
  - BLIP-generated captions
  - LLM reasoning and explanations
  - Narrative descriptions
  - Top-5 candidates with confidence scores
  - Detected domain information

## 📋 Prerequisites

- Node.js 18+ 
- Backend server running on `http://localhost:8000`

## 🛠️ Setup

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Configure backend URL** (optional):
   
   The default backend URL is `http://localhost:8000`. To change it, create or edit `.env.local`:
   ```env
   NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
   ```

3. **Run development server**:
   ```bash
   npm run dev
   ```

4. **Open in browser**:
   ```
   http://localhost:3000
   ```

## 📁 Project Structure

```
frontend/
├── app/
│   ├── page.tsx              # Home page with features
│   ├── upload/
│   │   └── page.tsx          # Main classification interface
│   └── evaluate/
│       └── page.tsx          # Model evaluation page
├── components/
│   ├── ClassManager.tsx      # Add/manage classes
│   ├── ImageUploadCard.tsx   # Image upload component
│   └── ui/                   # shadcn/ui components
├── types/
│   └── index.ts              # TypeScript type definitions
└── .env.local                # Environment variables
```

## 🎯 Usage

### Adding Classes

1. Navigate to the upload page
2. Find the "Class Manager" card on the left
3. Enter a class name (e.g., "cat", "dog", "car")
4. Select the appropriate domain
5. Click "Add Class"

**Available Domains**:
- Natural Photos
- Medical Images (X-rays, scans)
- Anime/Cartoon
- Sketches (line art)
- Satellite/Aerial
- Unknown (auto-detect)

### Classifying Images

1. Upload an image via drag-and-drop or file picker
2. (Optional) Add a domain hint like "medical X-ray" or "anime character"
3. Click "Classify Image"
4. View results including:
   - Top prediction with confidence
   - BLIP caption
   - LLM reasoning
   - Narrative description
   - Top 5 candidates

## 🔧 Build & Deploy

```bash
# Production build
npm run build

# Start production server
npm start

# Type checking
npm run type-check

# Linting
npm run lint
```

## 🌐 API Integration

The frontend connects to these backend endpoints:

- `GET /api/classes` - List all classes
- `POST /api/add-class` - Add a new class
- `POST /api/classify` - Classify an image

See backend documentation for API details.

## 🎨 Tech Stack

- **Framework**: Next.js 14+ (App Router)
- **UI**: shadcn/ui + Tailwind CSS
- **Icons**: Lucide React
- **TypeScript**: Full type safety
- **State**: React Hooks

## 📝 Notes

- Make sure the backend is running before using the frontend
- Add at least one class before classifying images
- Domain hints improve classification accuracy
- LLM features require Gemini API key in backend

## 🐛 Troubleshooting

**"Failed to fetch classes"**
- Ensure backend is running on port 8000
- Check `NEXT_PUBLIC_BACKEND_URL` in `.env.local`

**"No classes defined" error**
- Add at least one class using the Class Manager
- Or run the backend test script: `python test_add_classes.py`

**Slow initial classification**
- First request loads CLIP and BLIP models (takes time)
- Subsequent requests are much faster
