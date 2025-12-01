# Frontend - Zero-Shot Image Classification

Modern React frontend built with Next.js for zero-shot image classification.

## � Key Features

- **Image Upload**: Drag & drop or click to upload
- **Label Input**: Dynamic label management
- **Results Display**: Detailed classification results
- **AI Reasoning**: Enhanced explanations from LLM
- **Responsive Design**: Works on all devices

## 🚀 Quick Start

```bash
npm install
npm run dev
```

Access the application at `http://localhost:3000`

## 📁 Project Structure

```
zero-shot/
├── app/                    # Next.js frontend application
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # App layout
│   └── page.tsx           # Main page
├── backend/               # Python FastAPI backend
│   ├── main.py           # FastAPI server
│   ├── inference.py      # Enhanced classification logic
│   ├── models.py         # Model management
│   ├── domain_adaptation.py # Domain adaptation utilities
│   ├── evaluation.py     # Model evaluation tools
│   └── requirements.txt  # Python dependencies
├── components/           # React components
├── lib/                 # Utility functions
├── types/               # TypeScript type definitions
└── public/              # Static assets
```

## 🛠️ Technology Stack

### Frontend
- **Next.js 15** - React framework with Turbopack
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Lucide React** - Icons

### Backend  
- **FastAPI** - Python web framework
- **CLIP (ViT-B/32)** - Vision-language model
- **DistilGPT-2** - Language model for reasoning
- **PyTorch** - Deep learning framework
- **PIL** - Image processing

## 📖 Usage

1. **Start the Backend**: Run `python main.py` in the backend directory
2. **Start the Frontend**: Run `npm run dev` in the root directory  
3. **Upload Image**: Use the web interface to upload an image
4. **Add Labels**: Enter classification labels separated by commas
5. **Classify**: Click classify to get results with detailed AI reasoning

## 🔧 Configuration

The application works out of the box with default settings. Models are downloaded automatically on first run.

## 📄 License

This project is for educational purposes.