# app/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from PIL import Image
import io
import logging

from .clip_service import (
    classify_image,
    create_class_prototype,
    list_classes,
    encode_image,
    compute_text_similarity,
    CLASS_PROTOTYPES,
)
from .caption_service import generate_caption
from .domain_service import infer_domain_from_hint, infer_domain
from .llm_service import llm_reason_and_label, llm_narrative, extract_objects
from .evaluation_service import evaluate_dataset

# Import hybrid classification routes
from .routes.classify import router as classify_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hybrid ViT-H/14 + MedCLIP Classification System",
    description="Multi-domain zero-shot image classification with automatic model routing",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include hybrid classification routes
app.include_router(classify_router)


@app.on_event("startup")
async def startup_event():
    """Server startup event"""
    logger.info("=" * 80)
    logger.info("Starting Hybrid ViT-H/14 + MedCLIP Classification System")
    logger.info("=" * 80)
    logger.info("Note: Models will load on first request (lazy loading)")
    logger.info("This may take 1-2 minutes for the first classification")
    logger.info("=" * 80)
    logger.info("Server ready!")
    logger.info("=" * 80)


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "system": "Hybrid ViT-H/14 + MedCLIP",
        "version": "2.0.0"
    }


@app.get("/api/classes")
def api_classes():
    return {"classes": list_classes()}


@app.post("/api/init-medical-classes")
def api_init_medical_classes():
    """Initialize default medical imaging classes."""
    try:
        medical_classes = [
            "lungs",
            "rib cage",
            "pulmonary opacity",
            "heart",
            "chest",
            "normal chest x-ray",
            "pneumonia",
            "pleural effusion",
            "atelectasis",
            "cardiomegaly"
        ]
        
        # Clear existing classes
        CLASS_PROTOTYPES.clear()
        
        # Create prototypes for each medical class
        for label in medical_classes:
            create_class_prototype(label=label, domain="medical", images=None)
        
        return {
            "status": "ok",
            "message": f"Initialized {len(medical_classes)} medical classes",
            "classes": medical_classes
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/init-default-classes")
def api_init_default_classes(domain: str = Form(default="natural")):
    """Initialize default classes for a given domain."""
    try:
        default_classes = {
            "natural": ["dog", "cat", "bird", "car", "tree", "person", "building", "flower"],
            "medical": [
                "lungs", "rib cage", "pulmonary opacity", "heart", "chest",
                "normal chest x-ray", "pneumonia", "pleural effusion", "atelectasis", "cardiomegaly"
            ],
            "anime": ["anime character", "manga character", "cartoon character"],
            "satellite": ["road", "building", "vegetation", "water", "urban area"],
            "sketch": ["portrait sketch", "landscape sketch", "object sketch"]
        }
        
        classes = default_classes.get(domain, default_classes["natural"])
        
        # Clear existing classes
        CLASS_PROTOTYPES.clear()
        
        # Create prototypes for each class
        for label in classes:
            create_class_prototype(label=label, domain=domain, images=None)
        
        return {
            "status": "ok",
            "message": f"Initialized {len(classes)} {domain} classes",
            "classes": classes
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/add-class")
async def api_add_class(
    label: str = Form(...),
    domain: Optional[str] = Form(default="natural"),
    files: Optional[List[UploadFile]] = File(default=None),
):
    try:
        label = label.strip()
        if not label:
            return JSONResponse(status_code=400, content={"error": "Label must not be empty."})

        pil_images: List[Image.Image] = []
        if files:
            for f in files:
                data = await f.read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
                pil_images.append(img)

        info = create_class_prototype(
            label=label,
            domain=domain or "natural",
            images=pil_images if pil_images else None,
        )

        return {
            "status": "ok",
            "label": label,
            "domain": domain,
            "num_images_used": info["num_images"],
            "embedding_norm": info["norm"],
            "message": f"class '{label}' added/updated successfully",
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/classify")
async def api_classify(
    file: UploadFile = File(...),
    user_text: Optional[str] = Form(default=None),
):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # 1) Generate caption first (needed for domain detection)
        caption = generate_caption(img)

        # 2) Infer domain from both caption and user hint
        domain = infer_domain(caption=caption, user_hint=user_text or "")

        # 3) Auto-initialize classes if none exist
        if not CLASS_PROTOTYPES:
            # Initialize based on detected domain
            if domain == "medical":
                medical_classes = [
                    "lungs", "rib cage", "pulmonary opacity", "heart", "chest",
                    "normal chest x-ray", "pneumonia", "pleural effusion", "atelectasis", "cardiomegaly"
                ]
                for label in medical_classes:
                    create_class_prototype(label=label, domain="medical", images=None)
            else:
                # Initialize with general classes
                general_classes = ["dog", "cat", "bird", "car", "tree", "person", "building", "flower"]
                for label in general_classes:
                    create_class_prototype(label=label, domain="natural", images=None)

        # 4) CLIP classification (auto-tuning happens inside)
        cls = classify_image(img, top_k=5)

        # 5) LLM reasoning + narrative
        reasoning = llm_reason_and_label(
            caption=caption,
            candidates=cls["candidates"],
            user_hint=user_text or "",
            domain=domain,
        )
        narrative = llm_narrative(
            caption=caption,
            candidates=cls["candidates"],
            user_hint=user_text or "",
            domain=domain,
        )

        # 6) Extract objects from image
        objects = extract_objects(
            caption=caption,
            candidates=cls["candidates"],
            domain=domain,
        )

        # 7) Compute CLIP similarity scores for validation
        img_vec = encode_image(img)
        domain_similarity = compute_text_similarity(img_vec, domain)
        caption_similarity = compute_text_similarity(img_vec, caption)

        # 8) Compute final confidence using the specified formula:
        # Confidence = 0.6 * DomainSim + 0.4 * CaptionSim
        confidence_score = 0.6 * domain_similarity + 0.4 * caption_similarity

        # 9) Return structured JSON response with all required fields
        return {
            "domain": domain,
            "confidence": confidence_score,
            "objects": objects,
            "caption": caption,
            "explanation": reasoning["reason"],
            "label": reasoning["label"],
            "narrative": narrative,
            "candidates": cls["candidates"],
            "validation": {
                "domain_similarity": domain_similarity,
                "caption_similarity": caption_similarity,
            }
        }
    except RuntimeError as re:
        # e.g. no classes defined
        return JSONResponse(status_code=400, content={"error": str(re)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/evaluate")
async def api_evaluate(
    files: List[UploadFile] = File(...),
    labels: List[str] = Form(...),
):
    """
    Evaluate the model on a test dataset.
    
    Expects:
    - files: List of image files
    - labels: Corresponding ground truth labels (comma-separated or list)
    
    Returns comprehensive metrics including accuracy, precision, recall, F1, mAP, etc.
    """
    try:
        # Parse labels if they come as a single comma-separated string
        if len(labels) == 1 and ',' in labels[0]:
            labels = [l.strip() for l in labels[0].split(',')]
        
        if len(files) != len(labels):
            return JSONResponse(
                status_code=400, 
                content={"error": f"Number of files ({len(files)}) must match number of labels ({len(labels)})"}
            )
        
        # Read all files
        file_data = []
        for f in files:
            contents = await f.read()
            file_data.append((contents, f.filename or "unknown"))
        
        # Evaluate
        metrics = await evaluate_dataset(file_data, labels)
        
        return {
            "status": "ok",
            "metrics": metrics
        }
        
    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
