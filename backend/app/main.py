# app/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from PIL import Image
import io

from .clip_service import (
    classify_image,
    create_class_prototype,
    list_classes,
)
from .caption_service import generate_caption
from .domain_service import infer_domain_from_hint
from .llm_service import llm_reason_and_label, llm_narrative
from .evaluation_service import evaluate_dataset

app = FastAPI(title="Adaptive CLIP–LLM Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/classes")
def api_classes():
    return {"classes": list_classes()}


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

        # 1) infer domain from hint
        domain = infer_domain_from_hint(user_text)

        # 2) CLIP classification (auto-tuning happens inside)
        cls = classify_image(img, top_k=5)

        # 3) Caption
        caption = generate_caption(img)

        # 4) LLM reasoning + narrative
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

        return {
            "label": reasoning["label"],
            "confidence": cls["confidence"],
            "candidates": cls["candidates"],
            "caption": caption,
            "explanation": reasoning["reason"],
            "narrative": narrative,
            "domain": domain,
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
