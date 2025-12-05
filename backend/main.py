"""
FastAPI backend server for the Advanced Zero-Shot Classification Framework.
This version integrates the new advanced features.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import tempfile
import os
import logging

# Import our modules, including the new advanced framework
from models import initialize_models
from advanced_inference import AdvancedZeroShotFramework

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Advanced Zero-Shot Framework", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Component Initialization ---
# Initialize the new advanced framework
classifier = AdvancedZeroShotFramework()

# --- Pydantic Models ---
class ClassificationResponse(BaseModel):
    predictions: Dict[str, float]
    top_prediction: Dict[str, float]
    narrative: str
    domain_info: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting up Advanced Zero-Shot Framework...")
    if not initialize_models():
        raise RuntimeError("Model initialization failed")
    logger.info("Models initialized successfully")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Advanced Zero-Shot Framework API is running"}

@app.post("/api/classify", response_model=ClassificationResponse)
async def classify_image_endpoint(
    file: UploadFile = File(...),
):
    """
    Classify an uploaded image using automatic label generation and scenario description.
    """
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name
        
        # Generate automatic labels based on image analysis
        class_names = classifier.generate_image_labels(temp_path)
        logger.info(f"Auto-generated labels: {class_names}")
        
        # Use the new advanced classifier with automatic domain adaptation
        scores, domain_info = classifier.classify(temp_path, class_names)
        
        # Get top predictions for scenario generation
        top_classes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_labels = [label for label, _ in top_classes]
        
        # Generate detailed scenario description
        domain_desc = classifier.domain_adapter.get_domain_info(domain_info['domain'])['description']
        scenario_prompt = (
            f"Describe this image in detail: The image appears to be from the {domain_info['domain']} domain. "
            f"It likely contains: {', '.join(top_labels)}. Provide a vivid description of the scene, "
            f"including the setting, objects, activities, and atmosphere."
        )
        narrative = classifier.model_manager.generate_narrative(scenario_prompt, max_length=150)

    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    if not scores:
        raise HTTPException(status_code=500, detail="Classification failed to produce scores.")

    top_class = max(scores, key=scores.get)
    top_prediction = {top_class: scores[top_class]}

    return {
        "predictions": scores,
        "top_prediction": top_prediction,
        "narrative": narrative,
        "domain_info": domain_info,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
