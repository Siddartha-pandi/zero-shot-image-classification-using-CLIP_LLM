# backend/config.py
from typing import Dict, List

# Configuration for dynamic domains
# We can easily add new domains here without changing model logic

DOMAINS: List[str] = [
    "animal",
    "industrial",
    "medical",
    "vegetable",
    "vehicle",
    "satellite image",
    "food",
    "fashion item"
]

# Keywords for text embedding when matching domain
DOMAIN_PROMPTS: Dict[str, str] = {
    "animal": "an image of an animal, wildlife, or pet",
    "industrial": "an image of industrial machinery, welding, metalwork, manufacturing, factory equipment, or mechanical component",
    "medical": "a medical radiology image such as x-ray, MRI, or CT scan",
    "vegetable": "an image of a vegetable, plant, or crop",
    "vehicle": "an image of a vehicle, car, truck, or traffic scene",
    "satellite image": "a satellite or aerial top view image",
    "food": "an image of food, meal, or dish",
    "fashion item": "an image of clothing, fashion item, or apparel"
}

# Thresholds
MEDICAL_THRESHOLD = 0.08
TRAFFIC_THRESHOLD = 0.35
