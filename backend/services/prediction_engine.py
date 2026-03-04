# backend/services/prediction_engine.py
import logging
from typing import List, Dict, Tuple
from PIL import Image

from config import MEDICAL_THRESHOLD
from services.domain_detector import get_domain_detector
from services.llm_auto_tuner import get_llm_auto_tuner
from models.clip_model import get_vith14_model
from models.medclip_model import get_medclip_model

logger = logging.getLogger(__name__)

class PredictionEngine:
    def __init__(self):
        self.domain_detector = get_domain_detector()
        self.auto_tuner = get_llm_auto_tuner()
        self.clip = get_vith14_model()
        self.medclip = get_medclip_model()

    def get_dynamic_labels_for_domain(self, domain: str) -> List[str]:
        # Provide base labels directly, allowing dynamic addition in the future
        # In a real dynamic system, these could be loaded from a DB or generated via LLM based on caption
        base_classes = {
            "medical": [
                # Imaging Types
                "chest x-ray", "normal chest x-ray", "radiology image", "MRI scan", "CT scan", "brain MRI",
                "fundus image", "retinal photograph", "ophthalmology", "optic disc",
                
                # Pulmonology (Lung/Chest)
                "pneumonia", "tuberculosis", "lung cancer", "covid-19", "pneumothorax", "pulmonary edema",
                "asthma", "bronchitis", "emphysema", "chronic obstructive pulmonary disease",
                
                # Neurology (Brain)
                "alzheimer's disease", "brain atrophy", "dementia", "neurodegenerative disease",
                "parkinson's disease", "huntington's disease", "multiple sclerosis", "stroke",
                "brain tumor", "glioma", "meningioma", "brain metastasis", "grey matter atrophy",
                "white matter disease", "traumatic brain injury", "brain lesion",
                
                # Ophthalmology (Eye)
                "diabetic retinopathy", "glaucoma", "retinopathy", "eye disease",
                "retinal damage", "age-related macular degeneration", "macular edema",
                "cataract", "retinal detachment", "optic neuritis", "optic disc hemorrhage",
                "cotton wool spots", "hard exudates", "microaneurysms", "branch retinal artery occlusion",
                "central retinal artery occlusion", "diabetic macular edema",
                
                # Orthopedics (Bones/Joints)
                "bone fracture", "osteoarthritis", "rheumatoid arthritis", "bone tumor",
                "osteoporosis", "spinal stenosis", "herniated disc", "ligament tear",
                "meniscus tear", "dislocated joint", "bone metastasis", "bone lesion",
                
                # Oncology (Cancer)
                "leukemia", "lymphoma", "melanoma", "skin cancer", "breast cancer",
                "prostate cancer", "colon cancer", "pancreatic cancer", "liver cancer",
                
                # Gastroenterology
                "ulcer", "gastritis", "colitis", "crohn's disease", "polyp",
                
                # Other Conditions
                "epidermoid cyst", "cyst", "infection", "inflammation", "hemorrhage",
                "edema", "fibrosis", "nodule", "mass", "abscess",
                "normal anatomy", "healthy", "abnormal", "pathological finding"
            ],
            "industrial": [
                "machinery", "factory", "assembly line", "warehouse", 
                "industrial equipment", "manufacturing", "welding", "weld bead",
                "metalwork", "fabrication", "metal joint", "weld defect", 
                "corrosion", "rust", "oxidation", "metal surface",
                "machining", "casting", "forging", "industrial process"
            ],
            "vegetable": [
                "vegetable", "carrot", "broccoli", "tomato", "potato", 
                "leafy green", "root vegetable"
            ],
            "animal": [
                "dog", "cat", "bird", "wildlife", "mammal", "reptile", "animal",
                "lion", "tiger", "bear", "elephant", "giraffe", "zebra", "monkey", "gorilla", "chimpanzee", "ape",
                "wolf", "fox", "deer", "moose", "elk", "rabbit", "hare", "squirrel", "raccoon",
                "horse", "cow", "pig", "sheep", "goat", "chicken", "duck", "goose", "turkey",
                "eagle", "hawk", "owl", "falcon", "penguin", "ostrich", "parrot", "flamingo",
                "snake", "lizard", "turtle", "crocodile", "alligator", "frog", "toad", "salamander",
                "fish", "shark", "whale", "dolphin", "seal", "walrus", "octopus", "squid", "crab", "lobster",
                "spider", "insect", "butterfly", "bee", "ant", "beetle", "fly", "mosquito"
            ],
            "vehicle": [
                "car", "truck", "bus", "motorcycle", "bicycle", "vehicle", "traffic scene"
            ],
            "satellite image": [
                "satellite view", "urban area", "vegetation", "water", 
                "road network", "agricultural land"
            ],
            "fashion item": [
                "dress", "shirt", "pants", "jacket", "shoes", "apparel"
            ],
            "food": [
                "meal", "dish", "dessert", "food ingredients", "fruit"
            ]
        }
        return base_classes.get(domain, ["object", "natural scene", "item"])

    def _merge_labels(self, domain: str) -> List[str]:
        base_labels = self.get_dynamic_labels_for_domain(domain)
        learned_labels = self.auto_tuner.get_learned_labels(domain)
        merged = []
        for label in base_labels + learned_labels:
            key = str(label).strip().lower()
            if key and key not in merged:
                merged.append(key)
        return merged

    def _apply_self_learning_boosts(self, predictions: List[Dict], domain: str) -> List[Dict]:
        if not predictions:
            return predictions

        top_label = str(predictions[0].get("label", "")).lower()
        boosted = []
        max_boost = 0.0
        for pred in predictions:
            label = str(pred.get("label", "")).lower()
            score = float(pred.get("score", 0.0))
            boost = self.auto_tuner.get_correction_boost(domain, top_label, label)
            max_boost = max(max_boost, boost)
            boosted.append({"label": label, "score": score + boost})

        # Normalize more conservatively to preserve confidence gains
        if max_boost > 0:
            # Apply softer normalization when boosts are present
            total = sum(p["score"] for p in boosted)
            if total > 1.0:
                # Only normalize if sum exceeds 1.0
                for pred in boosted:
                    pred["score"] = float(pred["score"] / total)
        else:
            # Standard normalization when no boosts
            total = sum(p["score"] for p in boosted) or 1.0
            for pred in boosted:
                pred["score"] = float(pred["score"] / total)

        boosted.sort(key=lambda item: item["score"], reverse=True)
        return boosted

    def predict(
        self,
        image: Image.Image,
        top_k: int = 3,
        force_domain: str = None,
        caption: str = "",
    ) -> Tuple[List[Dict], str, str, float]:
        """
        Detects domain and runs classification.
        Returns: (predictions, model_used, domain, domain_confidence)
        """
        # 1. Detect Domain
        if force_domain:
            domain = self.auto_tuner.normalize_domain(force_domain)
            domain_conf = 1.0
            all_scores = {domain: 1.0}
        else:
            domain, domain_conf, all_scores = self.domain_detector.detect_domain(image)
            domain = self.auto_tuner.normalize_domain(domain)

        # 2. Select Model based on exact configured logic
        medical_score = all_scores.get("medical", all_scores.get("medical image", 0.0))
        
        if domain == "medical" and medical_score >= MEDICAL_THRESHOLD:
            model_name = "MedCLIP"
            model = self.medclip
            logger.info(f"Routing to MedCLIP for {domain} (score: {medical_score:.3f})")
        else:
            model_name = "ViT-H/14"
            model = self.clip
            logger.info(f"Routing to {model_name} for {domain} (score: {domain_conf:.3f})")

        # 3. Get labels dynamically
        labels = self._merge_labels(domain)

        # 4. Classify
        predictions, _ = model.classify(image, labels, top_k=top_k)
        
        # 5. Confidence Filtering Fallback
        if predictions and predictions[0]['score'] < 0.55:
            logger.info(f"Confidence {predictions[0]['score']:.2f} < 0.55. Applying fallback logic with expanded prompts.")
            
            top_candidate_labels = [p['label'] for p in predictions[:3]]
            custom_ensembles = {}

            llm_ensembles = self.auto_tuner.build_llm_prompt_ensembles(
                domain=domain,
                caption=caption,
                candidate_labels=top_candidate_labels,
            )
            custom_ensembles.update(llm_ensembles)
            
            for label in top_candidate_labels:
                if label in custom_ensembles:
                    continue

                base_prompts = [f"a photo of a {label}", f"an image of a {label}"]
                if domain == "medical":
                    label_lower = label.lower()
                    # Enhanced eye condition prompts
                    if any(term in label_lower for term in ['glaucoma']):
                        base_prompts.extend([
                            f"fundus image showing glaucoma",
                            f"optic disc with glaucomatous cupping",
                            f"ophthalmology image of glaucoma",
                            f"retinal image indicating glaucoma",
                            f"glaucoma with increased intraocular pressure",
                            f"optic nerve head changes"
                        ])
                    elif any(term in label_lower for term in ['diabetic retinopathy', 'retinopathy', 'macular edema']):
                        base_prompts.extend([
                            f"fundus image showing diabetic retinopathy",
                            f"diabetic eye damage",
                            f"diabetic microaneurysms and hemorrhages",
                            f"retinal damage from diabetes",
                            f"cotton wool spots and exudates",
                            f"diabetic retinal changes"
                        ])
                    elif any(term in label_lower for term in ['macular degeneration', 'amd', 'age-related']):
                        base_prompts.extend([
                            f"fundus image with macular degeneration",
                            f"drusen deposits in macula",
                            f"macular atrophy",
                            f"age-related retinal changes"
                        ])
                    elif any(term in label_lower for term in ['cataract']):
                        base_prompts.extend([
                            f"slit lamp photograph of cataract",
                            f"lens opacity from cataract",
                            f"anterior segment imaging",
                            f"cataract maturation"
                        ])
                    elif any(term in label_lower for term in ['retinal detachment']):
                        base_prompts.extend([
                            f"fundus image with retinal detachment",
                            f"separated retinal tissue",
                            f"retinal break imaging",
                            f"detached retina photograph"
                        ])
                    # Enhanced brain condition prompts
                    elif any(term in label_lower for term in ['alzheimer', 'dementia', 'brain atrophy']):
                        base_prompts.extend([
                            f"brain MRI showing {label}",
                            f"neurological scan with {label}",
                            f"brain atrophy indicating {label}",
                            f"MRI demonstrating {label}",
                            f"diagnostic neuroimaging of {label}",
                            f"brain scan revealing {label}"
                        ])
                    elif any(term in label_lower for term in ['parkinson', 'movement disorder']):
                        base_prompts.extend([
                            f"brain imaging showing Parkinson's",
                            f"substantia nigra changes",
                            f"neurological MRI of Parkinson's",
                            f"movement disorder imaging"
                        ])
                    elif any(term in label_lower for term in ['stroke', 'infarct', 'ischemic']):
                        base_prompts.extend([
                            f"brain CT showing acute stroke",
                            f"MRI revealing cerebral infarction",
                            f"ischemic brain lesion",
                            f"stroke imaging"
                        ])
                    elif any(term in label_lower for term in ['glioma', 'tumor', 'brain mass', 'cancer', 'lesion']):
                        base_prompts.extend([
                            f"brain tumor scan showing {label}",
                            f"MRI revealing {label}",
                            f"brain mass indicating {label}",
                            f"pathological brain scan with {label}"
                        ])
                    # Chest/Pulmonary conditions
                    elif any(term in label_lower for term in ['pneumonia', 'tuberculosis', 'covid', 'lung', 'pulmonary', 'thoracic']):
                        base_prompts.extend([
                            f"chest x-ray showing {label}",
                            f"pulmonary imaging of {label}",
                            f"lung infiltrate indicating {label}",
                            f"thoracic radiograph with {label}"
                        ])
                    # Bone/Orthopedic conditions
                    elif any(term in label_lower for term in ['fracture', 'break', 'bone', 'osteoarthritis', 'arthritis', 'skeletal']):
                        base_prompts.extend([
                            f"x-ray showing {label}",
                            f"radiographic image of {label}",
                            f"orthopedic imaging of {label}",
                            f"skeletal radiograph with {label}"
                        ])
                    else:
                        base_prompts.extend([
                            f"a medical {label} radiograph",
                            f"a diagnostic scan showing {label}",
                            f"a radiology scan with {label}",
                            f"a hospital {label} image",
                            f"clinical evidence of {label}"
                        ])
                elif domain == "animal":
                    base_prompts.extend([
                        f"a domestic {label} animal",
                        f"a pet {label}",
                        f"a wild {label} animal",
                        f"a {label} standing",
                        f"a canine or feline {label}",
                        f"a wildlife photo of a {label}"
                    ])
                elif domain == "vegetable":
                    base_prompts.extend([
                        f"a fresh {label} vegetable",
                        f"a farm grown {label}",
                        f"a ripe {label} crop",
                        f"a close up of {label}"
                    ])
                elif domain == "industrial":
                    label_lower = label.lower()
                    if any(term in label_lower for term in ['weld', 'welding', 'joint', 'bead']):
                        base_prompts.extend([
                            f"a close-up of {label}",
                            f"metalwork showing {label}",
                            f"manufacturing process with {label}",
                            f"industrial {label} on metal",
                            f"fabrication showing {label}"
                        ])
                    elif any(term in label_lower for term in ['corrosion', 'rust', 'oxidation', 'defect']):
                        base_prompts.extend([
                            f"metal surface with {label}",
                            f"material showing {label}",
                            f"close-up of {label}",
                            f"industrial {label} damage"
                        ])
                    else:
                        base_prompts.extend([
                            f"an industrial {label} component",
                            f"a mechanical {label}",
                            f"a factory {label} machinery",
                            f"heavy industrial {label}",
                            f"manufacturing {label}"
                        ])
                else:
                    base_prompts.extend([
                        f"a clear {label} object",
                        f"the specific {label} item",
                        f"a beautiful {label}",
                        f"a typical {label} scene"
                    ])
                custom_ensembles[label] = base_prompts
                
            fallback_predictions, _ = model.classify(image, top_candidate_labels, top_k=top_k, custom_ensembles=custom_ensembles)
            logger.info(f"Fallback improved score to {fallback_predictions[0]['score']:.2f} for {fallback_predictions[0]['label']}")
            predictions = fallback_predictions

        predictions = self._apply_self_learning_boosts(predictions, domain)

        # 6. Automatic self-learning from confident predictions.
        if predictions:
            top_label = str(predictions[0].get("label", "")).strip().lower()
            top_score = float(predictions[0].get("score", 0.0))
            self.auto_tuner.auto_reinforce_prediction(
                domain=domain,
                predicted_label=top_label,
                confidence=top_score,
                caption=caption,
            )
        
        return predictions, model_name, domain, domain_conf

_engine = None
def get_prediction_engine() -> PredictionEngine:
    global _engine
    if _engine is None:
        _engine = PredictionEngine()
    return _engine
