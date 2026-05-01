# backend/utils/text_extractor.py
import logging
import re
from typing import Dict, List, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Lazy import for easyocr to avoid loading model on startup
_ocr_reader = None

def get_ocr_reader():
    """Lazy load OCR reader"""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            _ocr_reader = easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR: {e}")
            _ocr_reader = False
    return _ocr_reader if _ocr_reader else None

def extract_text_from_image(image: Image.Image) -> str:
    """Extract all text from image using OCR"""
    try:
        reader = get_ocr_reader()
        if not reader:
            logger.debug("OCR reader not available")
            return ""
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Perform OCR
        results = reader.readtext(img_array)
        
        # Extract text
        extracted_text = "\n".join([text[1] for text in results])
        logger.info(f"Extracted text: {extracted_text[:100]}...")
        
        return extracted_text
    except Exception as e:
        logger.warning(f"Text extraction failed: {e}")
        return ""

def parse_document_fields(text: str) -> Dict[str, Optional[str]]:
    """Parse extracted text to identify common document fields"""
    fields = {
        "name": None,
        "id_number": None,
        "date": None,
        "institution": None,
        "expiration": None,
        "blood_group": None,
        "contact": None,
    }
    
    if not text:
        return fields
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    try:
        # Process each line
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # ID Number - look for patterns like "Reg No:", "ID:", "Student ID:", "ID Number:"
            if any(prefix in line_lower for prefix in ['reg no:', 'id:', 'student id:', 'id number:', 'id no.']):
                id_match = re.search(r'[:\s]([A-Z0-9\-]{5,})', line)
                if id_match and not fields["id_number"]:
                    fields["id_number"] = id_match.group(1).strip()
            
            # Date patterns - look for "Date:", "Issue Date:", "Validity:", etc.
            if any(prefix in line_lower for prefix in ['date:', 'issue date:', 'validity:', 'expiration date:']):
                # Match DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD formats
                date_match = re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{2,4}', line)
                if date_match:
                    if 'expir' in line_lower or 'valid' in line_lower:
                        if not fields["expiration"]:
                            fields["expiration"] = date_match.group(0)
                    else:
                        if not fields["date"]:
                            fields["date"] = date_match.group(0)
            
            # Blood group - standalone pattern
            if re.search(r'\b(O|A|B|AB)[-+]?\b', line) and not any(c.isalpha() and c not in 'OAB+-' for c in line.replace(' ', '')):
                blood_match = re.search(r'\b(O|A|B|AB)[-+]?\b', line)
                if blood_match and not fields["blood_group"]:
                    fields["blood_group"] = blood_match.group(0)
            
            # Contact - phone or email patterns
            if re.search(r'[\w\.-]+@[\w\.-]+\.\w+|\+?\d{10,}', line):
                contact_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+|\+?\d{10,}', line)
                if contact_match and not fields["contact"]:
                    fields["contact"] = contact_match.group(0)
            
            # Institution - keywords
            institution_keywords = ['university', 'institute', 'institute', 'college', 'school', 'board', 'council', 'ncrial', 'iit', 'nit']
            if any(keyword in line_lower for keyword in institution_keywords):
                if not fields["institution"]:
                    fields["institution"] = line.strip()
        
        # Try to identify name - usually appears near beginning, is capitalized, has spaces
        for line in lines[:8]:  # Check first 8 lines
            line_stripped = line.strip()
            # Name should: not start with common prefixes, contain mostly letters, possibly have spaces
            if not any(c.isdigit() for c in line_stripped) and len(line_stripped) > 5 and not fields["name"]:
                # Skip common prefixes
                if not any(prefix in line_stripped.lower() for prefix in ['reg', 'id', 'date', 'validity', 'blood', 'contact', 'card', 'certificate']):
                    # Check if looks like a name (capitals and spaces)
                    if (' ' in line_stripped or re.match(r'^[A-Z][a-z]', line_stripped)) and len(line_stripped.split()) > 0:
                        fields["name"] = line_stripped
                        break
        
        logger.info(f"Parsed fields: {fields}")
        return fields
        
    except Exception as e:
        logger.warning(f"Field parsing failed: {e}")
        return fields

def format_extracted_details(fields: Dict[str, Optional[str]]) -> str:
    """Format extracted fields into readable detail string"""
    details = []
    
    if fields.get("name"):
        details.append(f"Name: {fields['name']}")
    if fields.get("id_number"):
        details.append(f"ID Number: {fields['id_number']}")
    if fields.get("date"):
        details.append(f"Date: {fields['date']}")
    if fields.get("institution"):
        details.append(f"Institution: {fields['institution']}")
    if fields.get("expiration"):
        details.append(f"Expiration: {fields['expiration']}")
    if fields.get("blood_group"):
        details.append(f"Blood Group: {fields['blood_group']}")
    if fields.get("contact"):
        details.append(f"Contact: {fields['contact']}")
    
    return " | ".join(details) if details else ""

def extract_and_parse_document_details(image: Image.Image) -> Dict[str, any]:
    """Complete pipeline: extract text -> parse fields -> format"""
    try:
        # Extract text from image
        text = extract_text_from_image(image)
        
        # Parse fields
        fields = parse_document_fields(text)
        
        # Format for display
        formatted_details = format_extracted_details(fields)
        
        return {
            "success": bool(text),
            "raw_text": text,
            "fields": fields,
            "formatted": formatted_details
        }
    except Exception as e:
        logger.error(f"Document detail extraction failed: {e}")
        return {
            "success": False,
            "raw_text": "",
            "fields": {},
            "formatted": ""
        }
