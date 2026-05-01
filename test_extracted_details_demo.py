#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced explanation generator
with extracted details (name, ID, date, etc.)
"""
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from utils.text_extractor import extract_and_parse_document_details, format_extracted_details

def demo_extracted_details():
    """Demonstrate the extracted details feature"""
    
    print("\n" + "="*80)
    print("DOCUMENT DETAILS EXTRACTION DEMO")
    print("="*80 + "\n")
    
    # Simulated OCR extraction results
    test_cases = [
        {
            "title": "📋 Student ID Card",
            "raw_text": """NCRIAL
            
SIDDARTHA PANDIYAN
Reg No: 52214118
STUDENT ID CARD

Programmes: B.Tech
Validity: 2022 - 2026

Blood Group: O+
Contact: Contact No: +91-9535301220"""
        },
        {
            "title": "🏥 Medical Certificate",
            "raw_text": """MEDICAL CERTIFICATE

Name: John Smith
ID: DOC-12345
Date: 15-03-2024
Expiration: 15-03-2025
Blood Group: A+"""
        },
        {
            "title": "🎓 University ID",
            "raw_text": """UNIVERSITY IDENTIFICATION CARD

Student Name: Alice Johnson
Student ID: UNI-2024-98765
Issue Date: 01-01-2024
Expiration Date: 31-12-2024
Institution: Stanford University"""
        }
    ]
    
    # Process each test case
    for case in test_cases:
        print(f"\n{case['title']}")
        print("-" * 80)
        print(f"Raw OCR Text:\n{case['raw_text']}\n")
        
        # Simulate extraction (in real use, this comes from image)
        from utils.text_extractor import parse_document_fields, format_extracted_details
        
        fields = parse_document_fields(case['raw_text'])
        formatted = format_extracted_details(fields)
        
        print("✅ EXTRACTED FIELDS:")
        print("-" * 80)
        for key, value in fields.items():
            if value:
                print(f"  {key.upper():20s}: {value}")
        
        print("\n📝 FORMATTED FOR EXPLANATION:")
        print("-" * 80)
        print(f"  {formatted}")
        print()

def demo_with_explanation():
    """Show how details are integrated into explanations"""
    
    print("\n" + "="*80)
    print("EXPLANATION WITH EXTRACTED DETAILS")
    print("="*80 + "\n")
    
    extracted_text = """Name: Siddartha Pandiyan | ID: 52214118 | Date: 2022-2026 | Institution: NCRIAL"""
    
    example_explanation = f"""
EXAMPLE EXPLANATION (with extracted details):

"The image presents a formal student identification card featuring comprehensive institutional documentation. The card displays a professional portrait photograph of student SIDDARTHA PANDIYAN positioned prominently, accompanied by essential identifying information including registration number 52214118, degree program B.Tech, and validity period 2022-2026. The document incorporates distinctive institutional branding with NCRIAL logo and colored accent borders in blue and gold. The card demonstrates quality lamination with protective coating, professional presentation with organized information hierarchy. Observable security features include institutional verification stamps and multiple text field divisions. Additional details indicate blood group O+ and contact information +91-9535301220 for verification purposes. These combined visual indicators including the photographic identification component, structured data layout, institutional branding, security features, color scheme (blue and white), and verification marks collectively establish this as an official student identification card from NCRIAL. The classification model identifies this with 95% confidence, demonstrating strong alignment between observed document characteristics and established institutional ID card patterns and specifications."

Word Count: 155 words ✅
Extracted Details Included: Name ✓, ID Number ✓, Dates ✓, Institution ✓, Blood Group ✓, Contact ✓
    """
    
    print(example_explanation)

def demo_response_format():
    """Show the JSON response format with extracted details"""
    
    print("\n" + "="*80)
    print("FULL API RESPONSE FORMAT")
    print("="*80 + "\n")
    
    import json
    
    response = {
        "domain": "Identification",
        "model_used": "ViT-H/14 CLIP + Gemini Vision",
        "prediction": "Student ID Card",
        "confidence": 0.95,
        "top_predictions": [
            {"label": "Student ID Card", "score": 0.95},
            {"label": "Identity Card", "score": 0.78},
            {"label": "Document", "score": 0.65}
        ],
        "caption": "A photo ID card with student information, showing a portrait photo, name, registration number, degree program, and validity dates",
        "extracted_details": "Name: Siddartha Pandiyan | ID Number: 52214118 | Date: 2022-2026 | Institution: NCRIAL | Blood Group: O+ | Contact: +91-9535301220",
        "explanation": "The image presents a formal student identification card featuring comprehensive institutional documentation. The card displays a professional portrait photograph of student SIDDARTHA PANDIYAN positioned prominently, accompanied by essential identifying information including registration number 52214118, degree program B.Tech, and validity period 2022-2026. The document incorporates distinctive institutional branding with NCRIAL logo... [full explanation continues]"
    }
    
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# ENHANCED EXPLANATION SYSTEM - EXTRACTED DETAILS DEMO")
    print("#"*80)
    
    demo_extracted_details()
    demo_with_explanation()
    demo_response_format()
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE")
    print("="*80)
    print("""
Key Features:
✓ Automatic OCR text extraction from images
✓ Intelligent field parsing (name, ID, dates, institution, etc.)
✓ Extracted details integrated into narrative explanations
✓ Details displayed separately in JSON response
✓ Works with ID cards, certificates, invoices, and documents

Frontend can now display:
1. Classification results
2. Caption
3. Comprehensive explanation (with details mentioned in text)
4. Extracted details (structured, separately for easy display)
    """)
