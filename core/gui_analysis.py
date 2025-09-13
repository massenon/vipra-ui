# core/gui_analysis.py

import cv2
import xml.etree.ElementTree as ET
import os
from PIL import Image

# Note: We will handle the OCR part differently with the Florence-2 model,
# as it performs OCR as part of its analysis. This simplifies the pipeline.
# This module will now focus on XML parsing and visual annotation.

def parse_xml_hierarchy(xml_path: str) -> list:
    """
    Parses a View Hierarchy XML to extract UI element properties. (Implements FR3)
    
    Args:
        xml_path: Path to the XML file.
    
    Returns:
        A list of dictionaries, where each dictionary represents a UI element.
    """
    if not os.path.exists(xml_path):
        print(f"Warning: XML file not found at {xml_path}")
        return []
        
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"Warning: Could not parse XML file at {xml_path}")
        return []
        
    elements = []
    for node in root.iter():
        if 'bounds' in node.attrib:
            elements.append({
                'class': node.attrib.get('class', 'N/A'),
                'text': node.attrib.get('text', ''),
                'resource-id': node.attrib.get('resource-id', ''),
                'clickable': node.attrib.get('clickable', 'false'),
                'bounds': node.attrib.get('bounds')
            })
    return elements

def annotate_screenshot(image_path: str, elements: list) -> str:
    """
    Draws bounding boxes and labels on a screenshot for visual grounding. (Implements FR5)
    
    Args:
        image_path: Path to the original screenshot.
        elements: A list of UI elements from parse_xml_hierarchy.
    
    Returns:
        The file path of the saved annotated image.
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found at {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image at {image_path}")
        return None

    label_counter = 1
    # Create a list to hold details of annotated widgets for the prompt
    annotated_widget_details = []

    for el in elements:
        # We annotate clickable or important elements
        is_clickable = el.get('clickable') == 'true'
        has_text = el.get('text', '') != ''
        
        if is_clickable or has_text:
            try:
                # Parse bounds like "[x1,y1][x2,y2]"
                coords_str = el['bounds'].replace('][', ',').strip('[]')
                coords = [int(c) for c in coords_str.split(',')]
                (x1, y1, x2, y2) = coords

                # Draw green box for clickable elements, blue for others with text
                color = (0, 255, 0) if is_clickable else (255, 0, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3) # Thicker line
                
                # Add numerical label in a filled rectangle for visibility
                label_text = str(label_counter)
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1 - 5), color, -1)
                cv2.putText(img, label_text