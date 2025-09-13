# app.py
import gradio as gr
import os
import json

# Import our core ViPRA-UI modules
from core.gui_analysis import parse_xml_hierarchy, annotate_screenshot
from core.review_analysis import preprocess_and_segment_review, extract_functional_snippets
from core.mllm_integration import MLLMAnalyzer

print("Initializing ViPRA-UI System...")
# Initialize the MLLM Analyzer once when the app starts
try:
    analyzer = MLLMAnalyzer(config_path='config.yaml')
    print("ViPRA-UI System Initialized Successfully.")
except Exception as e:
    print(f"FATAL: Could not initialize MLLMAnalyzer. Error: {e}")
    analyzer = None

def run_vipra_ui_analysis(screenshot_image, review_text, xml_file):
    """
    The main function that orchestrates the entire ViPRA-UI pipeline.
    This function is called by the Gradio interface.
    """
    if not analyzer:
        raise gr.Error("MLLM Analyzer is not available. Check server logs for initialization errors.")
        
    if screenshot_image is None or not review_text.strip():
        raise gr.Error("Please upload a screenshot and provide a user review.")

    # --- Step 1: Save inputs to temporary files ---
    # Gradio provides inputs as temporary file paths or objects
    screenshot_path = screenshot_image.name
    
    xml_path = None
    if xml_file is not None:
        xml_path = xml_file.name
    
    # --- Step 2: Run Review Analysis ---
    sentences = preprocess_and_segment_review(review_text)
    functional_snippet = extract_functional_snippets(sentences)
    
    # --- Step 3: Run GUI Analysis ---
    # If no XML is provided, we can still proceed but with less structural info
    ui_elements = []
    if xml_path:
        ui_elements = parse_xml_hierarchy(xml_path)
    
    # Annotate the screenshot for visual grounding
    annotated_image_path, widget_details = annotate_screenshot(screenshot_path, ui_elements)
    
    if not annotated_image_path:
        raise gr.Error("Failed to process the uploaded screenshot. It might be corrupted.")

    # --- Step 4: Run MLLM Analysis ---
    print("Performing MLLM analysis...")
    analysis_result = analyzer.analyze_ui_review_pair(
        image_path=annotated_image_path,
        widget_details=widget_details,
        review_snippet=functional_snippet
    )
    print("Analysis complete.")

    # --- Step 5: Format and return results for Gradio UI ---
    # The result is expected to be a dictionary (or a dict with an 'error' key)
    if "error" in analysis_result:
        error_message = analysis_result.get("error", "An unknown error occurred.")
        raw_output = analysis_result.get("raw_output", "")
        raise gr.Error(f"{error_message}\n\nRaw Output: {raw_output}")

    # Pretty-print the JSON for better display
    formatted_json = json.dumps(analysis_result, indent=2)
    
    return annotated_image_path, formatted_json

# --- Build the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="ViPRA-UI Demo") as demo:
    gr.Markdown(
        """
        # ViPRA-UI: Automated UI-User Review Mismatch Detection
        Upload a mobile application screenshot, provide the corresponding user review, and optionally include the XML View Hierarchy to detect potential mismatches.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Inputs")
            screenshot_input = gr.Image(type="pil", label="Mobile UI Screenshot*")
            review_input = gr.Textbox(lines=5, label="User Review Text*", placeholder="e.g., The login button is broken, it does nothing when I tap it.")
            xml_input = gr.File(label="XML View Hierarchy (Optional)", file_types=[".xml"])
            
            analyze_button = gr.Button("Analyze for Mismatch", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### 2. Analysis Results")
            annotated_output_image = gr.Image(label="Annotated Screenshot with Grounding")
            json_output = gr.JSON(label="MLLM Analysis Result")

    # Define the action when the button is clicked
    analyze_button.click(
        fn=run_vipra_ui_analysis,
        inputs=[screenshot_input, review_input, xml_input],
        outputs=[annotated_output_image, json_output]
    )
    
    gr.Markdown("---")
    gr.Markdown("### Example Usage")
    gr.Examples(
        examples=[
            ["examples/example_login_fail.png", "The 'Login' button is broken, it does nothing when I tap it.", "examples/example_login_fail.xml"],
            ["examples/example_missing_feature.png", "I can't find the dark mode setting anywhere, even though the app store description said it was available.", "examples/example_missing_feature.xml"],
        ],
        inputs=[screenshot_input, review_input, xml_input]
    )

# --- Launch the application ---
if __name__ == "__main__":
    # Create a dummy example directory for Gradio
    os.makedirs("examples", exist_ok=True)
    # You would place your example images/xmls in this 'examples' folder
    
    print("Launching ViPRA-UI Gradio Demo...")
    demo.launch()