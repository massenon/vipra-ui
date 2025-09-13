# core/mllm_integration.py

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import yaml
import json
import os

class MLLMAnalyzer:
    """
    Handles the core analysis by interacting with a multimodal large language model.
    (Implements FR8, FR9, FR10)
    """
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initializes the analyzer by loading the configuration and the MLLM.
        
        Args:
            config_path: Path to the YAML configuration file.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}. Please create it.")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.model_name = config['mllm']['model_name']
        self.device = config['mllm']['device']
        self.max_tokens = config['mllm']['max_new_tokens']
        
        # Check for CUDA availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA is not available. Falling back to CPU. This will be very slow.")
            self.device = 'cpu'

        print(f"Loading MLLM: {self.model_name} onto device: {self.device}...")
        
        # Load model with bfloat16 for better performance on supported GPUs
        torch_dtype = torch.bfloat16 if self.device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        
        print("MLLM loaded successfully.")

    def construct_prompt(self, widget_details: list, review_snippet: str) -> str:
        """
        Constructs the detailed text prompt for the MLLM. (Implements FR8)

        Args:
            widget_details: A list of formatted strings describing annotated widgets.
            review_snippet: The key sentence from the user review.

        Returns:
            The fully formatted prompt string.
        """
        # Convert list of widget details to a numbered string
        widget_str = "\n".join(widget_details) if widget_details else "No widget details extracted."

        # This prompt guides the model to perform a structured analysis and output JSON.
        prompt = f"""
Analyze the provided mobile UI screenshot to determine if there is a mismatch between the visual interface and the user's feedback.

**User Review Snippet:**
"{review_snippet}"

**Annotated Widget Details:**
(Refer to the numbered widgets in the image)
{widget_str}

**Your Task:**
1.  **Identify the User's Core Complaint:** What specific UI element and problem is the user describing?
2.  **Ground the Complaint:** Locate the relevant widget in the image and its details using the provided annotations.
3.  **Analyze for Mismatch:** Compare the user's complaint with the visual and structural evidence. Does the UI imply a functionality that the user reports as broken or missing?
4.  **Provide a Conclusion in JSON format.**

**JSON Output Format:**
Please provide your final analysis ONLY in the following JSON format. Do not add any text before or after the JSON block.
{{
  "mismatch_detected": "Yes" or "No",
  "confidence_score": <A float between 0.0 and 1.0>,
  "mismatch_type": "<'Non-Functional Element', 'Feature Misrepresentation', 'Visual Glitch', or 'None'>",
  "rationale": "<A concise, step-by-step explanation of your reasoning. Start by identifying the user's claim, then ground it to a widget, and finally explain why it is or is not a mismatch.>",
  "relevant_widget_id": <The number of the most relevant widget, or null>
}}
"""
        return prompt

    def analyze_ui_review_pair(self, image_path: str, widget_details: list, review_snippet: str) -> dict:
        """
        Performs the full analysis on a screenshot-review pair. (Implements FR9)

        Args:
            image_path: Path to the visually annotated screenshot.
            widget_details: A list of formatted strings describing annotated widgets.
            review_snippet: The key sentence from the user review.

        Returns:
            A dictionary containing the parsed analysis results.
        """
        if not image_path or not review_snippet:
            return {"error": "Missing image path or review snippet."}

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return {"error": f"Annotated image not found at {image_path}"}
        
        task_prompt = self.construct_prompt(widget_details, review_snippet)
        
        # Florence-2 uses a specific prompt format for this kind of task
        # We combine a task prefix with our detailed prompt
        final_prompt = f"<MORE_DETAILED_OCR>{task_prompt}"

        inputs = self.processor(text=final_prompt, images=image, return_tensors="pt")
        
        # Move inputs to the correct device and set the correct dtype
        torch_dtype = torch.bfloat16 if self.device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
        inputs = {k: v.to(self.device, dtype=torch_dtype if v.dtype.is_floating_point else v.dtype) for k, v in inputs.items()}

        # Generate the output
        try:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False, # Use greedy decoding for consistent results
                num_beams=3 # Use beam search for potentially better quality
            )
            
            # Decode the generated text
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        except Exception as e:
            print(f"An error occurred during MLLM generation: {e}")
            return {"error": "MLLM generation failed.", "details": str(e)}
        
        # Parse the output to extract the JSON part (Implements FR10)
        return self.parse_llm_output(generated_text)

    def parse_llm_output(self, generated_text: str) -> dict:
        """
        Extracts and validates the JSON block from the MLLM's raw output.
        
        Args:
            generated_text: The full text generated by the MLLM.

        Returns:
            A dictionary with the analysis results or an error message.
        """
        try:
            # A more robust way to find the JSON within the generated text
            # The model might sometimes add introductory text before the JSON
            json_str = generated_text[generated_text.find('{') : generated_text.rfind('}')+1]
            
            if not json_str:
                raise ValueError("JSON block not found in the output.")

            parsed_json = json.loads(json_str)

            # --- Basic Validation ---
            required_keys = ["mismatch_detected", "confidence_score", "mismatch_type", "rationale"]
            if not all(key in parsed_json for key in required_keys):
                 raise ValueError("The generated JSON is missing required keys.")

            # Type checking
            if not isinstance(parsed_json["mismatch_detected"], str) or parsed_json["mismatch_detected"].lower() not in ["yes", "no"]:
                raise TypeError("`mismatch_detected` must be 'Yes' or 'No'.")
            if not isinstance(parsed_json["confidence_score"], (int, float)):
                raise TypeError("`confidence_score` must be a number.")
            
            return parsed_json

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"Error parsing MLLM output: {e}")
            print(f"--- Raw MLLM Output ---\n{generated_text}\n-------------------------")
            return {
                "error": "Failed to parse a valid JSON response from the MLLM.",
                "raw_output": generated_text
            }