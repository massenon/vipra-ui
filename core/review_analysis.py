# core/review_analysis.py
import spacy

# Load the spaCy model once for efficiency
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    nlp = None

def preprocess_and_segment_review(text: str) -> list:
    """
    Cleans, normalizes, and segments a review text into sentences. (Implements FR6)
    
    Args:
        text: The raw user review string.
        
    Returns:
        A list of cleaned sentence strings.
    """
    if not nlp or not text:
        return []
        
    doc = nlp(text.lower().strip())
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def extract_functional_snippets(sentences: list) -> str:
    """
    Identifies the most likely sentence describing a functional issue. (Implements FR7)
    This is a simple heuristic. A more advanced system could use a classifier.
    
    Args:
        sentences: A list of sentences from the review.
        
    Returns:
        The single most relevant snippet string, or the first sentence if none are found.
    """
    if not sentences:
        return ""

    keywords = [
        "bug", "crash", "error", "broken", "glitch", "issue", "stuck",
        "not working", "doesn't work", "fails to", "unable to", "can't"
    ]
    
    for sent in sentences:
        if any(kw in sent for kw in keywords):
            return sent # Return the first sentence that contains a keyword
    
    return sentences[0] # Default to the first sentence if no keyword is found