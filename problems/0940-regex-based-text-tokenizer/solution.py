import re

def tokenize_text(text: str) -> list:
    """
    Split raw text into tokens using regex-based splitting on whitespace
    and punctuation. Returns a list of non-empty stripped tokens.
    """
    # Pattern captures the double-dash or individual punctuation marks
    # and splits on whitespace as well
    pattern = r'(--|[\s,.:;?_!"()\'\-])'
    
    # re.split captures the matches in parentheses, keeping them as tokens
    raw_tokens = re.split(pattern, text)
    
    # Strip whitespace from each token and filter out empty strings
    tokens = [t.strip() for t in raw_tokens if t and t.strip()]
    
    return tokens