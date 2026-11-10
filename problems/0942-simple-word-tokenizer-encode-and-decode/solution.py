import re

def encode(text: str, vocab: dict[str, int]) -> list[int]:
    """
    Split text into tokens based on whitespace and punctuation,
    and map them to their integer IDs using the provided vocabulary.
    """
    # Pattern to capture punctuation, double-dash, or whitespace
    pattern = r'([,.:;?_!"()\']|--|\s)'
    
    # Split the text, keeping delimiters caught by the capturing parentheses
    raw_pieces = re.split(pattern, text)
    
    # Clean the pieces and drop empty strings
    tokens = [piece.strip() for piece in raw_pieces if piece and piece.strip()]
    
    # Map the tokens to their corresponding vocabulary IDs
    ids = [vocab[token] for token in tokens]
    
    return ids


def decode(ids: list[int], vocab: dict[str, int]) -> str:
    """
    Reconstruct text from a list of integer IDs using the vocabulary,
    joining with spaces and adjusting spacing before specific punctuation.
    """
    if not ids:
        return ""
        
    # Create the inverse mapping from integer ID to token string
    inv_vocab = {idx: token for token, idx in vocab.items()}
    
    # Map IDs back to tokens
    tokens = [inv_vocab[idx] for idx in ids]
    
    # Initially join tokens with a single space
    joined_text = " ".join(tokens)
    
    # Clean up whitespace before specific punctuation marks ,.?!"()'
    # \1 replaces the match with the captured punctuation mark from group 1
    cleaned_text = re.sub(r'\s+([,.?!"()\'])', r'\1', joined_text)
    
    return cleaned_text