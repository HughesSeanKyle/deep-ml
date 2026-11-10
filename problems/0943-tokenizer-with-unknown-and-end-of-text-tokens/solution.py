import re

def tokenize(vocab: dict[str, int], text, mode: str):
    """
    vocab: dict[str, int] containing '<|unk|>' and '<|endoftext|>'
    text: str (if mode='encode') or list[int] (if mode='decode')
    mode: 'encode' or 'decode'
    """
    if mode == 'encode':
        # Pattern to capture punctuation, double-dash, or whitespace
        pattern = r'([,.:;?_!"()\']|--|\s)'
        
        # Split text and preserve captured delimiters
        raw_pieces = re.split(pattern, text)
        
        # Filter and strip tokens
        tokens = [piece.strip() for piece in raw_pieces if piece and piece.strip()]
        
        # Fetch the fallback ID for Out-of-Vocabulary (OOV) tokens
        unk_id = vocab['<|unk|>']
        
        # Map tokens to IDs, falling back to <|unk|> if not in vocabulary
        token_ids = [vocab.get(token, unk_id) for token in tokens]
        
        return token_ids
        
    elif mode == 'decode':
        # Create inverse mapping: int ID -> token string
        inv_vocab = {idx: token for token, idx in vocab.items()}
        
        # Convert IDs back to strings
        tokens = [inv_vocab[idx] for idx in text]
        
        # Initial assembly with single space spacing
        joined_text = " ".join(tokens)
        
        # Remove spaces preceding ,.?!"()' as well as :;.
        cleaned_text = re.sub(r'\s+([,.?!"()\':;])', r'\1', joined_text)
        
        return cleaned_text