import re
from Lectures import transcripts as RAW_TRANSCRIPTS

def _remove_digits_and_colons(text: str) -> str:
    if text is None:
        return text
    cleaned = re.sub(r'[\d:]+', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# Persisted cleaned transcripts dictionary (ready to import)
CLEANED_TRANSCRIPTS = {k: _remove_digits_and_colons(v) for k, v in RAW_TRANSCRIPTS.items()}

__all__ = ["CLEANED_TRANSCRIPTS"]

