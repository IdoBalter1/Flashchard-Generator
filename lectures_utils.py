import re
from Lectures import transcripts as raw_transcripts

def remove_digits_and_colons(text: str) -> str:
    """
    Remove all digits and colons from the transcript string and collapse whitespace.
    """
    if text is None:
        return text
    # Remove any sequence of digits and colons
    cleaned = re.sub(r'[\d:]+', '', text)
    # Collapse repeated whitespace (including newlines) into single spaces and strip ends
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def get_clean_transcripts() -> dict:
    """
    Return a dictionary mapping lecture number -> cleaned transcript
    (digits and colons removed).
    """
    return {k: remove_digits_and_colons(v) for k, v in raw_transcripts.items()}

if __name__ == "__main__":
    # Quick CLI for testing
    import pprint
    pprint.pprint(get_clean_transcripts())

