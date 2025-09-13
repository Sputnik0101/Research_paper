import re, math
from typing import List, Dict
import nltk

# Ensure punkt is present (download on first run)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def estimate_tokens(text: str) -> int:
    # crude token estimate ~4 chars/token
    return max(1, math.ceil(len(text) / 4))

def sentence_split(text: str) -> List[str]:
    from nltk.tokenize import sent_tokenize
    # normalize whitespace
    norm = re.sub(r"\s+", " ", text).strip()
    if not norm:
        return []
    return sent_tokenize(norm)

def chunk_text(text: str, max_tokens: int = 350, overlap_tokens: int = 60, use_sentence_merge: bool = True) -> List[Dict]:
    if not text or not text.strip():
        return []
    if not use_sentence_merge:
        # simple hard wrap
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + max_tokens * 4)  # 4 chars per token approx
            piece = text[start:end]
            chunks.append({"text": piece, "token_estimate": estimate_tokens(piece)})
            start = end - min(overlap_tokens * 4, end - start)
        return chunks

    sents = sentence_split(text)
    chunks, cur, cur_tokens = [], [], 0
    for s in sents:
        t = estimate_tokens(s)
        if cur_tokens + t > max_tokens and cur:
            joined = " ".join(cur).strip()
            chunks.append({"text": joined, "token_estimate": estimate_tokens(joined)})
            # overlap by tokens (approximate with sentences)
            cur = [s]  # start next with current sentence
            cur_tokens = t
        else:
            cur.append(s)
            cur_tokens += t

    if cur:
        joined = " ".join(cur).strip()
        chunks.append({"text": joined, "token_estimate": estimate_tokens(joined)})
    return chunks
