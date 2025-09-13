from typing import Dict, List, Tuple
import fitz, re, os, pathlib

HEADING_RE = re.compile(r'^\s*(abstract|introduction|background|methods?|materials and methods|results?|discussion|conclusion|limitations?|references?)\s*[:\.]?\s*$', re.I)

def read_pdf_text_blocks(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    blocks = []
    for page in doc:
        text = page.get_text("text")
        blocks.append(text)
    doc.close()
    return blocks

def extract_title_and_abstract(full_text: str) -> Tuple[str, str]:
    # naive: first non-empty line as title; abstract is block after 'Abstract' heading
    lines = [l.strip() for l in full_text.splitlines() if l.strip()]
    title = lines[0] if lines else None
    # abstract
    abs_text = ""
    m = re.search(r'\bAbstract\b\s*[\n\r]+(.+?)(\n\s*\w+\s*\n|$)', full_text, flags=re.S | re.I)
    if m:
        abs_text = m.group(1).strip()
    return title, abs_text

def split_sections(full_text: str) -> List[Dict]:
    sections = []
    # simple heading-based split
    tokens = full_text.splitlines()
    current_title = "Body"
    current_buf = []

    def flush():
        nonlocal current_title, current_buf
        if current_buf:
            sections.append({"section_title": current_title, "text": "\n".join(current_buf).strip()})
            current_buf = []

    for line in tokens:
        if HEADING_RE.match(line.strip()):
            flush()
            current_title = line.strip().title()
        else:
            current_buf.append(line)
    flush()

    # Trim small sections
    clean = []
    for s in sections:
        txt = re.sub(r'\s+', ' ', s["text"]).strip()
        if len(txt) < 50:
            continue
        s["text"] = txt
        clean.append(s)
    return clean

def extract_citations(full_text: str) -> List[Dict]:
    # two basic patterns: [12] style and (Smith, 2020) style references list
    refs = []
    # try to find 'References' section lines
    m = re.search(r'\bReferences?\b(.+)$', full_text, flags=re.S | re.I)
    if m:
        refs_block = m.group(1)
        lines = [l.strip() for l in refs_block.splitlines() if l.strip()]
        for l in lines:
            year_m = re.search(r'(19|20)\d{2}', l)
            first_author = None
            # author at start until comma
            fa_m = re.match(r'([A-Z][A-Za-z\-\s]+?)[,\.]\s', l)
            if fa_m:
                first_author = fa_m.group(1).strip()
            refs.append({"raw": l, "year": int(year_m.group()) if year_m else None, "first_author": first_author})
    return refs

def parse_pdf(pdf_path: str) -> Dict:
    blocks = read_pdf_text_blocks(pdf_path)
    full_text = "\n".join(blocks)
    title, abstract = extract_title_and_abstract(full_text)
    sections = split_sections(full_text)
    citations = extract_citations(full_text)

    # crude author + year from first page lines (optional heuristics)
    authors = []
    first_page = blocks[0] if blocks else ""
    # try to detect author line (common: names separated by commas before Abstract)
    author_line = None
    for line in first_page.splitlines()[:30]:
        if line.strip() and len(line.split()) <= 20 and not re.search(r'@|http|doi', line, re.I):
            # heuristic: line after title often authors
            if title and line.strip() != title.strip() and len(line) < 120:
                author_line = line.strip()
                break
    if author_line and re.search(r'[A-Za-z]', author_line):
        # split by comma or 'and'
        parts = re.split(r',| and ', author_line)
        authors = [p.strip() for p in parts if p.strip() and len(p.strip()) <= 60]

    doc_info = {
        "title": title,
        "authors": authors or None,
        "abstract": abstract or None,
        "sections": sections,
        "citations": citations,
    }
    return doc_info
