# paper_kb — Research Paper → Knowledge Base (Local)

This mini‑system ingests PDF research papers, extracts clean sections & metadata, chunks text, builds embeddings, and stores everything locally for semantic search.

**Why this?**
- Lightweight & local (no external APIs).
- Clean, reproducible pipeline.
- Uses SQLite for structured metadata + ChromaDB for vector search.

## Features
- Parse PDFs with PyMuPDF (fast, reliable)
- Heuristic section splitter (Title, Abstract, Intro, Methods, Results, Discussion, Conclusion, References)
- Reference & inline‑citation detection (basic patterns)
- Sentence‑aware chunking with overlap
- Embeddings with `sentence-transformers` (default: `all-MiniLM-L6-v2`)
- Metadata + sections saved in SQLite; chunks embedded into ChromaDB (persistent disk)
- Simple semantic query script

## Quickstart

```bash
# 1) Create and activate a virtual env (example for Linux/macOS)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Put PDFs inside the ./pdfs folder
mkdir -p pdfs
# copy your papers into ./pdfs

# 4) Ingest
python ingest.py --pdf_dir ./pdfs

# 5) Query
python sample_query.py --q "graph neural networks for materials" --k 5
```

## Project Layout

```
paper_kb/
  ingest.py             # end-to-end orchestrator
  parse_pdf.py          # PDF text + metadata extraction
  chunker.py            # sentence-aware chunking
  embed_store.py        # embeddings + Chroma store
  db.py                 # SQLite schema + helpers
  schema.sql            # tables for papers/sections/citations
  sample_query.py       # demo semantic search
  config.yaml           # tunables
  requirements.txt
  Dockerfile            # optional container
  pdfs/                 # place your PDFs here
  storage/              # SQLite DB + Chroma persistent dir
```

## Notes
- Table extraction is **optional** and can be added via Camelot or Tabula later. This starter focuses on clean text sections first.
- If you later move to Postgres + pgvector, swap `embed_store.py` to push vectors to pgvector and store only IDs in Chroma or replace Chroma entirely.

## License
MIT
