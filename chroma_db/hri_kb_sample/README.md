# HRI KB Sample + Training Set

This bundle includes:
- `sql/hri_kb.sqlite` — minimal KB (papers, sections, chunks)
- `data/chunks.jsonl` — chunked passages
- `data/qa.jsonl` — synthetic Q&A pairs
- `data/triples.jsonl` — anchor/positive/negative triplets for bi-encoder training
- `scripts/train_sbert.py` — quick Sentence-Transformers training script

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch sentence-transformers

# Optional: inspect SQLite
# sqlite3 sql/hri_kb.sqlite 'select title, count(*) from chunks join papers using(paper_id) group by title;'

# Train a small bi-encoder
python scripts/train_sbert.py --triples data/triples.jsonl --epochs 2 --batch_size 64 --out_dir models/hri-bi-encoder
```

## Notes
- Content is paraphrased from *Computational Human-Robot Interaction* (Thomaz, Hoffman, Cakmak, 2016) for demo purposes.
- Expand by adding more PDFs and regenerating chunks/QA/triples.
- For supervised QA fine-tunes (e.g., LoRA on an LLM), adapt `qa.jsonl` into your framework's format (OpenAI, HF, vLLM, etc.).
