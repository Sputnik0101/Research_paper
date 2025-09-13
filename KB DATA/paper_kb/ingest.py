import os, json, uuid, yaml, glob, argparse, pathlib
from tqdm import tqdm
from parse_pdf import parse_pdf
from db import get_conn, init_db, upsert_paper, insert_sections, insert_citations, insert_chunks
from chunker import chunk_text
from embed_store import EmbedStore

def main(pdf_dir: str, config_path: str = "config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    storage_dir = cfg["storage_dir"]
    pathlib.Path(storage_dir).mkdir(parents=True, exist_ok=True)

    conn = get_conn(cfg["sqlite_path"])
    init_db(conn, "schema.sql")

    embedder = EmbedStore(cfg["chroma_dir"], cfg["collection_name"], cfg["embedding"]["model"])

    paths = sorted(glob.glob(os.path.join(pdf_dir, cfg["pdf_glob"])))
    for p in tqdm(paths, desc="Ingesting PDFs"):
        info = parse_pdf(p)
        paper = {
            "filename": os.path.basename(p),
            "title": info.get("title"),
            "authors": info.get("authors") or [],
            "abstract": info.get("abstract"),
            "published_year": None,
            "venue": None,
            "doi": None,
            "url": None,
        }
        paper_id = upsert_paper(conn, paper)

        # sections
        section_ids = insert_sections(conn, paper_id, info.get("sections") or [])

        # citations
        insert_citations(conn, paper_id, info.get("citations") or [])

        # chunks + embeddings
        # include abstract as a pseudo-section
        all_sections = list(info.get("sections") or [])
        if info.get("abstract"):
            all_sections = [{"section_title": "Abstract", "text": info["abstract"]}] + all_sections

        for idx, sec in enumerate(all_sections):
            chunks = chunk_text(sec["text"],
                                max_tokens=cfg["chunk"]["max_tokens"],
                                overlap_tokens=cfg["chunk"]["overlap_tokens"],
                                use_sentence_merge=cfg["chunk"]["use_sentence_merge"])
            # push to chroma
            docs = [{
                "id": str(uuid.uuid4()),
                "text": ch["text"],
                "metadata": {
                    "paper_id": paper_id,
                    "section_title": sec.get("section_title"),
                    "source_filename": os.path.basename(p),
                    "chunk_index": i
                }
            } for i, ch in enumerate(chunks)]
            ids = embedder.add_texts(docs)

            # record in sqlite
            sqlite_chunks = []
            for i, ch in enumerate(chunks):
                sqlite_chunks.append({
                    "text": ch["text"],
                    "token_estimate": ch["token_estimate"],
                    "embedding_id": ids[i],
                })
            sid = None
            if idx < len(section_ids):  # because we prepended Abstract
                sid = section_ids[idx] if info.get("abstract") is None else (None if idx == 0 else section_ids[idx-1])
            insert_chunks(conn, paper_id, sid, sqlite_chunks)

    print(f"Done. SQLite at {cfg['sqlite_path']}; Chroma at {cfg['chroma_dir']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="Directory containing PDFs")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = ap.parse_args()
    main(args.pdf_dir, args.config)
