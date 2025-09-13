import argparse, yaml, sqlite3, json
from embed_store import EmbedStore

def main(q: str, k: int, config_path: str = "config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    es = EmbedStore(cfg["chroma_dir"], cfg["collection_name"], cfg["embedding"]["model"])
    res = es.query(q, k=k)

    print(f"Query: {q}\n")
    for i in range(len(res['ids'][0])):
        meta = res['metadatas'][0][i]
        doc = res['documents'][0][i]
        dist = res['distances'][0][i]
        print(f"[{i+1}] paper_id={meta.get('paper_id')} section={meta.get('section_title')} chunk={meta.get('chunk_index')} dist={dist:.4f}")
        print(doc[:500].replace("\n", " ") + ("..." if len(doc) > 500 else ""))
        print("-" * 80)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="query text")
    ap.add_argument("--k", type=int, default=5, help="top-k results")
    ap.add_argument("--config", default="config.yaml", help="config path")
    args = ap.parse_args()
    main(args.q, args.k, args.config)
