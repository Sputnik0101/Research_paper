import sqlite3, json, uuid, pathlib
from typing import Dict, List, Optional, Tuple

SQLITE_PATH = None

def get_conn(sqlite_path: str):
    global SQLITE_PATH
    SQLITE_PATH = sqlite_path
    pathlib.Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(conn, schema_sql_path: str):
    with open(schema_sql_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()

def upsert_paper(conn, paper: Dict) -> str:
    paper_id = paper.get("paper_id") or str(uuid.uuid4())
    conn.execute(
        '''INSERT OR REPLACE INTO papers(paper_id, filename, title, authors, published_year, venue, doi, url, abstract)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            paper_id,
            paper.get("filename"),
            paper.get("title"),
            json.dumps(paper.get("authors") or []),
            paper.get("published_year"),
            paper.get("venue"),
            paper.get("doi"),
            paper.get("url"),
            paper.get("abstract"),
        )
    )
    conn.commit()
    return paper_id

def insert_sections(conn, paper_id: str, sections: List[Dict]) -> List[str]:
    ids = []
    for i, s in enumerate(sections):
        sid = s.get("section_id") or str(uuid.uuid4())
        conn.execute(
            '''INSERT INTO sections(section_id, paper_id, section_title, section_order, text)
               VALUES (?, ?, ?, ?, ?)''',
            (sid, paper_id, s.get("section_title"), i, s.get("text"))
        )
        ids.append(sid)
    conn.commit()
    return ids

def insert_citations(conn, paper_id: str, citations: List[Dict]) -> List[str]:
    ids = []
    for c in citations:
        cid = c.get("citation_id") or str(uuid.uuid4())
        conn.execute(
            '''INSERT INTO citations(citation_id, paper_id, raw, year, first_author)
               VALUES (?, ?, ?, ?, ?)''',
            (cid, paper_id, c.get("raw"), c.get("year"), c.get("first_author"))
        )
        ids.append(cid)
    conn.commit()
    return ids

def insert_chunks(conn, paper_id: str, section_id: Optional[str], chunks: List[Dict]) -> List[str]:
    ids = []
    for i, ch in enumerate(chunks):
        cid = ch.get("chunk_id") or str(uuid.uuid4())
        conn.execute(
            '''INSERT INTO chunks(chunk_id, paper_id, section_id, chunk_index, text, token_estimate, embedding_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (cid, paper_id, section_id, i, ch.get("text"), ch.get("token_estimate"), ch.get("embedding_id"))
        )
        ids.append(cid)
    conn.commit()
    return ids
