-- SQLite schema for papers & sections
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS papers (
  paper_id TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  title TEXT,
  authors TEXT,          -- JSON array of strings
  published_year INTEGER,
  venue TEXT,
  doi TEXT,
  url TEXT,
  abstract TEXT,
  created_ts DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sections (
  section_id TEXT PRIMARY KEY,
  paper_id TEXT NOT NULL,
  section_title TEXT,
  section_order INTEGER,
  text TEXT,
  FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS citations (
  citation_id TEXT PRIMARY KEY,
  paper_id TEXT NOT NULL,
  raw TEXT,           -- raw reference line
  year INTEGER,
  first_author TEXT,
  FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  paper_id TEXT NOT NULL,
  section_id TEXT,
  chunk_index INTEGER,
  text TEXT,
  token_estimate INTEGER,
  embedding_id TEXT,     -- chroma id to back-reference
  FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE,
  FOREIGN KEY (section_id) REFERENCES sections(section_id) ON DELETE CASCADE
);
