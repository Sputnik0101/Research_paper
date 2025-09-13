import os, uuid, chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class EmbedStore:
    def __init__(self, chroma_dir: str, collection_name: str, model_name: str):
        os.makedirs(chroma_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_dir, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(collection_name)
        self.model = SentenceTransformer(model_name)

    def add_texts(self, docs: List[Dict]) -> List[str]:
        texts = [d["text"] for d in docs]
        ids = [d.get("id") or str(uuid.uuid4()) for d in docs]
        meta = [d.get("metadata") or {} for d in docs]
        embeds = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.collection.add(ids=ids, embeddings=embeds, documents=texts, metadatas=meta)
        return ids

    def query(self, q: str, k: int = 5):
        qv = self.model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        res = self.collection.query(embedding=qv[0], n_results=k, include=["documents", "metadatas", "distances"])
        return res
