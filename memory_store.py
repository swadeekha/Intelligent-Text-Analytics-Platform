# memory_store.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple

class MemoryStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.texts = []  # list of (id, user, text)
        self.embeddings = None

    def add(self, id_: str, role: str, text: str):
        self.texts.append((id_, role, text))
        emb = self.embedder.encode([text])
        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

    def nearest(self, query: str, top_k=3) -> List[Tuple[float, Tuple[str,str,str]]]:
        if self.embeddings is None:
            return []
        q_emb = self.embedder.encode([query])
        sims = np.dot(self.embeddings, q_emb.T).squeeze()
        idxs = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.texts[i]) for i in idxs]
