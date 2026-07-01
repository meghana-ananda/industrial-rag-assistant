"""
Hybrid retriever: BM25 (keyword) + FAISS (semantic) fused with Reciprocal Rank Fusion.

RRF score = sum(1 / (rank + k)) across both retrievers, where k=60 dampens outlier ranks.
Top-N documents by combined RRF score are returned.
"""

import re
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(self, vectorstore, k: int = 5, rrf_k: int = 60):
        """
        vectorstore : loaded FAISS vectorstore
        k           : number of docs to return
        rrf_k       : RRF constant (higher = less aggressive rank compression)
        """
        self.vectorstore = vectorstore
        self.k = k
        self.rrf_k = rrf_k

        # Extract all docs from FAISS to build BM25 index
        # FAISS stores docs in docstore._dict
        all_docs = list(vectorstore.docstore._dict.values())
        self.docs = all_docs
        corpus = [_tokenize(doc.page_content) for doc in all_docs]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, threshold: float = 1.0) -> list:
        """Return top-k docs by hybrid RRF score. Threshold filters FAISS results."""
        tokens = _tokenize(query)

        # BM25 rankings (top 20 candidates)
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]

        # FAISS semantic rankings (top 20 candidates, pre-threshold filter)
        faiss_results = self.vectorstore.similarity_search_with_score(query, k=20)
        faiss_docs_ordered = [doc for doc, score in faiss_results if score < threshold]

        # Map FAISS docs back to indices in self.docs by page_content identity
        faiss_indices = []
        for fdoc in faiss_docs_ordered:
            for i, doc in enumerate(self.docs):
                if doc.page_content == fdoc.page_content:
                    faiss_indices.append(i)
                    break

        # Reciprocal Rank Fusion
        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(bm25_top):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + 1 + self.rrf_k)
        for rank, idx in enumerate(faiss_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + 1 + self.rrf_k)

        # Return top-k docs sorted by RRF score
        top_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[: self.k]
        return [self.docs[i] for i in top_indices]
