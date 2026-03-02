import pickle
import os
import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class SparseIndexManager:
    """Manages the BM25 keyword index for Hybrid Retrieval."""
    
    def __init__(self, index_path: str = "data/bm25_index.pkl"):
        self.index_path = index_path
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_ids: List[str] = []
        self._load()

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace/lowercase tokenizer for BM25."""
        if not text:
            return []
        # A more advanced implementation might use nltk/spacy or regex
        # For simplicity and speed, we lower, split, and strip basic punctuation.
        import re
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words

    def build_and_save(self, texts: List[str], ids: List[str]):
        """Builds the BM25 index from a list of texts and serializes it to disk."""
        logger.info(f"Building BM25 index for {len(texts)} documents...")
        tokenized_corpus = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_ids = ids
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        with open(self.index_path, 'wb') as f:
            pickle.dump({"bm25": self.bm25, "ids": self.corpus_ids}, f)
            
        logger.info(f"Successfully saved BM25 index to {self.index_path}")

    def _load(self):
        """Loads the BM25 index from disk if it exists."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25 = data["bm25"]
                    self.corpus_ids = data["ids"]
                logger.info(f"Loaded BM25 index with {len(self.corpus_ids)} documents.")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")
        else:
            logger.info("No BM25 index found. Hybrid keyword searches will return empty results until ingestion runs.")

    def search(self, query: str) -> Dict[str, float]:
        """Runs a BM25 keyword search and returns a mapping of document ID -> Score."""
        if not self.bm25 or not self.corpus_ids:
            return {}
            
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Map IDs to their BM25 score.
        # Filter out 0.0 scores to keep things efficient.
        results = {}
        for idx, score in enumerate(doc_scores):
            if score > 0:
                results[self.corpus_ids[idx]] = score
                
        return results

# Singleton instance
sparse_index = SparseIndexManager()
