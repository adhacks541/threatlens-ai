from sentence_transformers import SentenceTransformer
import logging
from functools import lru_cache
from .config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
    @lru_cache(maxsize=1000)
    def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector for a single string."""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding.")
            # Return zero vector if empty string
            return [0.0] * settings.EMBEDDING_DIMENSION
            
        vector = self.model.encode(text)
        return vector.tolist()

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of strings."""
        if not texts:
            return []
            
        vectors = self.model.encode(texts)
        return vectors.tolist()

# Singleton instance for the application
embedding_service = EmbeddingService()
