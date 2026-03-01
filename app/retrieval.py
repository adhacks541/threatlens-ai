import logging
import requests
from typing import List, Dict, Any, Optional
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
from .embeddings import embedding_service
from .config import settings

logger = logging.getLogger(__name__)


# -----------------------------
# Evaluation Metric
# -----------------------------

def measure_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    if not retrieved_ids or k == 0:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_retrieved = set(top_k).intersection(set(relevant_ids))
    return len(relevant_retrieved) / k


# -----------------------------
# HTTP Helpers
# -----------------------------

def _get_headers() -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    if settings.ENDEE_AUTH_TOKEN:
        headers["Authorization"] = settings.ENDEE_AUTH_TOKEN

    return headers


# -----------------------------
# Retrieval Function
# -----------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
@lru_cache(maxsize=100)
def retrieve_documents(
    query: str,
    top_k: Optional[int] = None,
    metadata_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    
    actual_top_k = top_k if top_k is not None else settings.DEFAULT_TOP_K
    logger.info(f"Retrieving top {actual_top_k} documents for query: '{query}'")

    # 1️⃣ Generate embedding
    query_vector = embedding_service.generate_embedding(query)

    if len(query_vector) != settings.EMBEDDING_DIMENSION:
        logger.error(f"Embedding dimension mismatch. Expected {settings.EMBEDDING_DIMENSION}, got {len(query_vector)}")
        raise ValueError("Query embedding dimension mismatch.")

    # 2️⃣ Build request payload
    payload: Dict[str, Any] = {
        "vector": [float(x) for x in query_vector],
        "k": actual_top_k
    }

    if metadata_filter:
        payload["filter"] = metadata_filter

    url = f"{settings.ENDEE_URL}/api/v1/index/{settings.ENDEE_COLLECTION_NAME}/search"

    # 3️⃣ Send request
    try:
        response = requests.post(
            url,
            json=payload,
            headers=_get_headers(),
            timeout=10 # Reasonable timeout for search
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during retrieval: {e}")
        raise RuntimeError(f"Failed to connect to Endee vector database: {e}")

    if response.status_code != 200:
        logger.error(f"Search failed [{response.status_code}]: {response.text}")
        raise RuntimeError(
            f"Search failed [{response.status_code}]: {response.text}"
        )

    hits = response.json()
    
    # Handle case where searching empty collection or no results
    if not hits:
        logger.info("Search returned no results.")
        return []

    # 4️⃣ Parse results
    results = []
    for hit in hits:
        results.append({
            "id": hit.get("id"),
            "score": hit.get("score"),
            "content": hit.get("meta", ""),
            "filter": hit.get("filter", "")
        })

    logger.info(f"Retrieved {len(results)} documents.")
    return results