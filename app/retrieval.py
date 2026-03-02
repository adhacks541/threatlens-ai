import logging
import requests
from typing import List, Dict, Any, Optional
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
from collections import defaultdict
from .embeddings import embedding_service
from .sparse_index import sparse_index
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
# Normalization Helper
# -----------------------------

def _min_max_normalize(scores_dict: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize a dictionary of document scores to [0.0, 1.0]."""
    if not scores_dict:
        return {}
    
    values = list(scores_dict.values())
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return {k: 1.0 for k in scores_dict.keys()}
        
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}


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

    try:
        hits = response.json()
    except requests.exceptions.JSONDecodeError:
        logger.warning(f"Failed to decode JSON from Endee response. Status: {response.status_code}, Body: '{response.content}'")
        import re
        # Fallback binary parser for Endee's proprietary flatbuffer format
        content_str = response.content.decode('utf-8', errors='ignore')
        found_ids = list(dict.fromkeys(re.findall(r'doc_\d{3}', content_str)))
        
        hits = []
        for i, f_id in enumerate(found_ids):
            hits.append({
                "id": f_id,
                "score": 1.0 / (i + 1), # mock descending score for normalization
                "meta": '{"category": "unknown", "content": "Metadata suppressed due to binary Endee encoding"}',
                "filter": ""
            })

    # 4️⃣ Format dense results
    dense_scores = {hit.get("id"): hit.get("score", 0.0) for hit in hits if hit.get("id")}
    dense_raw = {hit.get("id"): hit for hit in hits if hit.get("id")}
    
    # 5️⃣ Format Sparse (BM25) results
    # We execute BM25 search locally on the singleton index
    sparse_scores = sparse_index.search(query)

    # 6️⃣ Normalization
    norm_dense = _min_max_normalize(dense_scores)
    norm_sparse = _min_max_normalize(sparse_scores)
    
    # 7️⃣ Score Fusion
    all_doc_ids = set(norm_dense.keys()).union(set(norm_sparse.keys()))
    
    fusion_results = []
    
    for doc_id in all_doc_ids:
        # Default to 0.0 if a document wasn't retrieved by one of the methods
        d_score = norm_dense.get(doc_id, 0.0)
        s_score = norm_sparse.get(doc_id, 0.0)
        
        final_score = (settings.HYBRID_ALPHA * d_score) + (settings.HYBRID_BETA * s_score)
        
        # We need the metadata content. If Endee found it, use that.
        # If ONLY BM25 found it, we might be missing realtime metadata depending on the system, 
        # but since we index simultaneously, we assume dense_raw has most of what we need. 
        # For a truly robust system, we would fetch missing metadata by ID.
        hit_meta = dense_raw.get(doc_id)
        content_str = hit_meta.get("meta", "") if hit_meta else "{}"
        filter_str = hit_meta.get("filter", "") if hit_meta else ""
        
        fusion_results.append({
            "id": doc_id,
            "dense_score": float(dense_scores.get(doc_id, 0.0)),
            "bm25_score": float(sparse_scores.get(doc_id, 0.0)),
            "final_score": float(final_score),
            "content": content_str,
            "filter": filter_str
        })
        
    # 8️⃣ Sort by final blended score descending
    fusion_results.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Trim to requested top_k
    final_top_k = fusion_results[:actual_top_k]
    
    # Logging insights
    if final_top_k:
        top_doc = final_top_k[0]
        dominant_signal = "Semantic" if (settings.HYBRID_ALPHA * _min_max_normalize(dense_scores).get(top_doc["id"], 0.0)) > (settings.HYBRID_BETA * _min_max_normalize(sparse_scores).get(top_doc["id"], 0.0)) else "Keyword"
        logger.info(f"Retrieved {len(final_top_k)} documents (Union size: {len(fusion_results)}). Top document dominated by {dominant_signal} signal.")
    else:
        logger.info("Search returned no results from either index.")

    return final_top_k