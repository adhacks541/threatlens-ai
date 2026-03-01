import json
import logging
import requests
from typing import List, Dict, Any
from .embeddings import embedding_service
from .config import settings
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def get_headers():
    headers = {
        "Content-Type": "application/json"
    }
    if settings.ENDEE_AUTH_TOKEN:
        headers["Authorization"] = settings.ENDEE_AUTH_TOKEN
    return headers


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def create_index():
    """
    Create index in Endee if it does not exist.
    Uses correct parameters from Endee C++ source:
    - index_name
    - dim
    - space_type
    - precision (optional but explicitly set to avoid quantization issues)
    """

    url = f"{settings.ENDEE_URL}/api/v1/index/create"

    payload = {
        "index_name": settings.ENDEE_COLLECTION_NAME,
        "dim": settings.EMBEDDING_DIMENSION,   # Must match embedding size (384)
        "space_type": "cosine",                # Valid: cosine / l2 / ip
        "precision": "float32"                 # Avoid default int8 quantization
    }

    response = requests.post(
        url,
        json=payload,
        headers=get_headers(),
        timeout=20
    )

    if response.status_code not in (200, 201):
        # If index already exists, don't crash
        if "already exists" in response.text.lower():
            logger.info("Index already exists.")
        else:
            raise RuntimeError(
                f"Index creation failed [{response.status_code}]: {response.text}"
            )
    else:
        logger.info("Index created successfully.")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def insert_vectors(ids, vectors, metadata):
    url = f"{settings.ENDEE_URL}/api/v1/index/{settings.ENDEE_COLLECTION_NAME}/vector/insert"

    payload = [
        {
            "id": str(ids[i]),
            "vector": [float(x) for x in vectors[i]],  # <-- FORCE float
            "meta": json.dumps(metadata[i])
        }
        for i in range(len(ids))
    ]

    response = requests.post(
        url,
        json=payload,
        headers=get_headers(),
        timeout=30
    )

    if response.status_code != 200:
        logger.error(f"Server response during insertion: {response.text}")
        raise RuntimeError(
            f"Insertion failed [{response.status_code}]: {response.text}"
        )

    logger.info("Vectors inserted successfully.")


def ingest_data(file_path: str = "data/sample_threat_data.json"):
    """
    Load threat intelligence JSON,
    generate embeddings,
    create index,
    insert vectors.
    """

    logger.info(f"Loading data from {file_path}")

    with open(file_path, "r") as f:
        documents: List[Dict[str, Any]] = json.load(f)

    if not documents:
        logger.warning("No documents found.")
        return 0

    texts = [doc["content"] for doc in documents]
    ids = [doc["id"] for doc in documents]

    metadata = [
        {
            "category": doc["category"],
            "severity": doc["severity"],
            "source": doc["source"],
            "date": doc["date"],
            "content": doc["content"]
        }
        for doc in documents
    ]

    logger.info("Generating embeddings...")
    vectors = embedding_service.generate_embeddings_batch(texts)

    if len(vectors[0]) != settings.EMBEDDING_DIMENSION:
        raise RuntimeError("Embedding dimension mismatch.")

    logger.info("Ensuring index exists...")
    create_index()

    logger.info("Inserting vectors...")
    insert_vectors(ids, vectors, metadata)

    logger.info("Ingestion completed successfully.")
    return len(ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        count = ingest_data()
        logger.info(f"Ingested {count} documents successfully.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")