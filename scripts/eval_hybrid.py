import sys
import os
import requests

# Ensure we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.retrieval import retrieve_documents, measure_precision_at_k
from app.config import settings

def run_evaluation():
    print("--- ThreatLens Hybrid vs Dense Evaluation ---")
    
    # We simulate a "CVE search", where hybrid Shines because dense semantic embeddings
    # often fail to perfectly match specific serial/CVE formats compared to BM25 keywords.
    query = "CVE-2021-44228 Log4Shell vulnerabilities"
    relevant_ground_truth = ["doc_003"] # Assuming doc_003 is exactly about log4j in the sample set.
    
    k = 3
    
    # 1. Run Baseline Dense Evaluation
    # Temporarily set hybrid config so it acts purely as dense
    original_alpha = settings.HYBRID_ALPHA
    original_beta = settings.HYBRID_BETA
    
    # Simulate Dense Only
    settings.HYBRID_ALPHA = 1.0
    settings.HYBRID_BETA = 0.0
    
    dense_docs = retrieve_documents(query=query, top_k=k)
    dense_ids = [d["id"] for d in dense_docs]
    dense_p_at_k = measure_precision_at_k(dense_ids, relevant_ground_truth, k)
    
    print("\n[Baseline] Dense Search Only (Alpha=1.0, Beta=0.0):")
    for d in dense_docs:
        print(f"  -> ID: {d['id']} | Semantic Score: {d['final_score']:.4f}")
    print(f"Dense Precision@{k}: {dense_p_at_k:.2f}")
    
    # 2. Run Hybrid Evaluation
    settings.HYBRID_ALPHA = original_alpha
    settings.HYBRID_BETA = original_beta
    
    hybrid_docs = retrieve_documents(query=query, top_k=k)
    hybrid_ids = [d["id"] for d in hybrid_docs]
    hybrid_p_at_k = measure_precision_at_k(hybrid_ids, relevant_ground_truth, k)
    
    print(f"\n[Proposed] Hybrid Search (Alpha={original_alpha}, Beta={original_beta}):")
    for d in hybrid_docs:
        print(f"  -> ID: {d['id']} | Blended Score: {d['final_score']:.4f} (D:{d['dense_score']:.2f}, S:{d['bm25_score']:.2f})")
    print(f"Hybrid Precision@{k}: {hybrid_p_at_k:.2f}")
    
    print("\n--- Summary ---")
    if hybrid_p_at_k > dense_p_at_k:
        print("Success! Hybrid retrieval improved precision for exact keyword queries.")
    elif hybrid_p_at_k == dense_p_at_k:
        print("Hybrid retrieval maintained baseline performance.")
    else:
        print("Warning: Hybrid retrieval degraded performance on this query.")

if __name__ == "__main__":
    run_evaluation()
