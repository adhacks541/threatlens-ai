from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
import logging

from .config import settings
from .ingestion import ingest_data
from .retrieval import retrieve_documents, measure_precision_at_k
from .rag import generate_rag_response

# Standard structured logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME, 
    description="AI-powered Threat Intelligence Assistant using Endee vector database",
    version="1.0.0"
)

# Advanced Feature: Latency measurement logging via Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"API Request | Path: {request.url.path} | Method: {request.method} | Latency: {process_time:.4f}s | Status: {response.status_code}")
    response.headers["X-Process-Time"] = str(process_time)
    return response

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query for threat intelligence", examples=["What are recent ransomware attack patterns?"])
    top_k: Optional[int] = Field(None, description="Number of documents to retrieve via Endee semantic search. Uses DEFAULT_TOP_K if null.")
    stream: bool = Field(False, description="Whether to stream the generated RAG LLM response")
    metadata_filter: Optional[str] = Field(None, description="Endee filter string (e.g. meta.category == 'ransomware')")

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_documents: List[Dict[str, Any]]
    precision_at_k_example: Optional[float] = None

@app.get("/health")
async def health_check():
    """Returns the operational status of the API and its vector DB connection."""
    return {
        "status": "ok", 
        "service": settings.PROJECT_NAME, 
        "endee_url": settings.ENDEE_URL,
        "llm_model": settings.LLM_MODEL
    }

@app.post("/ingest")
async def ingest_endpoint(background_tasks: BackgroundTasks):
    """Asynchronously triggers the ingestion of threat intelligence data into Endee."""
    background_tasks.add_task(ingest_data, "data/sample_threat_data.json")
    return {"message": "Ingestion task started in the background. Documents are being embedded and loaded into Endee."}

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    """Retrieves relevant threat intel documents and generates a contextual RAG response."""
    logger.info(f"Processing threat query: '{req.query}'")
    
    try:
        # Step 1: Retrieval via Endee
        db_start_time = time.time()
        docs = retrieve_documents(
            query=req.query, 
            top_k=req.top_k, 
            metadata_filter=req.metadata_filter
        )
        db_latency = time.time() - db_start_time
        logger.info(f"DB Retrieval Latency: {db_latency:.4f}s")
        
        # Advanced Feature: Compute Precision@k (Example Implementation)
        k_val = req.top_k if req.top_k else settings.DEFAULT_TOP_K
        
        # In a real environment, ground truth would be dynamically supplied for evaluation.
        # Here we mock it just to demonstrate the logic.
        dummy_ground_truth = ["doc_001", "doc_003"]
        retrieved_ids = [doc.get("id", "") for doc in docs]
        precision = measure_precision_at_k(retrieved_ids, dummy_ground_truth, k=k_val)
        
        # Step 2: Generation via LLM API
        if req.stream:
            # Advanced Feature: Streaming LLM responses via Server Sent Events
            stream_generator = await generate_rag_response(
                query=req.query, 
                retrieved_docs=docs, 
                stream=True
            )
            return StreamingResponse(stream_generator, media_type="text/event-stream")
            
        else:
            # Standard blocking Generation
            llm_start_time = time.time()
            answer = await generate_rag_response(
                query=req.query, 
                retrieved_docs=docs, 
                stream=False
            )
            llm_latency = time.time() - llm_start_time
            logger.info(f"LLM Generation Latency: {llm_latency:.4f}s | Combined Latency: {(db_latency + llm_latency):.4f}s")
            
            return QueryResponse(
                query=req.query,
                answer=answer,
                retrieved_documents=docs,
                precision_at_k_example=precision
            )
            
    except Exception as e:
        logger.error(f"Failed query operation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
