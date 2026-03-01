from openai import AsyncOpenAI
from .config import settings
import logging
from typing import List, Dict, Any, AsyncGenerator, Union
import json
import tiktoken

logger = logging.getLogger(__name__)

async def generate_rag_response(
    query: str, 
    retrieved_docs: List[Dict[str, Any]], 
    stream: bool = False
) -> Union[str, AsyncGenerator[str, None]]:
    """
    Generates a contextual response using an LLM combined with the retrieved documents.
    Supports streaming answers as defined in Advanced Features.
    """
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    # Cleanly construct the context prompt from the retrieved JSON documents
    contexts = []
    
    # Initialize tokenizer for the specific LLM model (or fallback to default cl100k_base)
    try:
        encoding = tiktoken.encoding_for_model(settings.LLM_MODEL)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
        
    # Token control: token budgeting to prevent exceeding the model's context window limits
    # We allocate a fixed budget (e.g., 4000 tokens) for retrieved context to ensure enough
    # room remains for the system prompt, the user query, and the LLM's generated response.
    MAX_TOKEN_BUDGET = 4000 
    current_tokens = 0
    
    for idx, doc in enumerate(retrieved_docs, start=1):
        # The content field is a JSON string of metadata from Endee insertion
        raw_meta = doc.get("content", "{}")
        try:
            meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
        except json.JSONDecodeError:
            meta = {}
            
        category = meta.get('category', 'unknown').upper()
        severity = meta.get('severity', 'unknown').upper()
        content = meta.get('content', '')
        source = meta.get('source', 'Unknown Source')
        date = meta.get('date', 'Unknown Date')
        
        doc_text = f"DOCUMENT {idx}:\nSource: {source} ({date})\nSeverity: {severity} | Category: {category}\nContent: {content}\n"
        
        doc_tokens = len(encoding.encode(doc_text))
        
        if current_tokens + doc_tokens > MAX_TOKEN_BUDGET:
            logger.warning(f"Truncating contexts to avoid exceeding maximum token budget ({MAX_TOKEN_BUDGET} tokens).")
            # Explanatory comment for the trimming strategy:
            # We break early once our current accumulated tokens plus the next document's tokens 
            # exceed our budget, effectively prioritizing higher-ranked retrieval results.
            break
            
        contexts.append(doc_text)
        current_tokens += doc_tokens
        
    context_text = "\n".join(contexts) if contexts else "No relevant threat intelligence found in the database."
    
    system_prompt = (
        "You are 'ThreatLens', an expert AI Threat Intelligence Assistant responding to cybersecurity analysts.\n"
        "Your primary task is to answer the user's question strictly using the provided contextual documents.\n"
        "- If the answer cannot be determined from the context, explicitly state: 'Based on the available intelligence, I cannot provide an answer.'\n"
        "- Never hallucinate, guess, or bring in outside knowledge.\n"
        "- Cite your sources clearly by referencing the Document Number or Source name.\n"
        "- Format your response professionally using Markdown (e.g., bullet points, bold text for critical IOCs or threat actors)."
    )
    
    user_prompt = f"### CONTEXTUAL THREAT DATA:\n{context_text}\n\n### USER QUESTION:\n{query}"
    
    logger.info(f"Sending prompt to LLM '{settings.LLM_MODEL}' (stream={stream}, context_tokens={current_tokens})")
    
    if stream:
        async def stream_generator() -> AsyncGenerator[str, None]:
            try:
                response = await client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    stream=True,
                    temperature=0.1
                )
                async for chunk in response:
                    content_chunk = chunk.choices[0].delta.content
                    if content_chunk is not None:
                        yield content_chunk
            except Exception as e:
                logger.error(f"Error during OpenAI streaming: {e}")
                yield f"\n\n[Generation Error: {str(e)}]"
        
        return stream_generator()
    else:
        try:
            response = await client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise RuntimeError(f"Error communicating with LLM: {str(e)}")
