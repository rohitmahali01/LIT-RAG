# main.py (Version 8.5.0 - LlamaParse & Cohere Reranker)
#
# This version replaces the PyMuPDF parser with the more advanced LlamaParse
# while retaining the high-performance Cohere Reranker API. The architecture
# remains stateless and dynamically adjusts to available resources.

import os
import re
import uuid
import tempfile
import asyncio
import httpx
import aiofiles
import hashlib
import time
from collections import defaultdict
import google.generativeai as genai
from dotenv import load_dotenv
from urllib.parse import urlparse
import mimetypes

from llama_parse import LlamaParse

from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Generator, Optional

from pinecone import Pinecone
from pinecone.exceptions import NotFoundException

# --- Configuration & Initialization ---
load_dotenv()

app = FastAPI(
    title="LIT RAG with Gemini, LlamaParse, and Cohere Reranker",
    description="Processes documents on-demand with LlamaParse and reranks results using Cohere's API in a stateless, auto-scaling architecture.",
    version="8.5.0"
)

# Global objects
models: Dict[str, Any] = {}
pc: Pinecone = None
pinecone_index = None

# Model and Dimension constants
DENSE_MODEL = "llama-text-embed-v2"
SPARSE_MODEL = "pinecone-sparse-english-v0"
DENSE_DIMENSION = 1024
RERANK_MODEL = "cohere-rerank-3.5"


# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initialize service connections and models."""
    print("Server starting up...")
    print("Using LlamaParse for document parsing.")
    print("Using Pinecone's Cohere Reranker API for document reranking.")

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not found.")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    models["generation_model"] = genai.GenerativeModel('gemini-2.5-flash-lite')

    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        raise ValueError("LLAMA_CLOUD_API_KEY environment variable not found.")
    models["parser"] = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY"), result_type='markdown', output_tables_as_HTML=True)

    global pc, pinecone_index
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable not found.")
    pc = Pinecone(api_key=pinecone_api_key)
    
    index_name = "hybrid-challenge-index-llamaparse"
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=DENSE_DIMENSION, metric="dotproduct", spec={"serverless": {"cloud": "aws", "region": "us-east-1"}})
    pinecone_index = pc.Index(index_name)

    print("All components are live. Server is ready. ✅")

# --- Parsing and Chunking Helpers ---

def chunk_markdown_file(file_content: str) -> List[str]:
    """Splits markdown content into semantic chunks based on headers."""
    if not file_content: return []
    # A regex to split the text by markdown headers (e.g., #, ##, ###)
    raw_chunks = re.split(r'(?=^#)', file_content, flags=re.MULTILINE)
    # Filter out any empty strings that may result from the split
    return [chunk.strip() for chunk in raw_chunks if chunk.strip()]


# --- Helper Functions ---
def batch_generator(data: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def generate_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]

async def cleanup_namespace(namespace: str) -> bool:
    try:
        await run_in_threadpool(pinecone_index.delete, delete_all=True, namespace=namespace)
        print(f"[CLEANUP] Successfully deleted existing namespace: {namespace}")
        return True
    except NotFoundException:
        print(f"[CLEANUP] Namespace {namespace} did not exist. No action needed.")
        return True
    except Exception as e:
        print(f"[CLEANUP] An unexpected error occurred while deleting namespace {namespace}: {e}")
        return False

async def wait_for_index_readiness(namespace: str, expected_chunks: int, max_wait: int = 120) -> bool:
    """Waits for the Pinecone index to be ready by checking the vector count."""
    print(f"[{namespace}] Waiting for index to be ready with at least {expected_chunks} vectors...")
    for attempt in range(max_wait):
        try:
            index_stats = await run_in_threadpool(pinecone_index.describe_index_stats)
            namespaces = index_stats.get('namespaces', {})
            vector_count = namespaces.get(namespace, {}).get('vector_count', 0)
            
            print(f"[{namespace}] Readiness Check (Attempt {attempt + 1}/{max_wait}): Found {vector_count}/{expected_chunks} vectors.")
            
            if vector_count >= expected_chunks:
                print(f"[{namespace}] Index has reached the expected vector count.")
                test_query_response = await run_in_threadpool(
                    pinecone_index.query, namespace=namespace, top_k=1, vector=[0.0] * DENSE_DIMENSION
                )
                print(f"[{namespace}] Test query successful. Index is ready!")
                return True
            
            await asyncio.sleep(1)
        except Exception as e:
            print(f"[{namespace}] Error during readiness check: {e}. Retrying...")
            await asyncio.sleep(1)
    
    print(f"[{namespace}] WARNING: Index readiness timeout after {max_wait} seconds. The index may not be fully populated.")
    return False

# --- Security & API Models ---
security = HTTPBearer()
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not (credentials and credentials.scheme == "Bearer" and credentials.credentials == os.getenv("API_BEARER_TOKEN")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")

class SubmissionRequest(BaseModel):
    documents: str
    questions: List[str]

class SubmissionResponse(BaseModel):
    answers: List[str]

# --- Core Processing Functions ---
async def process_single_query(query: str, namespace: str, max_retries: int = 3) -> str:
    """Processes a query with retry logic for enhanced robustness."""
    for attempt in range(max_retries):
        try:
            dense_response, sparse_response = await asyncio.gather(
                run_in_threadpool(pc.inference.embed, model=DENSE_MODEL, inputs=[query], parameters={"input_type": "query"}),
                run_in_threadpool(pc.inference.embed, model=SPARSE_MODEL, inputs=[query], parameters={"input_type": "query"})
            )
            dense_embedding = dense_response[0]['values']
            sparse_vector_payload = {'indices': sparse_response[0]['sparse_indices'], 'values': sparse_response[0]['sparse_values']}
           
            query_response = await run_in_threadpool(
                pinecone_index.query, namespace=namespace, top_k=100, vector=dense_embedding,
                sparse_vector=sparse_vector_payload, include_metadata=True
            )
           
            retrieved_docs = [match['metadata']['text'] for match in query_response['matches']]
            if not retrieved_docs:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[{namespace}] No documents found for query. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return "Could not find relevant information in the document after multiple retries."

            rerank_response = await run_in_threadpool(
                pc.inference.rerank, model=RERANK_MODEL, query=query,
                documents=retrieved_docs[:30], top_n=10, return_documents=True
            )
            reranked_docs_text = [result.document.text for result in rerank_response.data]

            context = "\n\n---\n\n".join(reranked_docs_text)
            prompt = f"You are an AI assistant tasked with explaining policy documents. Your response must be factual, clear, and easy to understand, with a supportive tone.Analyze the user's 'QUESTION' and formulate an answer using *exclusively* the provided 'CONTEXT'. If the topic is complex, feel free to use bullet points to structure the information for clarity. Under no circumstances should you use outside knowledge. If the context is insufficient, please state that the document does not provide the necessary details.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:"
            generation_response = await models["generation_model"].generate_content_async(prompt)
            
            return generation_response.text.strip()

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"[{namespace}] Error processing query (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"[{namespace}] Error processing query '{query}' after all retries: {e}")
                return "An internal error occurred while answering this question."
    return "Failed to process query after multiple retries."

# --- Optimized Ingestion Function ---
async def process_and_index_document(document_url: str, namespace: str) -> bool:
    """Processes and indexes a document with throttling and readiness verification."""
    temp_file_path = None
    try:
        print(f"[{namespace}] Downloading document...")
        async with httpx.AsyncClient() as client:
            response = await client.get(document_url, follow_redirects=True, timeout=120.0)
            response.raise_for_status()

        parsed_url = urlparse(document_url)
        _, file_ext = os.path.splitext(parsed_url.path)
        if not file_ext:
            content_type = response.headers.get('content-type')
            if content_type: file_ext = mimetypes.guess_extension(content_type) or '.pdf'
        
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"{namespace}{file_ext}")
        async with aiofiles.open(temp_file_path, "wb") as f:
            await f.write(response.content)

        print(f"[{namespace}] Parsing document with LlamaParse... (This may take time)")
        parsed_docs = await models["parser"].aload_data(temp_file_path)
        full_text_content = "\n\n".join(doc.text for doc in parsed_docs)
        if not full_text_content:
            print(f"[{namespace}] Failed to extract any text.")
            return False
            
        document_chunks = chunk_markdown_file(full_text_content)
        if not document_chunks:
            print(f"[{namespace}] Failed to chunk the document text.")
            return False
        
        print(f"[{namespace}] Document processed into {len(document_chunks)} chunks.")

        async def embed_and_upsert_batch(chunk_batch: List[str], batch_start_index: int) -> bool:
            try:
                dense_response, sparse_response = await asyncio.gather(
                    run_in_threadpool(pc.inference.embed, model=DENSE_MODEL, inputs=chunk_batch, parameters={"input_type": "passage"}),
                    run_in_threadpool(pc.inference.embed, model=SPARSE_MODEL, inputs=chunk_batch, parameters={"input_type": "passage"})
                )
                vectors_to_upsert = [{
                    "id": f"chunk-{batch_start_index + j}",
                    "values": dense_response[j]['values'],
                    "sparse_values": {'indices': sparse_response[j]['sparse_indices'], 'values': sparse_response[j]['sparse_values']},
                    "metadata": {'text': chunk}
                } for j, chunk in enumerate(chunk_batch)]
                if vectors_to_upsert:
                    await run_in_threadpool(pinecone_index.upsert, vectors=vectors_to_upsert, namespace=namespace)
                    await asyncio.sleep(0.1)
                return True
            except Exception as e:
                print(f"[{namespace}] FAILED to process batch starting at index {batch_start_index}: {e}")
                return False

        batch_size = 95
        pipeline_tasks = [embed_and_upsert_batch(batch, i * batch_size) for i, batch in enumerate(batch_generator(document_chunks, batch_size))]
        task_results = await asyncio.gather(*pipeline_tasks)
       
        if not all(task_results):
            print(f"[{namespace}] One or more ingestion pipelines failed. Cleaning up partial data.")
            await cleanup_namespace(namespace)
            return False
        
        print(f"[{namespace}] All ingestion pipelines completed. Verifying index readiness...")
        
        is_ready = await wait_for_index_readiness(namespace, len(document_chunks))
        if not is_ready:
            print(f"[{namespace}] Warning: Proceeding without full index readiness confirmation.")
        
        return True
       
    except Exception as e:
        print(f"[{namespace}] A critical error occurred during document processing: {e}")
        await cleanup_namespace(namespace)
        return False
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# --- Main API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=SubmissionResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: SubmissionRequest):
    """Implements the stateless hybrid RAG pipeline."""
    url_hash = generate_url_hash(request.documents)
    namespace = f"doc-{url_hash}"
   
    try:
        print(f"[{namespace}] Ensuring a clean slate by deleting existing namespace data...")
        await cleanup_namespace(namespace)

        print(f"[{namespace}] Starting document processing and indexing...")
        processing_successful = await process_and_index_document(request.documents, namespace)
        if not processing_successful:
            raise HTTPException(status_code=500, detail="Failed to process and index the document.")
        
        print(f"[{namespace}] Processing {len(request.questions)} questions concurrently...")
        query_tasks = [process_single_query(query, namespace) for query in request.questions]
        all_answers = await asyncio.gather(*query_tasks)
       
        print(f"[{namespace}] All questions processed successfully!")
        return SubmissionResponse(answers=all_answers)

    except HTTPException:
        raise
    except Exception as e:
        print(f"An unexpected error occurred in run_submission: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
