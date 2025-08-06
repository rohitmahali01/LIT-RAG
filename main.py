# main.py (Version 9.0.0 - Production Architecture)
#
# This version implements the final and most robust architecture by delegating
# the entire multiprocessing PDF parsing task to an isolated, sandboxed
# process. This completely resolves the underlying conflict between gRPC
# threads in the main application and the use of multiprocessing, ensuring
# stability and correctness even under heavy load in a multi-worker environment.

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

import multiprocessing as mp
import pymupdf

from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Generator, Optional

from pinecone import Pinecone
from pinecone.exceptions import NotFoundException
from filelock import FileLock, Timeout

# --- Fork-Safety Configuration ---
# Set the start method to 'spawn' to ensure clean child processes.
# This is critical for preventing conflicts with libraries like gRPC.
try:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method("spawn", force=True)
        print("Set multiprocessing start method to 'spawn' for gRPC and fork safety.")
except RuntimeError:
    pass

# --- Configuration & Initialization ---
load_dotenv()

app = FastAPI(
    title="LIT RAG with Gemini, PyMuPDF, and Cohere Reranker (Auto-Scaling)",
    description="Processes documents on-demand with a parallelized PyMuPDF parser that auto-adjusts to available CPU cores.",
    version="9.0.0"
)

lock_path = os.path.join(tempfile.gettempdir(), "rag_pipeline.lock")
global_lock = FileLock(lock_path)

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
    print(f"[{os.getpid()}] Server worker starting up...")
    print(f"[{os.getpid()}] Using multiprocessing start method: {mp.get_start_method()}")
    print(f"[{os.getpid()}] Using Pinecone's Cohere Reranker API for document reranking.")

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not found.")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    models["generation_model"] = genai.GenerativeModel('gemini-2.5-flash-lite')

    global pc, pinecone_index
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable not found.")
    pc = Pinecone(api_key=pinecone_api_key)
    
    index_name = "hybrid-challenge-index"
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=DENSE_DIMENSION, metric="dotproduct", spec={"serverless": {"cloud": "aws", "region": "us-east-1"}})
    pinecone_index = pc.Index(index_name)

    print(f"[{os.getpid()}] All components are live. Server is ready. ✅")

# --- MODIFIED: Isolated PDF Parsing Logic ---

def extract_text_from_pages(vector: tuple) -> str:
    """This is the lowest-level worker function, it processes a subset of pages."""
    process_idx, total_cpus, filename = vector
    page_text_snippets = []
    try:
        doc = pymupdf.open(filename)
        num_pages = doc.page_count
        pages_per_process = (num_pages + total_cpus - 1) // total_cpus
        start_page = process_idx * pages_per_process
        end_page = min(start_page + pages_per_process, num_pages)
        for page_num in range(start_page, end_page):
            try:
                page = doc[page_num]
                page_text_snippets.append(f"### Page {page_num + 1}\n\n")
                page_text_snippets.append(page.get_text("text"))
                page_text_snippets.append("\n\n---\n\n")
            except Exception as e:
                print(f"Sub-process {os.getpid()}: Failed to process page {page_num} - {e}")
        doc.close()
    except Exception as e:
        print(f"Sub-process {os.getpid()}: Failed to open '{filename}' - {e}")
    return "".join(page_text_snippets)

def pymupdf_worker(filename: str, queue: mp.Queue):
    """
    This function runs in a completely separate process. It is designed to be a
    target for mp.Process. It creates its OWN Pool, extracts text, and puts
    the result in a queue. It has no knowledge of gRPC or the main web server.
    """
    try:
        print(f"[Isolated Worker {os.getpid()}] Starting PDF processing.")
        num_processes = mp.cpu_count() or 2
        vectors = [(i, num_processes, filename) for i in range(num_processes)]
        
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(extract_text_from_pages, vectors)
        
        full_text = "".join(results)
        if not full_text:
            raise ValueError("Extraction resulted in empty text.")
        queue.put(full_text)
        print(f"[Isolated Worker {os.getpid()}] PDF processing successful.")
    except Exception as e:
        print(f"[Isolated Worker {os.getpid()}] CRITICAL FAILURE during PDF processing: {e}")
        queue.put(e)

async def run_pymupdf_extraction(filename: str) -> str:
    """
    Launches the PyMuPDF extraction in a dedicated, isolated process to avoid
    conflicts with the main application's gRPC threads.
    """
    ctx = mp.get_context()
    q = ctx.Queue()
    
    process = ctx.Process(target=pymupdf_worker, args=(filename, q))
    
    def run_and_wait():
        process.start()
        process.join(timeout=180) # 3-minute timeout for the entire parsing job
        if process.is_alive():
            print(f"[Parent {os.getpid()}] PyMuPDF process timed out, terminating.")
            process.terminate()
            process.join()
            return "TIMEOUT"
        if process.exitcode != 0:
            print(f"[Parent {os.getpid()}] PyMuPDF process exited with non-zero code: {process.exitcode}")
            return "PROCESS_ERROR"

    # Run the blocking process management in a thread pool to not block the event loop.
    status = await run_in_threadpool(run_and_wait)
    if status in ["TIMEOUT", "PROCESS_ERROR"]:
        return ""

    if q.empty():
        print(f"[Parent {os.getpid()}] PyMuPDF worker queue is empty after process finished. This indicates an early failure.")
        return ""
        
    result = q.get()
    if isinstance(result, Exception):
        print(f"[Parent {os.getpid()}] Received an exception from the PyMuPDF worker: {result}")
        return ""
        
    return result


def recursive_character_split(text: str, max_length: int = 4000, overlap: int = 50) -> List[str]:
    if not text: return []
    chunks = []
    current_chunk_start = 0
    while current_chunk_start < len(text):
        end_pos = current_chunk_start + max_length
        if end_pos >= len(text):
            chunks.append(text[current_chunk_start:].strip())
            break
        split_pos = text.rfind("\n\n", current_chunk_start, end_pos)
        if split_pos == -1: split_pos = text.rfind("\n", current_chunk_start, end_pos)
        if split_pos == -1: split_pos = text.rfind(". ", current_chunk_start, end_pos)
        if split_pos == -1: split_pos = end_pos
        chunk = text[current_chunk_start:split_pos].strip()
        if chunk: chunks.append(chunk)
        current_chunk_start = max(current_chunk_start + 1, split_pos - overlap)
    return [c for c in chunks if c]

# --- Helper Functions ---
def batch_generator(data: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def generate_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]

async def cleanup_namespace(namespace: str) -> bool:
    try:
        await run_in_threadpool(pinecone_index.delete, delete_all=True, namespace=namespace)
        print(f"[{namespace}] Successfully deleted existing namespace.")
        return True
    except NotFoundException:
        print(f"[{namespace}] Namespace did not exist. No action needed.")
        return True
    except Exception as e:
        print(f"[{namespace}] An unexpected error occurred while deleting namespace {namespace}: {e}")
        return False

async def wait_for_index_readiness(namespace: str, expected_chunks: int, max_wait: int = 120) -> bool:
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
    
    print(f"[{namespace}] CRITICAL: Index readiness timeout after {max_wait} seconds.")
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

async def process_and_index_document(document_url: str, namespace: str) -> bool:
    """Processes and indexes a document with throttling and readiness verification."""
    temp_file_path = None
    try:
        await cleanup_namespace(namespace)

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

        print(f"[{namespace}] Delegating parsing to isolated process...")
        full_text_content = await run_pymupdf_extraction(temp_file_path)
        if not full_text_content:
            print(f"[{namespace}] Failed to extract any text from the document.")
            return False
            
        document_chunks = recursive_character_split(full_text_content)
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
                    "id": f"chunk-{batch_start_index + j}", "values": dense_response[j]['values'],
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
            print(f"[{namespace}] One or more ingestion pipelines failed. Cleaning up.")
            await cleanup_namespace(namespace)
            return False
        
        print(f"[{namespace}] All ingestion pipelines completed. Verifying index readiness...")
        
        is_ready = await wait_for_index_readiness(namespace, len(document_chunks))
        if not is_ready:
            print(f"[{namespace}] Index readiness check failed. Aborting and cleaning up.")
            await cleanup_namespace(namespace)
            return False
        
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
    url_hash = generate_url_hash(request.documents)
    namespace = f"doc-{url_hash}"
    
    try:
        await run_in_threadpool(global_lock.acquire, timeout=300)
        print(f"[{namespace}] Global lock acquired by process {os.getpid()}.")

        try:
            print(f"[{namespace}] Starting document processing and indexing...")
            processing_successful = await process_and_index_document(request.documents, namespace)
            if not processing_successful:
                raise HTTPException(status_code=500, detail="Failed to process and index the document. Check logs for details.")
            
            print(f"[{namespace}] Processing {len(request.questions)} questions concurrently...")
            query_tasks = [process_single_query(query, namespace) for query in request.questions]
            all_answers = await asyncio.gather(*query_tasks)
           
            print(f"[{namespace}] All questions processed successfully!")
            return SubmissionResponse(answers=all_answers)

        except HTTPException:
            raise
        except Exception as e:
            print(f"An unexpected error occurred in the critical section of run_submission: {e}")
            raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

    except Timeout:
        print(f"[{namespace}] Could not acquire global lock, another process is holding it.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The service is currently processing another document. Please try again."
        )
    finally:
        if global_lock.is_locked:
            global_lock.release()
            print(f"[{namespace}] Global lock released by process {os.getpid()}.")
