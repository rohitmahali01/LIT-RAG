# main.py (Version 8.0.0 - Caching Removed)
#
# This version removes all caching layers for a stateless, on-demand processing
# pipeline suitable for hackathon environments. It uses a high-performance,
# parallelized PyMuPDF strategy for document parsing and a recursive chunking method.

import os
import re
import uuid
import tempfile
import asyncio
import httpx
import aiofiles
import torch
import hashlib
import time
from collections import defaultdict
import openai
from dotenv import load_dotenv
from urllib.parse import urlparse
import mimetypes

# --- NEW IMPORTS for PyMuPDF and Multiprocessing ---
import pymupdf
from multiprocessing import Pool, cpu_count
# ---

from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Generator, Optional

from sentence_transformers import CrossEncoder
from pinecone import Pinecone
from pinecone.exceptions import NotFoundException

# --- Configuration & Initialization ---
load_dotenv()
OPTIMAL_CORE_COUNT = 4


app = FastAPI(
    title="LIT RAG with ChatGPT and PyMuPDF (No Cache)",
    description="Processes documents on-demand using a parallelized PyMuPDF parser and concurrent mini-pipelines for embedding and upserting.",
    version="8.0.0"
)

# Global objects
models: Dict[str, Any] = {}
pc: Pinecone = None
pinecone_index = None

# Model and Dimension constants
DENSE_MODEL = "llama-text-embed-v2"
SPARSE_MODEL = "pinecone-sparse-english-v0"
DENSE_DIMENSION = 1024


# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initialize service connections and models."""
    print("Server starting up...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} for CrossEncoder")

    models["cross_encoder"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L2-v2", device='cuda')

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not found.")
    models["generation_model"] = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    global pc, pinecone_index
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "hybrid-challenge-index"
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=DENSE_DIMENSION, metric="dotproduct", spec={"serverless": {"cloud": "aws", "region": "us-east-1"}})
    pinecone_index = pc.Index(index_name)

    print("All components are live. Server is ready. ✅")

# --- NEW PARSING AND CHUNKING HELPERS ---

def extract_text_from_pages(vector: tuple) -> str:
    """
    Extracts plain text from a range of pages. Must be a top-level function.
    """
    process_idx, total_cpus, filename = vector
    page_text_snippets = []
    try:
        doc = pymupdf.open(filename)
    except Exception as e:
        print(f"Process {process_idx}: Failed to open '{filename}' - {e}")
        return ""

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
            print(f"Process {process_idx}: Failed to process page {page_num} - {e}")

    doc.close()
    return "".join(page_text_snippets)

def run_pymupdf_extraction(filename: str) -> str:
    """
    Synchronous wrapper for the multiprocessing text extraction.
    Uses a predefined optimal number of processes for best performance.
    """
    try:
        # Use the optimized constant instead of cpu_count()
        num_processes = OPTIMAL_CORE_COUNT
        
        # Add a log to confirm which value is being used
        print(f"[{os.getpid()}] Starting PyMuPDF extraction with a pool of {num_processes} processes.")
        
        vectors = [(i, num_processes, filename) for i in range(num_processes)]
        with Pool(processes=num_processes) as pool:
            results = pool.map(extract_text_from_pages, vectors)
        return "".join(results)
    except Exception as e:
        print(f"An error occurred during multiprocessing text extraction: {e}")
        return ""

def recursive_character_split(text: str, max_length: int = 5000, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of a maximum length with recursive fallback and overlap.
    """
    if not text:
        return []
    chunks = []
    current_chunk_start = 0
    while current_chunk_start < len(text):
        end_pos = current_chunk_start + max_length
        if end_pos >= len(text):
            chunks.append(text[current_chunk_start:].strip())
            break
        
        split_pos = -1
        for sep in ["\n\n", "\n", ". ", "? ", "! ", "; ", ", "]:
            found_pos = text.rfind(sep, current_chunk_start, end_pos)
            if found_pos != -1:
                split_pos = found_pos + len(sep)
                break
        if split_pos == -1:
            split_pos = end_pos
            
        chunk = text[current_chunk_start:split_pos].strip()
        if chunk:
            chunks.append(chunk)
            
        current_chunk_start = max(current_chunk_start + 1, split_pos - overlap)
    return [c for c in chunks if c]


# --- Helper Functions ---
def batch_generator(data: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def generate_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]

async def cleanup_namespace(namespace: str) -> bool:
    """
    Deletes all vectors within a given namespace.
    Handles the case where the namespace doesn't exist as a success condition.
    """
    try:
        await run_in_threadpool(pinecone_index.delete, delete_all=True, namespace=namespace)
        print(f"[CLEANUP] Successfully deleted existing namespace: {namespace}")
        return True
    except NotFoundException:
        # This is expected and means the namespace is already clean.
        print(f"[CLEANUP] Namespace {namespace} did not exist. No action needed.")
        return True # Return True because the goal (a clean slate) is met.
    except Exception as e:
        # Catch any other unexpected errors (e.g., auth, network)
        print(f"[CLEANUP] An unexpected error occurred while deleting namespace {namespace}: {e}")
        return False

# --- Security & API Models (No changes) ---
security = HTTPBearer()
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not (credentials and credentials.scheme == "Bearer" and credentials.credentials == os.getenv("API_BEARER_TOKEN")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")

class SubmissionRequest(BaseModel):
    documents: str
    questions: List[str]

class SubmissionResponse(BaseModel):
    answers: List[str]

# --- Core Processing Functions (No changes) ---
async def process_single_query(query: str, namespace: str) -> str:
    try:
        dense_response, sparse_response = await asyncio.gather(
            run_in_threadpool(pc.inference.embed, model=DENSE_MODEL, inputs=[query], parameters={"input_type": "query"}),
            run_in_threadpool(pc.inference.embed, model=SPARSE_MODEL, inputs=[query], parameters={"input_type": "query"})
        )
        dense_embedding = dense_response[0]['values']
        sparse_vector_payload = {'indices': sparse_response[0]['sparse_indices'], 'values': sparse_response[0]['sparse_values']}
       
        query_response = await run_in_threadpool(
            pinecone_index.query, namespace=namespace, top_k=50, vector=dense_embedding,
            sparse_vector=sparse_vector_payload, include_metadata=True
        )
       
        retrieved_docs = [match['metadata']['text'] for match in query_response['matches']]
        if not retrieved_docs: return "Could not find relevant information in the document."

        cross_pairs = [[query, doc] for doc in retrieved_docs[:20]]
        cross_scores = await run_in_threadpool(models["cross_encoder"].predict, cross_pairs, show_progress_bar=False)
        reranked_final = sorted(zip(cross_scores, retrieved_docs), key=lambda x: x[0], reverse=True)

        context = "\n\n---\n\n".join([doc for _, doc in reranked_final[:10]])
        prompt = f"You are an expert legal and banking policy advisor. Rely *only* on the provided CONTEXT (bank policies, legal texts) to craft concise, precise answers. Cite relevant sections or paragraph numbers where applicable. Adopt a professional and warm tone.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:"
       
        chat_completion = await models["generation_model"].chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model="gpt-4.1-nano",
        )
        return chat_completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error processing query '{query}': {e}")
        return "An internal error occurred while answering this question."

# ===================================================================================
# === OPTIMIZED INGESTION FUNCTION (MODIFIED with PyMuPDF and new chunking) =======
# ===================================================================================
async def process_and_index_document(document_url: str, namespace: str) -> bool:
    """
    Processes a document using PyMuPDF for parsing and concurrent mini-pipelines for embedding and upserting.
    """
    temp_file_path = None
    try:
        # Step 1: Download document
        print(f"[{namespace}] Downloading document...")
        async with httpx.AsyncClient() as client:
            response = await client.get(document_url, follow_redirects=True, timeout=120.0)
            response.raise_for_status()

        # Dynamic file extension logic
        parsed_url = urlparse(document_url)
        _, file_ext = os.path.splitext(parsed_url.path)
        if not file_ext:
            content_type = response.headers.get('content-type')
            if content_type:
                guessed_ext = mimetypes.guess_extension(content_type)
                if guessed_ext: file_ext = guessed_ext
        if not file_ext: file_ext = '.pdf'

        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"{namespace}{file_ext}")
        print(f"[{namespace}] Saving temporary file as: {temp_file_path}")

        async with aiofiles.open(temp_file_path, "wb") as f:
            await f.write(response.content)

        # Step 2: Parse with PyMuPDF and chunk with new strategy
        print(f"[{namespace}] Parsing document with PyMuPDF...")
        
        # Get and display the number of CPU cores that will be used for extraction.
        cores_to_use = cpu_count() or 2
        print(f"[{namespace}] Utilizing {cores_to_use} CPU cores for parallel text extraction.")
        
        full_text_content = await run_in_threadpool(run_pymupdf_extraction, temp_file_path)
        
        if not full_text_content:
            print(f"[{namespace}] Failed to extract any text from the document using PyMuPDF.")
            return False
            
        print(f"[{namespace}] Text extracted. Now chunking content...")
        document_chunks = recursive_character_split(full_text_content)
        
        if not document_chunks:
            print(f"[{namespace}] Failed to chunk the document text.")
            return False
        
        print(f"[{namespace}] Document processed into {len(document_chunks)} chunks.")

        # Step 3: Embed and Upsert
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
                return True
            except Exception as e:
                print(f"[{namespace}] FAILED to process batch starting at index {batch_start_index}: {e}")
                return False

        batch_size = 95
        pipeline_tasks = [embed_and_upsert_batch(batch, i * batch_size) for i, batch in enumerate(batch_generator(document_chunks, batch_size))]
        print(f"[{namespace}] Launching {len(pipeline_tasks)} concurrent ingestion pipelines...")
        task_results = await asyncio.gather(*pipeline_tasks)
       
        if not all(task_results):
            print(f"[{namespace}] One or more ingestion pipelines failed. Cleaning up partial data.")
            await cleanup_namespace(namespace)
            return False

        print(f"[{namespace}] All ingestion pipelines completed successfully.")
        return True
       
    except Exception as e:
        print(f"[{namespace}] A critical error occurred during document processing: {e}")
        await cleanup_namespace(namespace) # Clean up on failure
        return False
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# --- Main API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=SubmissionResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: SubmissionRequest):
    """
    Implements the hybrid RAG pipeline with on-demand document ingestion. Caching is disabled.
    Each request processes the document from scratch.
    """
    url_hash = generate_url_hash(request.documents)
    namespace = f"doc-{url_hash}"
   
    try:
        # Step 1: Always clean up the namespace to ensure a fresh start.
        # This is critical in a no-cache setup to prevent data from previous runs.
        print(f"[{namespace}] Ensuring a clean slate by deleting existing namespace data...")
        await cleanup_namespace(namespace)

        # Step 2: Process and index the document from scratch for every request.
        print(f"[{namespace}] Starting document processing and indexing...")
        try:
            processing_successful = await process_and_index_document(request.documents, namespace)
            if not processing_successful:
                # The processing function already logs detailed errors.
                raise Exception("Document processing and indexing failed.")
        except Exception as e:
            # Wrap processing errors in an HTTPException for a clear client response.
            raise HTTPException(status_code=400, detail=f"Failed to process document: {str(e)}")

        # A short delay can help ensure the Pinecone index is fully queryable after upserting.
        print(f"[{namespace}] Ingestion finished. Waiting 2 seconds for index to update...")
        await asyncio.sleep(2)

        # Step 3: Process all questions against the newly created index.
        print(f"[{namespace}] Processing {len(request.questions)} questions concurrently...")
        query_tasks = [process_single_query(query, namespace) for query in request.questions]
        all_answers = await asyncio.gather(*query_tasks)
       
        print(f"[{namespace}] All questions processed successfully!")
        return SubmissionResponse(answers=all_answers)

    except HTTPException:
        # Re-raise HTTPExceptions directly to let FastAPI handle them.
        raise
    except Exception as e:
        # Catch any other unexpected errors and return a generic 500 error.
        print(f"An unexpected error occurred in run_submission: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")