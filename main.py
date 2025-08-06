# main.py (Version 8.4.0 - Auto-Adjusting Cores)
#
# This version dynamically detects available CPU cores for parallel processing,
# making it suitable for cloud deployments like Render.

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

import pymupdf
from multiprocessing import Pool, cpu_count

from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Generator, Optional

from pinecone import Pinecone
from pinecone.exceptions import NotFoundException

# --- Configuration & Initialization ---
load_dotenv()
# --- MODIFIED: Removed static core count for dynamic adjustment ---

app = FastAPI(
    title="LIT RAG with Gemini, PyMuPDF, and Cohere Reranker (Auto-Scaling)",
    description="Processes documents on-demand with a parallelized PyMuPDF parser that auto-adjusts to available CPU cores.",
    version="8.4.0"
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
    print("Using Pinecone's Cohere Reranker API for document reranking.")

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

    print("All components are live. Server is ready. ✅")

# --- Parsing and Chunking Helpers ---

def extract_text_from_pages(vector: tuple) -> str:
    """Extracts text from pages of a document within a specified range.

    Args:
        vector (tuple): A tuple containing process index, total number of processes, and the filename.

    Returns:
        str: Concatenated text snippets extracted from the pages.
    """
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
                print(f"Process {process_idx}: Failed to process page {page_num} - {e}")
        doc.close()
    except Exception as e:
        print(f"Process {process_idx}: Failed to open '{filename}' - {e}")
    return "".join(page_text_snippets)

def run_pymupdf_extraction(filename: str) -> str:
    """
    Synchronous wrapper for multiprocessing text extraction.
    Dynamically uses the available CPU cores for best performance.

    Args:
        filename (str): The path to the document file.

    Returns:
        str: Extracted text content from the document.
    """
    try:
        # --- MODIFIED: Dynamically determine process count ---
        # This allows the app to adapt to the resources of the deployment environment (e.g., Render).
        num_processes = cpu_count() or 2 # Fallback to 2 if detection fails
        print(f"[{os.getpid()}] Starting PyMuPDF extraction with a pool of {num_processes} processes (auto-detected).")
        
        vectors = [(i, num_processes, filename) for i in range(num_processes)]
        with Pool(processes=num_processes) as pool:
            results = pool.map(extract_text_from_pages, vectors)
        return "".join(results)
    except Exception as e:
        print(f"An error occurred during multiprocessing text extraction: {e}")
        return ""

def recursive_character_split(text: str, max_length: int = 4000, overlap: int = 50) -> List[str]:
    """Splits a long text into smaller chunks based on character overlap.

    Args:
        text (str): The input text to split.
        max_length (int): Maximum length of each chunk. Defaults to 4000.
        overlap (int): Number of overlapping characters between chunks. Defaults to 50.

    Returns:
        List[str]: A list of text chunks.
    """
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
    """Generates batches of data from a list.

    Args:
        data (List[Any]): The input list.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[Any], None, None]: A generator yielding batches of data.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def generate_url_hash(url: str) -> str:
    """Generates an MD5 hash from a URL.

    Args:
        url (str): The URL to hash.

    Returns:
        str: The first 16 characters of the MD5 hash.
    """
    return hashlib.md5(url.encode()).hexdigest()[:16]

async def cleanup_namespace(namespace: str) -> bool:
    """Deletes all vectors within a Pinecone namespace.

    Args:
        namespace (str): The namespace to clean up.

    Returns:
        bool: True if successful, False otherwise.
    """
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
    """Waits for the Pinecone index to be ready by checking the vector count.

    Args:
        namespace (str): The Pinecone namespace.
        expected_chunks (int): The expected number of vectors.
        max_wait (int): Maximum wait time in seconds. Defaults to 120.

    Returns:
        bool: True if the index is ready, False otherwise.
    """
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
    """Verifies the authentication token.

    Args:
        credentials (HTTPAuthorizationCredentials, optional): The security credentials. Defaults to Security(security).

    Raises:
        HTTPException: If the authentication token is invalid.
    """
    if not (credentials and credentials.scheme == "Bearer" and credentials.credentials == os.getenv("API_BEARER_TOKEN")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")

class SubmissionRequest(BaseModel):
    """Request model for submitting documents and questions."""
    documents: str
    questions: List[str]

class SubmissionResponse(BaseModel):
    """Response model containing the answers to the submitted questions."""
    answers: List[str]

# --- Core Processing Functions ---
async def process_single_query(query: str, namespace: str, max_retries: int = 3) -> str:
    """Processes a single query against the document index.

    Args:
        query (str): The query string.
        namespace (str): The Pinecone namespace.
        max_retries (int): Maximum number of retries. Defaults to 3.

    Returns:
        str: The answer to the query.
    """
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
            prompt = f"""You are a retrieval-augmented generation (RAG) assistant.

Input:
- context: text from source documents  
- question: a natural-language question  

Your job:
- Use ONLY the context to answer the question.  
- If the answer isn’t in the context, say exactly this:  
  Information not available in the provided context.  
- Return a friendly, helpful answer in plain text.  
- Use warm, human language — like you're explaining it to a curious friend.  
- You may use bullet points or short lists to make things clearer, but keep the tone conversational.  
- Do not add any extra info or rely on outside knowledge — just stick to the given context.  

Example:  
Context:  
> “According to the Remote Work Policy, employees may telecommute up to two days per week, provided they obtain manager approval in advance and maintain core business-hour availability.”

Question:  
> “Under the Remote Work Policy, what conditions must an employee meet to telecommute?”

Answer:  
Hey there! Here’s what the policy says:  
- You can work from home up to two days each week.  
- You’ll need to get your manager’s approval first.  
- Make sure you’re available during core business hours.

Hope that helps!

---

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
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
    """Processes and indexes a document with throttling and readiness verification.

    Args:
        document_url (str): URL of the document to process.
        namespace (str): The Pinecone namespace.

    Returns:
        bool: True if processing was successful, False otherwise.
    """
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

        print(f"[{namespace}] Parsing document with PyMuPDF...")
        full_text_content = await run_in_threadpool(run_pymupdf_extraction, temp_file_path)
        if not full_text_content:
            print(f"[{namespace}] Failed to extract any text.")
            return False
            
        document_chunks = recursive_character_split(full_text_content)
        if not document_chunks:
            print(f"[{namespace}] Failed to chunk the document text.")
            return False
        
        print(f"[{namespace}] Document processed into {len(document_chunks)} chunks.")

        async def embed_and_upsert_batch(chunk_batch: List[str], batch_start_index: int) -> bool:
            """Embeds and upserts a batch of chunks to Pinecone.

            Args:
                chunk_batch (List[str]): A list of text chunks.
                batch_start_index (int): The starting index of the batch.

            Returns:
                bool: True if successful, False otherwise.
            """
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
    """Implements the stateless hybrid RAG pipeline.

    Args:
        request (SubmissionRequest): The submission request containing the document URL and questions.

    Returns:
        SubmissionResponse: The answers to the questions.

    Raises:
        HTTPException: If there's an error processing the document or answering the questions.
    """
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
