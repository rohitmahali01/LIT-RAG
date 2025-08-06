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
import multiprocessing
# Use 'spawn' to avoid forking gRPC threads into child processes
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # start method already set or on Windows (spawn is default)
    pass
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

app = FastAPI(
    title="LIT RAG with Gemini, PyMuPDF, and Cohere Reranker (Auto-Scaling)",
    description="Processes documents on-demand with a parallelized PyMuPDF parser that auto-adjusts to available CPU cores.",
    version="8.4.0"
)

# Global objects
models: Dict[str, Any] = {}
pc: Pinecone = None
pinecone_index = None

# Embedding & rerank model constants
dense_model = "llama-text-embed-v2"
sparse_model = "pinecone-sparse-english-v0"
dense_dimension = 1024
rerank_model = "cohere-rerank-3.5"

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    print("Server starting up...")
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
        pc.create_index(
            name=index_name,
            dimension=dense_dimension,
            metric="dotproduct",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
        )
    pinecone_index = pc.Index(index_name)
    print("All components are live. Server is ready. ✅")

# --- Parsing and Chunking Helpers ---
def extract_text_from_pages(vector: tuple) -> str:
    idx, total, filename = vector
    snippets = []
    try:
        doc = pymupdf.open(filename)
        pages = doc.page_count
        per_proc = (pages + total - 1) // total
        start = idx * per_proc
        end = min(start + per_proc, pages)
        for p in range(start, end):
            try:
                page = doc[p]
                snippets.append(f"### Page {p+1}\n\n" + page.get_text("text") + "\n\n---\n\n")
            except Exception as ex:
                print(f"Worker {idx}: failed page {p} - {ex}")
        doc.close()
    except Exception as ex:
        print(f"Worker {idx}: failed open '{filename}' - {ex}")
    return "".join(snippets)


def run_pymupdf_extraction(filename: str) -> str:
    """
    Extract text by splitting pages across spawned worker processes.
    """
    try:
        num_procs = cpu_count() or 2
        print(f"[{os.getpid()}] Extracting with a spawn-pool of {num_procs} workers.")
        ctx = multiprocessing.get_context("spawn")
        tasks = [(i, num_procs, filename) for i in range(num_procs)]
        with ctx.Pool(processes=num_procs) as pool:
            parts = pool.map(extract_text_from_pages, tasks)
        return "".join(parts)
    except Exception as e:
        print(f"Extraction error: {e}")
        return ""


def recursive_character_split(text: str, max_length: int = 4000, overlap: int = 50) -> List[str]:
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = start + max_length
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # look for natural split
        split = text.rfind("\n\n", start, end) or text.rfind("\n", start, end) or text.rfind(". ", start, end) or end
        chunk = text[start:split].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start+1, split - overlap)
    return chunks


def batch_generator(data: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]


def generate_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]


async def cleanup_namespace(ns: str) -> bool:
    try:
        await run_in_threadpool(pinecone_index.delete, delete_all=True, namespace=ns)
        print(f"[CLEANUP] Deleted namespace: {ns}")
        return True
    except NotFoundException:
        return True
    except Exception as e:
        print(f"Cleanup error: {e}")
        return False


async def wait_for_index_readiness(ns: str, expected: int, max_wait: int = 120) -> bool:
    print(f"[{ns}] Awaiting {expected} vectors...")
    for i in range(max_wait):
        try:
            stats = await run_in_threadpool(pinecone_index.describe_index_stats)
            count = stats.get('namespaces', {}).get(ns, {}).get('vector_count', 0)
            print(f"[{ns}] {count}/{expected} vectors")
            if count >= expected:
                await run_in_threadpool(pinecone_index.query, namespace=ns, top_k=1, vector=[0.0]*dense_dimension)
                print(f"[{ns}] Ready!")
                return True
        except Exception:
            pass
        await asyncio.sleep(1)
    print(f"[{ns}] Readiness timeout.")
    return False

# --- Security & API Models ---
security = HTTPBearer()

def verify_token(creds: HTTPAuthorizationCredentials = Security(security)):
    if not (creds and creds.scheme == "Bearer" and creds.credentials == os.getenv("API_BEARER_TOKEN")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

class SubmissionRequest(BaseModel):
    documents: str
    questions: List[str]

class SubmissionResponse(BaseModel):
    answers: List[str]

# --- Core Processing ---
async def process_single_query(query: str, ns: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            dense, sparse = await asyncio.gather(
                run_in_threadpool(pc.inference.embed, model=dense_model, inputs=[query], parameters={"input_type":"query"}),
                run_in_threadpool(pc.inference.embed, model=sparse_model, inputs=[query], parameters={"input_type":"query"})
            )
            dense_vec = dense[0]['values']
            sparse_payload = {'indices': sparse[0]['sparse_indices'], 'values': sparse[0]['sparse_values']}
            res = await run_in_threadpool(
                pinecone_index.query,
                namespace=ns,
                top_k=100,
                vector=dense_vec,
                sparse_vector=sparse_payload,
                include_metadata=True
            )
            docs = [m['metadata']['text'] for m in res['matches']]
            if not docs and attempt < retries-1:
                await asyncio.sleep(2**attempt)
                continue
            if not docs:
                return "No relevant information found."
            rerank = await run_in_threadpool(
                pc.inference.rerank,
                model=rerank_model,
                query=query,
                documents=docs[:30],
                top_n=10,
                return_documents=True
            )
            top_texts = [d.document.text for d in rerank.data]
            context = "\n\n---\n\n".join(top_texts)
            prompt = (
                "You are an AI assistant tasked with explaining policy documents. "
                "Your response must be factual, clear, and easy to understand, with a supportive tone. "
                "Use ONLY the provided CONTEXT. If insufficient, say the document lacks the needed details.\n\n"
                f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:"
            )
            gen = await models["generation_model"].generate_content_async(prompt)
            return gen.text.strip()
        except Exception as e:
            if attempt < retries-1:
                await asyncio.sleep(2**attempt)
            else:
                return "An internal error occurred while answering."
    return "Failed after retries."

async def process_and_index_document(url: str, ns: str) -> bool:
    temp_path = None
    try:
        print(f"[{ns}] Downloading...")
        async with httpx.AsyncClient() as c:
            r = await c.get(url, follow_redirects=True, timeout=120)
            r.raise_for_status()
        path = urlparse(url).path
        ext = os.path.splitext(path)[1] or mimetypes.guess_extension(r.headers.get('content-type', '')) or '.pdf'
        temp_path = os.path.join(tempfile.gettempdir(), ns+ext)
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(r.content)

        print(f"[{ns}] Parsing with PyMuPDF...")
        text = await run_in_threadpool(run_pymupdf_extraction, temp_path)
        if not text:
            print(f"[{ns}] No text extracted.")
            return False
        chunks = recursive_character_split(text)
        if not chunks:
            return False

        print(f"[{ns}] {len(chunks)} chunks to index.")
        async def upsert_batch(batch, start):
            try:
                dense, sparse = await asyncio.gather(
                    run_in_threadpool(pc.inference.embed, model=dense_model, inputs=batch, parameters={"input_type":"passage"}),
                    run_in_threadpool(pc.inference.embed, model=sparse_model, inputs=batch, parameters={"input_type":"passage"})
                )
                vectors = [{
                    'id': f"chunk-{start+j}",
                    'values': dense[j]['values'],
                    'sparse_values': {'indices': sparse[j]['sparse_indices'], 'values': sparse[j]['sparse_values']},
                    'metadata': {'text': chunk}
                } for j, chunk in enumerate(batch)]
                if vectors:
                    await run_in_threadpool(pinecone_index.upsert, vectors=vectors, namespace=ns)
                    await asyncio.sleep(0.1)
                return True
            except Exception as e:
                print(f"[{ns}] Upsert error at {start}: {e}")
                return False

        tasks = [upsert_batch(batch, i*95) for i, batch in enumerate(batch_generator(chunks, 95))]
        results = await asyncio.gather(*tasks)
        if not all(results):
            await cleanup_namespace(ns)
            return False

        ready = await wait_for_index_readiness(ns, len(chunks))
        return True
    except Exception as e:
        print(f"[{ns}] Critical error: {e}")
        await cleanup_namespace(ns)
        return False
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/v1/hackrx/run", response_model=SubmissionResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: SubmissionRequest):
    print("--- New Request ---")
    print("URL:", request.documents)
    print("Questions:", request.questions)

    ns = f"doc-{generate_url_hash(request.documents)}"
    await cleanup_namespace(ns)
    success = await process_and_index_document(request.documents, ns)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to index document.")

    tasks = [process_single_query(q, ns) for q in request.questions]
    answers = await asyncio.gather(*tasks)
    return SubmissionResponse(answers=answers)
