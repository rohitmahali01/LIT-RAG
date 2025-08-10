# Advanced RAG Pipeline with Caching and Hybrid Parsing

![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)![Framework](https://img.shields.io/badge/Framework-FastAPI-05998b)

This repository contains the source code for a high-performance, asynchronous Retrieval-Augmented Generation (RAG) API. It is built with FastAPI and leverages Google's Gemini 1.5 Flash for generation, Pinecone for vector search, and a sophisticated hybrid parsing strategy for document ingestion. The system is optimized for speed and efficiency through multiprocessing and a smart caching layer.

## ✨ Core Features

*   **Asynchronous by Design:** Built on FastAPI and `asyncio` to handle a high volume of concurrent requests without blocking.
*   **Intelligent Caching:** Uses Pinecone namespaces as a persistent cache. Documents are processed only once; subsequent requests for the same URL skip the expensive ingestion step entirely.
*   **High-Performance Hybrid Parsing:**
    *   **For PDFs:** Utilizes `PyMuPDF` with Python's `multiprocessing` to process pages in parallel across all available CPU cores, drastically reducing ingestion time.
    *   **For Other Formats:** Leverages the `unstructured.io` library for semantic chunking of diverse file types (DOCX, HTML, TXT, etc.).
*   **Advanced Hybrid Search:** Combines dense vectors (from Google's `text-embedding-004`) and sparse vectors (SPLADE) to achieve superior retrieval accuracy, blending semantic meaning with keyword precision.
*   **Cohere Reranking:** Employs a reranking step to refine search results, ensuring only the most relevant context is passed to the language model.
*   **Security-Hardened Prompting:** The generation prompt is engineered to prevent prompt injection attacks by instructing the model to treat all retrieved context strictly as data, not instructions.
*   **Production-Ready:** Includes robust error handling, request retries with exponential backoff, and secure bearer token authentication.
*   **Specialized Handlers:** Contains flexible logic to bypass the RAG pipeline for specific, non-document-based challenges.

## ⚙️ Architectural Flow

The system follows an intelligent, cache-aware workflow for every request.

```mermaid
graph TD
    A[Client Request: URL + Questions] --> B{Cache Check};
    B -- Cache MISS --> C[Ingestion Pipeline];
    C -- 1. Download & Parse --> D[Hybrid Parser];
    D -- PDF --> E[PyMuPDF + Multiprocessing];
    D -- Other --> F[Unstructured.io];
    E --> G[Chunk Text];
    F --> G;
    G -- 2. Embed Chunks --> H[Create Dense & Sparse Vectors];
    H -- 3. Index --> I[Upsert to Pinecone Namespace];
    B -- Cache HIT --> J[Query Processing];
    I --> J;

    subgraph Query Processing
        J -- 1. Hybrid Search --> K[Retrieve Chunks from Pinecone];
        K -- 2. Rerank --> L[Cohere Rerank API];
        L -- 3. Generate --> M[Gemini 1.5 Flash];
    end

    M --> N[API Response: Answers];
    A --> N;
```

## 🛠️ Technology Stack

*   **Backend:** FastAPI, Uvicorn
*   **AI & NLP:**
    *   **Generation:** Google Gemini 1.5 Flash
    *   **Vector Database:** Pinecone
    *   **Embedding Models:** Google `text-embedding-004` (Dense), SPLADE (Sparse)
    *   **Reranking:** Cohere Rerank API
*   **Parsing:** PyMuPDF, Unstructured.io
*   **Async & HTTP:** `asyncio`, `httpx`, `aiofiles`
*   **Concurrency:** `multiprocessing`
*   **Configuration:** `python-dotenv`

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### 1. Prerequisites

*   Python 3.9+
*   A Pinecone account and API key.
*   A Google AI Studio API key.
*   A Cohere API key (if you plan to use their reranker).

### 2. Clone the Repository

```bash
git clone [<[your-repository-url](https://github.com/rohitmahali01/LIT-RAG)>]
cd <repository-directory>
```

### 3. Install Dependencies

It's recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```
*(Note: You will need to create a `requirements.txt` file based on the imports in the script.)*

A possible `requirements.txt`:
```
fastapi
uvicorn[standard]
python-dotenv
google-generativeai
pinecone-client
cohere
unstructured
pymupdf
httpx
aiofiles
splade
```


### 4. Configure Environment Variables

Create a file named `.env` in the root directory of the project and add your credentials. Use the `.env.example` as a template:

**.env.example**```env
# --- API Keys ---
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
COHERE_API_KEY="YOUR_COHERE_API_KEY" # Needed for reranking

# --- Security ---
# A secret token you will use to authenticate with the API endpoint
API_BEARER_TOKEN="YOUR_SECRET_SECURE_TOKEN"
```

### 5. Run the Application

Start the development server using Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will now be running at `http://localhost:8000`. You can access the interactive documentation at `http://localhost:8000/docs`.

## 📖 API Usage

The primary endpoint for processing documents is `/api/v1/hackrx/run`.

*   **Method:** `POST`
*   **URL:** `http://localhost:8000/api/v1/hackrx/run`
*   **Authentication:** `Bearer Token`

### Request Body

The request must be a JSON object with the following structure:

```json
{
  "documents": "URL_OF_THE_DOCUMENT_TO_PROCESS",
  "questions": [
    "Your first question about the document?",
    "Your second question about the document?"
  ]
}
```

### Example `curl` Request

Replace `YOUR_SECRET_SECURE_TOKEN` with the value you set in your `.env` file.

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/hackrx/run' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer YOUR_SECRET_SECURE_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": "https://arxiv.org/pdf/1706.03762.pdf",
    "questions": [
        "What is the title of the paper?",
        "What is a transformer?"
    ]
}'
```

### Success Response

The response will be a JSON object containing a list of answers corresponding to the list of questions.

```json
{
  "answers": [
    "The title of the paper is 'Attention Is All You Need'.",
    "A Transformer is a network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. It has been shown to be highly effective for sequence transduction tasks like machine translation."
  ]
}
'''
