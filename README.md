# LIT RAG: Top 38 Semifinalist @ HackRx

![HackRx Logo](https://img.shields.io/badge/Hackathon-HackRx-blue.svg)![Status](https://img.shields.io/badge/Status-Top%2038%20Semifinalist-brightgreen.svg)![Language](https://img.shields.io/badge/Language-Python-3776AB.svg?logo=python)![Framework](https://img.shields.io/badge/Framework-FastAPI-009688.svg?logo=fastapi)![AI/ML](https://img.shields.io/badge/LLM-Gemini%201.5-F4B400.svg?logo=google)![VectorDB](https://img.shields.io/badge/Vector%20DB-Pinecone-E57300.svg?logo=pinecone)

This repository contains the source code for our submission to the **HackRx 6.0 Hackathon 2025**, where we proudly finished as **Top 38 Semifinalists among 6000 participants**. The project is a highly optimized, intelligent, and scalable Retrieval-Augmented Generation (RAG) pipeline built to tackle complex document analysis challenges under pressure.

##  The Challenge

The hackathon required participants to build a robust API that could:
1.  **Ingest and process** documents from a given URL, handling various formats like PDF, TXT, DOCX, etc.
2.  **Answer questions** based *exclusively* on the content of the provided document.
3.  **Handle special "puzzle" endpoints** that required multi-step logic, external API calls, and even multi-modal analysis of documents.
4.  **Perform efficiently and reliably** under a stateless, serverless-style evaluation environment.
5.  **Defend against prompt injection** and other adversarial attacks embedded within the documents.

##  Our Solution: An Advanced Hybrid RAG Pipeline

We engineered a solution that combines high-performance parsing, an advanced hybrid search strategy, and intelligent caching to deliver fast and accurate answers. The system was designed for scalability, reliability, and clever problem-solving.

## Performance Highlight: From URL to Answer in Under 15 Seconds

Our proudest achievement is the system's raw speed. The entire end-to-end pipeline—from receiving a URL, downloading a **1000-page document**, parsing its content, embedding and indexing it, to finally querying and generating a response—is completed in **under 15 seconds**.

This level of performance, achieved through parallel processing and optimized architecture, was a key factor in our success and demonstrates the system's readiness for real-world, high-throughput applications.


### Key Features & Architecture

#### 1. High-Performance Hybrid Parsing Strategy
To maximize ingestion speed, we implemented a hybrid parsing strategy:
-   ** For PDFs**: We use a custom **multiprocessing parser** built with `PyMuPDF`. This approach bypasses slower, general-purpose libraries and leverages all available CPU cores to extract text in parallel, making it significantly faster for large PDF documents.
-   ** For Other Formats**: We use the powerful `unstructured` library, which automatically handles a wide variety of file types (`.txt`, `.docx`, etc.), ensuring broad compatibility.

#### 2. Advanced RAG Pipeline
Our core pipeline is a sophisticated implementation of the RAG pattern:
-   **Embedding**: We generate both dense (`llama-text-embed-v2`) and sparse (`pinecone-sparse-english-v0`) vectors for each document chunk to enable powerful hybrid search.
-   **Retrieval**: Pinecone Vector Database performs a hybrid search using both vector types to retrieve the most semantically and keyword-relevant chunks.
-   **Reranking**: We use `cohere-rerank-3.5` to re-order the retrieved results, pushing the most relevant context to the top for the final generation step.
-   **Generation**: **Google's Gemini 1.5 Flash** model generates the final answer. We chose Gemini 1.5 for its large context window, speed, and crucial multi-modal capabilities.
-   **Security**: A robust system prompt instructs the LLM to treat all document content strictly as data, ignoring any embedded malicious directives (like "HackRx" instructions).

#### 3. Intelligent Caching Layer
To prevent redundant processing of the same document, we implemented a smart caching layer using **Pinecone namespaces**.
-   Before processing, the system generates a unique hash from the document URL and checks if a corresponding namespace already exists in the Pinecone index.
-   **Cache HIT**: If the namespace exists and contains vectors, the entire ingestion step is skipped, saving significant time and compute resources.
-   **Cache MISS**: If the namespace is empty or doesn't exist, the system proceeds with the ingestion pipeline and populates the namespace for future requests.

#### 4. Dynamic & Multi-Modal Challenge Solvers
A key part of the hackathon was solving unique puzzles. We modularized this logic into a separate `challenge.py` file.
-   **The Flight Puzzle**: This required extracting a landmark for a specific city from a PDF. Our masterstroke here was using the **multi-modal capabilities of Gemini 1.5**. Instead of parsing the text (which was designed to be ambiguous), we sent the entire PDF file directly to Gemini and asked it to *visually* find the landmark associated with the city's *last appearance* in the document. This innovative approach solved the puzzle reliably.
-   **The Dynamic Token**: This challenge involved fetching a secret token from an HTML page. Our solver fetches the page and uses simple string manipulation to parse the token, demonstrating quick and effective handling of API-based puzzles.

##  Code Structure

The project is organized for clarity and separation of concerns:

```
.
├── main.py             # Main FastAPI application, routing, and RAG logic
├── challenge.py        # Handlers for the special hackathon puzzles (Flight Puzzle, Token)
├── requirements.txt    # List of Python dependencies
└── .env.example        # Example environment variables file
```
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

## 🛠 How to Run the Project

### Prerequisites
-   Python 3.9+
-   An API key for Google (for Gemini)
-   An API key for Pinecone

### Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root directory by copying the `.env.example` file. Then, fill in your API keys:
    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    API_BEARER_TOKEN="YOUR_SECRET_BEARER_TOKEN" # Create a secure token for your API
    ```

4.  **Run the FastAPI server:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The server will be running at `http://localhost:8000`.

##  API Endpoint

The primary endpoint for the RAG pipeline.

-   **URL**: `/api/v1/hackrx/run`
-   **Method**: `POST`
-   **Authentication**: `Bearer Token`

**Request Body:**
```json
{
    "documents": "https://hackrx.blob.core.windows.net/hackrx/rounds/News.pdf?sv=2023-01-03&spr=https&st=2025-08-07T17%3A10%3A11Z&se=2026-08-08T17%3A10%3A00Z&sr=b&sp=r&sig=ybRsnfv%2B6VbxPz5xF7kLLjC4ehU0NF7KDkXua9ujSf0%3D",
    "questions": [
        "ട്രംപ് ഏത് ദിവസമാണ് 100% ശുൽകം പ്രഖ്യാപിച്ചത്?",
        "ഏത് ഉത്പന്നങ്ങൾക്ക് ഈ 100% ഇറക്കുമതി ശുൽകം ബാധകമാണ്?",
        "ഏത് സാഹചര്യത്തിൽ ഒരു കമ്പനിയ്ക്ക് ഈ 100% ശുൽകത്തിൽ നിന്നും ഒഴികെയാക്കും?",
        "What was Apple’s investment commitment and what was its objective?",
        "What impact will this new policy have on consumers and the global market?"
    ]
}
```

**Response Body:**
```json
{
  "answers": [
    "ട്രംപ് ഓഗസ്റ്റ് 6, 2025 ന് 100% ശുൽക്കം പ്രഖ്യാപിച്ചു.",
    "വിദേശത്ത് നിർമ്മിച്ച കമ്പ്യൂട്ടർ ചിപ്പുകൾക്കും സെമികണ്ടക്ടറുകൾക്കും ഈ 100% ഇറക്കുമതി ശുൽകം ബാധകമാണ്.  യുഎസിൽ നിർമ്മിക്കാൻ പ്രതിജ്ഞാബദ്ധരായ കമ്പനികൾക്ക് ഈ ശുൽകം ബാധകമല്ല.",
    "യുഎസിൽ നിർമ്മിക്കാൻ പ്രതിജ്ഞാബദ്ധരായ കമ്പനികൾക്ക് ഈ 100% ശുൽകം ബാധകമല്ല.",
    "Apple announced a $600 billion future investment.  The provided text does not explicitly state Apple's objective for this investment, but it implies that the investment was made in light of the 100% tariff imposed on foreign-made computer chips and semiconductors.  The context suggests this tariff was intended to boost American domestic manufacturing and reduce foreign reliance.  Therefore, Apple's investment could be inferred as an attempt to align with and benefit from this policy shift.",
    "The new policy, imposing a 100% tariff on imported computer chips and semiconductors, aims to boost American domestic manufacturing and reduce foreign dependence.  While beneficial for US-based companies committed to domestic manufacturing (like Apple, which announced a $600 billion investment), the impact on consumers is likely to be increased prices.  The global market will likely see disruptions due to reduced access to US markets for foreign chip manufacturers, leading to potential trade conflicts and retaliatory measures."
  ]
}
```

## 🏆 Hackathon Achievement

We are proud to have finished as **Top 38 Semifinalists** in this competitive hackathon. This result is a testament to our robust architecture, innovative use of multi-modal AI, and efficient engineering.

---

##  The Team

Meet the OGs behind the Lit Rag. We are a team of passionate developers who collaborated to build this solution for the HackRx 6.0 hackathon.

-   **Rohit Mahali** - *Team Leader* - [LinkedIn Profile](https://www.linkedin.com/in/rohit-mahali-013949298/)
-   **Sahil Singh** - [LinkedIn Profile](https://www.linkedin.com/in/sahil-singh-51875b361/)
-   **Md Tarik Anvar** - [LinkedIn Profile](https://www.linkedin.com/in/tarik-anvar/)
-   **Nishan Das** - [LinkedIn Profile](https://www.linkedin.com/in/nishan-das-963b10312/)
-   **Mayank Suman** - [LinkedIn Profile](https://www.linkedin.com/in/mayank-suman/)

---
