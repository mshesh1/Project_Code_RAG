# Mutual Fund Prospectus RAG System

> An end-to-end RAG pipeline that lets analysts upload any mutual fund prospectus PDF and instantly extract structured sections, classify document content via Vision LLM, and ask natural language questions — powered by Docling, ChromaDB, and two-stage retrieval with LLM reranking.

---

## The Problem

Mutual fund prospectus documents are dense, unstructured PDFs — often 50–200 pages of regulatory text, tables, risk disclosures, and portfolio data. Analysts waste hours manually navigating these documents to extract specific information like expense ratios, risk factors, or portfolio holdings.

---

## The Solution

A full-stack RAG application with a Streamlit frontend and FastAPI backend that:

1. **Validates** the uploaded PDF is actually a mutual fund document (Vision LLM scoring)
2. **Discovers** all sections and subsections dynamically by scanning every page with GPT-4o-mini vision — no hardcoded section lists
3. **Processes** the document with Docling for structure-aware chunking
4. **Indexes** chunks into ChromaDB with OpenAI embeddings
5. **Retrieves** answers using two-stage retrieval: bi-encoder semantic search → LLM reranking
6. **Responds** via a chat interface or structured section extraction

---

## Architecture

```
User (Browser)
     │
     ▼
┌─────────────────────┐
│   Streamlit Frontend │  app.py
│   - Upload UI        │  - Document viewer (PDF page rendering)
│   - Section browser  │  - Section extraction panel
│   - Chat interface   │  - Q&A chatbot
└────────┬────────────┘
         │ HTTP (REST)
         ▼
┌─────────────────────────────────────────────────┐
│              FastAPI Backend  (api.py)           │
│                                                  │
│  POST /upload → validate → discover → process   │
│  POST /chat   → retrieve → rerank → respond     │
│  POST /extract-section → RAG → LLM summarize    │
└──────┬──────────────────┬────────────────────────┘
       │                  │
       ▼                  ▼
┌─────────────┐    ┌──────────────────────────────┐
│DocumentScanner│  │         RAG Pipeline          │
│(document_    │  │                               │
│scanner.py)   │  │  DocumentProcessor            │
│              │  │  (document_processor.py)      │
│ 1. Validate  │  │  - Docling PDF conversion     │
│    vision    │  │  - HybridChunker (512 tokens) │
│    scoring   │  │  - Bounding box extraction    │
│              │  │                               │
│ 2. Discover  │  │  RAGEngine (rag_engine.py)    │
│    sections  │  │  - OpenAI embeddings          │
│    per page  │  │  - ChromaDB vector store      │
│    (sliding  │  │  - Bi-encoder retrieval       │
│    window    │  │  - LLM reranking              │
│    memory)   │  │                               │
│              │  │  SectionExtractor             │
│ 3. Consolidate│  │  (section_extractor.py)      │
│    section   │  │  - Per-section RAG queries    │
│    structure │  │  - LLM summarization          │
└─────────────┘  └──────────────────────────────┘
```

### Two-Stage Retrieval Pipeline

```
User Query
    │
    ▼
Bi-encoder (OpenAI text-embedding-3-small)
    │  ChromaDB similarity search
    │  Returns top-10 candidate chunks
    ▼
LLM Reranker (GPT-4o-mini)
    │  Scores each chunk 0–10 for relevance
    │  Returns top-5 reranked chunks
    ▼
LLM Response (GPT-4o-mini)
    │  Generates answer grounded in chunks
    ▼
Answer + Source Citations
```

### Vision-Based Section Discovery

```
PDF Upload
    │
    ├── Pages 1–5 → Validation scoring
    │   Vision model scores each page (+10 MF / -10 not MF)
    │   Start: 50 | Threshold: 65 | Reject if below threshold
    │
    └── All pages → Section discovery
        Vision model extracts section titles, descriptions,
        keywords per page with sliding window memory
        (last 3 pages kept in LangChain message history)
            │
            ▼
        Text LLM consolidation
        Merges page-level findings → clean section tree
        with deduplication and cross-page merging
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend API | FastAPI + Uvicorn |
| Document Parsing | Docling (structure-aware PDF extraction) |
| Chunking | Docling HybridChunker (512 tokens, merge peers) |
| Vision Model | GPT-4o-mini (validation + section discovery) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB (local persistent) |
| Retrieval | Bi-encoder → LLM reranking (two-stage) |
| LLM | GPT-4o-mini |
| Memory | LangChain sliding window (HumanMessage / AIMessage) |
| PDF Rendering | PyMuPDF (fitz) |
| Language | Python 3.11 |

---

## Key Design Decisions

**Why Docling over PyMuPDF for chunking?**
PyMuPDF extracts raw text without structural awareness. Docling understands document layout — tables stay intact, headings are tagged, reading order is preserved across columns. For financial documents with complex table structures (portfolio holdings, risk matrices), this prevents chunk boundary splits that would destroy the semantic meaning.

**Why two-stage retrieval?**
Bi-encoder (dense vector search) is fast but imprecise — it finds semantically similar chunks but can miss the most relevant one. Adding an LLM reranker as a second stage significantly improves precision at the cost of a small latency increase. For financial document Q&A where accuracy is critical, this tradeoff is worth it.

**Why dynamic section discovery over hardcoded sections?**
Mutual fund documents vary significantly by AMC, scheme type, and regulatory jurisdiction. Hardcoded section lists break on documents that use different headings. Vision-based discovery adapts to any document structure automatically.

**Why sliding window memory for page scanning?**
Section content often spans multiple pages. Without memory context, the vision model has no way to know that page 5 is a continuation of the "Risk Factors" section that started on page 4. The LangChain sliding window (last 3 pages) solves this.

---

## Project Structure

```
mf-prospectus-rag/
├── app.py                  # Streamlit frontend
├── api.py                  # FastAPI backend — all REST endpoints
├── config.py               # Config (models, thresholds, prompts)
├── document_processor.py   # Docling PDF parsing + chunking
├── document_scanner.py     # Vision validation + section discovery
├── rag_engine.py           # ChromaDB + embeddings + reranking
├── section_extractor.py    # Per-section RAG extraction + LLM summary
├── test_apis.py            # API smoke tests
├── MF_Prospectus_Pipeline.ipynb  # End-to-end exploration notebook
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Quickstart

### Prerequisites
- Python 3.11+
- OpenAI API key

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/mf-prospectus-rag.git
cd mf-prospectus-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Start the FastAPI backend
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# 6. In a new terminal, start the Streamlit frontend
streamlit run app.py
```

The app will be available at `http://localhost:8501`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload PDF → validate → discover sections → index |
| `GET` | `/status` | System state (loaded, chunk count, sections) |
| `GET` | `/sections` | Dynamically discovered section definitions |
| `POST` | `/extract-section` | Extract one section via RAG + LLM |
| `POST` | `/extract-all` | Extract all discovered sections |
| `GET` | `/extracted-sections` | All previously extracted sections |
| `POST` | `/chat` | Natural language Q&A over the document |
| `GET` | `/page/{page_num}` | Render PDF page as base64 PNG |
| `GET` | `/chunks` | All indexed chunks (filterable by page) |

Interactive API docs available at `http://localhost:8000/docs` when the backend is running.

---

## Example Usage

**Upload and query a fund document:**

```python
import requests

# Upload
with open("fund_prospectus.pdf", "rb") as f:
    r = requests.post("http://localhost:8000/upload",
                      files={"file": f})
print(r.json())
# {"message": "Document processed", "num_chunks": 87, "sections_discovered": 9}

# Ask a question
r = requests.post("http://localhost:8000/chat",
                  json={"query": "What is the expense ratio?"})
print(r.json()["answer"])
# "The Total Expense Ratio (TER) for the Regular Plan is 1.85% p.a..."

# Extract a specific section
r = requests.post("http://localhost:8000/extract-section",
                  json={"section_key": "risk_factors"})
print(r.json()["summary"])
```

---

## Validation System

The system uses a scoring mechanism to reject non-MF documents:

| Parameter | Value |
|---|---|
| Initial score | 50 |
| Pages scanned | First 5 |
| Score per MF page | +10 |
| Score per non-MF page | -10 |
| Pass threshold | ≥ 65 |

A random PDF, invoice, or unrelated document will be rejected before any processing occurs — saving API costs and preventing garbage-in output.

---

## Built With

- [Docling](https://github.com/DS4SD/docling) — structure-aware document parsing
- [ChromaDB](https://www.trychroma.com/) — local vector store
- [LangChain](https://www.langchain.com/) — LLM orchestration + sliding window memory
- [FastAPI](https://fastapi.tiangolo.com/) — REST API backend
- [Streamlit](https://streamlit.io/) — frontend UI
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF page rendering
- [OpenAI](https://platform.openai.com/) — embeddings, vision, and chat models

---

## Author

Built as part of an AI Engineering portfolio targeting Senior AI Engineer roles in financial services and enterprise AI. Background: 8 years production Python engineering at JPMorgan Chase (TCIO / Athena platform), financial document domain expertise, green card holder based in Dallas-Fort Worth.
