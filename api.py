# =============================================================================
# FastAPI Backend - REST API layer for MF Prospectus Extraction System
# =============================================================================
#
# This module exposes all backend logic as REST endpoints.
# Key changes vs the previous version:
#   - Sections are discovered DYNAMICALLY (no hardcoded MF_SECTIONS)
#   - Document validation via vision model (score ≥ 65 to proceed)
#   - DocumentScanner performs page-wise scanning for section discovery
#
# Endpoints:
#   POST /upload              - Validate → scan → process → index
#   GET  /status              - Check state (includes validation info)
#   GET  /page-count          - Total pages in the loaded PDF
#   GET  /page/{page_num}     - Render a single PDF page as PNG
#   GET  /sections            - Return dynamically discovered sections
#   POST /extract-section     - Extract one section via RAG + LLM
#   POST /extract-all         - Extract all discovered sections
#   GET  /extracted-sections  - Return all previously extracted content
#   POST /chat                - Chat / Q&A over the document
#   GET  /chunks              - Get chunks (optionally filtered by page)
#
# Run with:  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# =============================================================================

import os
import io
import json
import shutil
import tempfile
import base64
import traceback
from typing import Optional, List, Dict

import fitz  # PyMuPDF – page rendering only
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from document_processor import DocumentProcessor
from document_scanner import DocumentScanner
from rag_engine import RAGEngine
from section_extractor import SectionExtractor

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="MF Prospectus Extraction API",
    description="REST API for processing, extracting, and querying mutual fund prospectus PDFs",
    version="2.0.0",
)

# =============================================================================
# In-memory state (single-user / demo mode)
# =============================================================================

_state: Dict = {
    "pdf_path": None,
    "processor": None,
    "chunks": [],
    "rag_engine": None,
    "section_extractor": None,
    "scanner": None,
    "discovered_sections": {},
    "validation_result": {},
    "total_pages": 0,
}


def _is_loaded() -> bool:
    return _state["rag_engine"] is not None


def _sanitize(obj):
    """Recursively convert to JSON-safe primitives."""
    if obj is None:
        return None
    if isinstance(obj, (int, bool)):
        return obj
    if isinstance(obj, float):
        if obj != obj:
            return 0.0
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(i) for i in obj]
    return str(obj)


# =============================================================================
# Pydantic models
# =============================================================================

class ChatRequest(BaseModel):
    query: str

class ExtractSectionRequest(BaseModel):
    section_key: str


# =============================================================================
# Endpoints
# =============================================================================

# ---- Status -----------------------------------------------------------------

@app.get("/status", tags=["Status"])
def get_status():
    extracted = []
    if _state["section_extractor"]:
        extracted = list(_state["section_extractor"].extracted_sections.keys())
    return {
        "document_loaded": _is_loaded(),
        "total_pages": _state["total_pages"],
        "num_chunks": len(_state["chunks"]),
        "extracted_sections": extracted,
        "discovered_sections_count": len(_state["discovered_sections"]),
        "validation": _state.get("validation_result", {}),
    }


# ---- Upload + validate + scan + process -------------------------------------

@app.post("/upload", tags=["Document"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF.  The pipeline:
      1. VALIDATE  – vision scan of first 5 pages (score ≥ 65 required)
      2. DISCOVER  – vision scan of all pages → section structure
      3. PROCESS   – Docling extraction + chunking
      4. INDEX     – embed chunks into ChromaDB

    If validation fails the response has HTTP 422 with score details.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # ---- 1. Validate ------------------------------------------------
        scanner = DocumentScanner()
        is_valid, score, summary = scanner.validate_document(tmp_path)

        if not is_valid:
            _state["validation_result"] = scanner.validation_result
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Document validation failed – not a mutual fund document",
                    "validation": _sanitize(scanner.validation_result),
                },
            )

        # ---- 2. Discover sections (vision) ------------------------------
        discovered_sections = scanner.discover_sections(tmp_path)

        # ---- 3. Process with Docling ------------------------------------
        processor = DocumentProcessor()
        chunks = processor.process_document(tmp_path)

        # ---- 4. Index into ChromaDB -------------------------------------
        rag_engine = RAGEngine()
        rag_engine.index_chunks(chunks)

        # ---- 5. Section extractor (uses dynamic sections) ---------------
        section_extractor = SectionExtractor(rag_engine, discovered_sections)

        # ---- 6. Page count via PyMuPDF ----------------------------------
        doc = fitz.open(tmp_path)
        total_pages = len(doc)
        doc.close()

        # ---- 7. Persist state -------------------------------------------
        _state.update({
            "pdf_path": tmp_path,
            "processor": processor,
            "chunks": chunks,
            "rag_engine": rag_engine,
            "section_extractor": section_extractor,
            "scanner": scanner,
            "discovered_sections": discovered_sections,
            "validation_result": scanner.validation_result,
            "total_pages": total_pages,
        })

        return {
            "message": "Document processed successfully",
            "filename": file.filename,
            "total_pages": total_pages,
            "num_chunks": len(chunks),
            "sections_discovered": len(discovered_sections),
            "validation": _sanitize(scanner.validation_result),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---- PDF page rendering ----------------------------------------------------

@app.get("/page-count", tags=["Document"])
def page_count():
    if not _is_loaded():
        raise HTTPException(status_code=400, detail="No document loaded.")
    return {"total_pages": _state["total_pages"]}


@app.get("/page/{page_num}", tags=["Document"])
def get_page_image(
    page_num: int,
    zoom: float = Query(1.5),
    bbox: Optional[str] = Query(None),
):
    """Render a 1-indexed PDF page as base64 PNG, with optional bbox highlight."""
    if not _is_loaded():
        raise HTTPException(status_code=400, detail="No document loaded.")

    page_idx = page_num - 1
    if page_idx < 0 or page_idx >= _state["total_pages"]:
        raise HTTPException(status_code=404, detail="Page number out of range.")

    doc = fitz.open(_state["pdf_path"])
    page = doc[page_idx]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    if bbox:
        try:
            from PIL import Image, ImageDraw
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            draw = ImageDraw.Draw(img)
            b = json.loads(bbox)
            page_h = page.rect.height
            x0 = b.get("left", 0) * zoom
            x1 = b.get("right", 100) * zoom
            y0 = (page_h - b.get("top", 0)) * zoom
            y1 = (page_h - b.get("bottom", 100)) * zoom
            if y0 > y1:
                y0, y1 = y1, y0
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
        except Exception as e:
            print(f"Bbox highlight error: {e}")
            png_bytes = pix.tobytes("png")
    else:
        png_bytes = pix.tobytes("png")

    doc.close()
    return {"image_base64": base64.b64encode(png_bytes).decode("utf-8")}


# ---- Sections (dynamic) ----------------------------------------------------

@app.get("/sections", tags=["Sections"])
def get_section_definitions():
    """Return the dynamically discovered section definitions."""
    return _state["discovered_sections"]


@app.post("/extract-section", tags=["Sections"])
def extract_section(req: ExtractSectionRequest):
    if not _is_loaded():
        raise HTTPException(status_code=400, detail="No document loaded.")
    sections = _state["discovered_sections"]
    if req.section_key not in sections:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown section: {req.section_key}. "
                   f"Available: {list(sections.keys())}",
        )
    try:
        section = _state["section_extractor"].extract_section(req.section_key)
        return JSONResponse(content=_sanitize(section.to_dict()))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-all", tags=["Sections"])
def extract_all_sections():
    if not _is_loaded():
        raise HTTPException(status_code=400, detail="No document loaded.")
    try:
        sections = _state["section_extractor"].extract_all_sections()
        result = {k: _sanitize(s.to_dict()) for k, s in sections.items()}
        return JSONResponse(content=result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/extracted-sections", tags=["Sections"])
def get_extracted_sections():
    if not _is_loaded():
        raise HTTPException(status_code=400, detail="No document loaded.")
    secs = _state["section_extractor"].extracted_sections
    return {k: s.to_dict() for k, s in secs.items()}


# ---- Chat -------------------------------------------------------------------

@app.post("/chat", tags=["Chat"])
def chat(req: ChatRequest):
    if not _is_loaded():
        raise HTTPException(status_code=400, detail="No document loaded.")
    try:
        answer, sources = _state["rag_engine"].chat(req.query)
        return JSONResponse(content={
            "answer": answer,
            "sources": _sanitize(sources),
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---- Chunks ----------------------------------------------------------------

@app.get("/chunks", tags=["Chunks"])
def get_chunks(page: Optional[int] = Query(None)):
    if not _is_loaded():
        raise HTTPException(status_code=400, detail="No document loaded.")
    chunks = _state["chunks"]
    if page is not None:
        chunks = [c for c in chunks if c.page_number == page]
    return [c.to_dict() for c in chunks]


# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
