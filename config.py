# =============================================================================
# Configuration file for MF Prospectus Extraction System
# =============================================================================
#
# Sections are NO LONGER hardcoded. They are discovered dynamically by
# scanning each page with a vision model (see document_scanner.py).
# =============================================================================

import os

# OpenAI API Key — set via environment variable, never hardcode here
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ChromaDB settings
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "mf_prospectus"

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# LLM settings
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.1

# Vision model (GPT-4o-mini supports vision)
VISION_MODEL = "gpt-4o-mini"

# Chunking settings
MAX_TOKENS_PER_CHUNK = 512
TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Re-ranker settings
RERANK_TOP_K = 10  # Initial retrieval count
FINAL_TOP_K = 5    # After re-ranking

# =============================================================================
# DOCUMENT VALIDATION SETTINGS
# =============================================================================
# Vision-based scoring to determine if upload is a valid MF document.
# Start at VALIDATION_INITIAL_SCORE, scan first N pages.
# Each page adds +DELTA (MF content) or -DELTA (not MF).
# Valid if final score >= THRESHOLD.

VALIDATION_INITIAL_SCORE = 50
VALIDATION_THRESHOLD = 65
VALIDATION_SCORE_DELTA = 10
VALIDATION_MAX_PAGES = 5

# Sliding-window size: keep last N pages in memory during section scanning
PAGE_MEMORY_WINDOW = 3

# =============================================================================
# PROMPTS
# =============================================================================

# -- Document validation (vision) -- NOT used with .format() ----------------
DOCUMENT_VALIDATION_PROMPT = """You are a financial document classifier.

Look at this PDF page image and determine whether it belongs to a
**mutual fund prospectus, factsheet, scheme information document (SID),
or key information memorandum (KIM)**.

Indicators of a mutual fund document:
- Fund name, AMC (Asset Management Company) name
- NAV, AUM, expense ratio, benchmark mentions
- Investment objective, asset allocation, portfolio holdings
- SEBI registration, regulatory disclaimers
- Risk-o-meter, riskometer
- SIP details, entry/exit load
- Performance/returns data
- Fund manager information

Respond with EXACTLY this JSON (no markdown fences):
{"is_mf_document": true, "confidence": "high", "reason": "brief explanation"}
or
{"is_mf_document": false, "confidence": "high", "reason": "brief explanation"}"""

# -- Page section scan (vision) -- used with .format(memory_context, page_num)
PAGE_SECTION_SCAN_PROMPT = """You are an expert financial document analyst.

Analyze this PDF page image and identify all **sections and subsections**
visible on the page.

{memory_context}

INSTRUCTIONS:
1. Identify every distinct section / heading / subsection on this page
2. For each section determine:
   - The title as it appears on the page
   - A brief description of what information it contains
   - Key topics, terms, or data points (keywords for search)
   - Any subsection headings nested under it
   - Whether it is a continuation from a previous page
3. Capture tables, charts, headers, footers, disclaimers
4. If a section from a previous page continues here, note that

Respond with EXACTLY this JSON (no markdown fences):
{{
    "page_number": {page_num},
    "sections_found": [
        {{
            "title": "Section title as it appears",
            "description": "What information this section contains",
            "keywords": ["keyword1", "keyword2", "keyword3"],
            "subsections": ["Sub-heading 1", "Sub-heading 2"],
            "is_continuation": false,
            "content_summary": "One-line summary of this section on this page"
        }}
    ],
    "page_summary": "One-line summary of what this entire page covers"
}}"""

# -- Consolidation prompt -- used with .format(page_analyses)
SECTION_CONSOLIDATION_PROMPT = """You are an expert financial document analyst.

Below are page-by-page analyses of a mutual fund document.
Consolidate them into a clean, deduplicated section/subsection structure.

PAGE-BY-PAGE FINDINGS:
{page_analyses}

INSTRUCTIONS:
1. Merge sections that span multiple pages into single entries
2. Group related subsections under their parent sections
3. Remove duplicates
4. Create a clean hierarchical structure
5. For each section compile ALL relevant keywords from all pages
6. Generate a section_key (lowercase, underscores, no special chars)
7. Order sections logically as they appear in the document

Respond with EXACTLY this JSON (no markdown fences):
{{
    "sections": [
        {{
            "section_key": "unique_key_like_this",
            "title": "Section Title",
            "description": "What this section covers",
            "subsections": ["Subsection 1", "Subsection 2"],
            "keywords": ["keyword1", "keyword2"],
            "pages": [1, 2]
        }}
    ]
}}"""

# -- Section extraction (works with dynamic sections) -----------------------
SECTION_EXTRACTION_PROMPT = """You are an expert financial analyst specializing in mutual fund prospectus analysis.

Given the following chunks from a mutual fund prospectus document, extract information relevant to the section: "{section_title}"

Section Description: {section_description}
Subsections to look for: {subsections}

CHUNKS:
{chunks}

INSTRUCTIONS:
1. Extract all relevant information for this section from the provided chunks
2. Organize the information under appropriate subsections
3. If information for a subsection is not found, indicate "Not found in document"
4. Be precise and quote exact figures when available (percentages, amounts, dates)
5. Maintain the original meaning - do not interpret or add assumptions

OUTPUT FORMAT:
Provide a structured summary with:
- Main findings for the section
- Details for each subsection found
- Key figures and data points
- Any important notes or caveats

YOUR RESPONSE:"""

# -- Chat prompt -------------------------------------------------------------
CHAT_SYSTEM_PROMPT = """You are an expert financial advisor assistant helping users understand mutual fund prospectus documents.

You have access to the following document chunks that were retrieved based on the user's query.

IMPORTANT GUIDELINES:
1. Answer ONLY based on the information in the provided document chunks
2. If the information is not in the chunks, say "I couldn't find this information in the document"
3. Be precise with numbers, percentages, and dates
4. Cite specific sections or pages when possible
5. Explain financial terms if the user seems unfamiliar
6. Be helpful but never make up information

DOCUMENT CHUNKS:
{context}

USER QUERY: {query}

YOUR RESPONSE:"""

# -- Re-rank prompt ----------------------------------------------------------
RERANK_PROMPT = """Given a query and a list of document chunks, rate how relevant each chunk is to answering the query.

Query: {query}

For each chunk, provide a relevance score from 0 to 10:
- 10: Directly answers the query with specific information
- 7-9: Highly relevant, contains useful related information
- 4-6: Somewhat relevant, may have partial information
- 1-3: Marginally relevant
- 0: Not relevant at all

Chunks to evaluate:
{chunks}

Return your response as a JSON array of objects with 'chunk_id' and 'score' fields.
Example: [{{"chunk_id": 0, "score": 8}}, {{"chunk_id": 1, "score": 3}}, ...]

YOUR RESPONSE (JSON only):"""
