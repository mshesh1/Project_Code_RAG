# =============================================================================
# Document Scanner - Vision-based validation & dynamic section discovery
# =============================================================================
#
# Uses GPT-4o-mini vision to:
#   1. VALIDATE whether a PDF is a mutual fund document (scoring system)
#   2. DISCOVER sections/subsections dynamically by scanning each page
#   3. Maintain LangChain-based sliding-window memory of last 3 pages
#
# The discovered sections replace the old hardcoded MF_SECTIONS.
# =============================================================================

import json
import base64
from typing import List, Dict, Tuple

import fitz  # PyMuPDF for page rendering

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from config import (
    OPENAI_API_KEY,
    VISION_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    VALIDATION_INITIAL_SCORE,
    VALIDATION_THRESHOLD,
    VALIDATION_SCORE_DELTA,
    VALIDATION_MAX_PAGES,
    PAGE_MEMORY_WINDOW,
    DOCUMENT_VALIDATION_PROMPT,
    PAGE_SECTION_SCAN_PROMPT,
    SECTION_CONSOLIDATION_PROMPT,
)


# =============================================================================
# Sliding-Window Memory (LangChain messages)
# =============================================================================

class SlidingWindowMemory:
    """
    LangChain-based sliding window memory for page-by-page scanning.

    Stores HumanMessage / AIMessage pairs (one pair per page).
    Returns the most recent `window_size` exchanges as context so
    the vision model can see continuity across pages.
    """

    def __init__(self, window_size: int = 3):
        self.messages: List = []          # LangChain message objects
        self.window_size: int = window_size

    def add_exchange(self, user_input: str, ai_output: str):
        """Record one page-analysis exchange."""
        self.messages.append(HumanMessage(content=user_input))
        self.messages.append(AIMessage(content=ai_output))

    def get_context_string(self) -> str:
        """Return a text block of the last *window_size* AI responses."""
        recent = self.messages[-(self.window_size * 2):]
        if not recent:
            return ""
        parts = []
        for msg in recent:
            if isinstance(msg, AIMessage):
                parts.append(msg.content)
        if not parts:
            return ""
        return "CONTEXT FROM PREVIOUS PAGES:\n" + "\n".join(parts)

    def get_messages(self):
        """Return the raw LangChain messages in the current window."""
        return self.messages[-(self.window_size * 2):]


# =============================================================================
# DocumentScanner
# =============================================================================

class DocumentScanner:
    """
    Vision-based document scanner.

    Public API
    ----------
    validate_document(pdf_path)  → (is_valid, score, summary)
    discover_sections(pdf_path)  → dict[section_key → section_config]
    """

    def __init__(self):
        # Vision LLM (for page images)
        self.vision_llm = ChatOpenAI(
            model=VISION_MODEL,
            temperature=0.1,
            max_tokens=2000,
            openai_api_key=OPENAI_API_KEY,
        )
        # Text LLM (for consolidation – no images needed)
        self.text_llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=4000,
            openai_api_key=OPENAI_API_KEY,
        )
        self.discovered_sections: Dict = {}
        self.validation_result: Dict = {}

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _render_page_to_base64(pdf_path: str, page_idx: int,
                               zoom: float = 1.5) -> str:
        """Render a 0-indexed PDF page to a base64 PNG string."""
        doc = fitz.open(pdf_path)
        page = doc[page_idx]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        doc.close()
        return base64.b64encode(png_bytes).decode("utf-8")

    @staticmethod
    def _get_page_count(pdf_path: str) -> int:
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count

    def _call_vision(self, prompt_text: str, image_base64: str) -> str:
        """Send text + image to the vision model and return the response."""
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    },
                },
            ]
        )
        response = self.vision_llm.invoke([message])
        return response.content

    @staticmethod
    def _parse_json_response(text: str) -> Dict:
        """Extract JSON from an LLM response (handles markdown fences)."""
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {}

    # =================================================================
    # 1. DOCUMENT VALIDATION
    # =================================================================

    def validate_document(self, pdf_path: str) -> Tuple[bool, int, str]:
        """
        Validate whether a PDF is a mutual-fund document.

        Scans the first *VALIDATION_MAX_PAGES* pages with the vision model.
        Starts at *VALIDATION_INITIAL_SCORE* (50).
        Each page adds ``+DELTA`` (MF content) or ``-DELTA`` (not MF).
        The document is **valid** when the final score ≥ *VALIDATION_THRESHOLD*.

        Returns
        -------
        (is_valid, final_score, summary_reason)
        """
        total_pages = self._get_page_count(pdf_path)
        pages_to_scan = min(total_pages, VALIDATION_MAX_PAGES)

        score = VALIDATION_INITIAL_SCORE
        reasons: List[str] = []

        print(f"Validating document ({pages_to_scan} pages to scan) …")
        print(f"  Initial score: {score}")

        for page_idx in range(pages_to_scan):
            try:
                image_b64 = self._render_page_to_base64(pdf_path, page_idx)
                response_text = self._call_vision(
                    DOCUMENT_VALIDATION_PROMPT, image_b64
                )
                result = self._parse_json_response(response_text)

                is_mf = result.get("is_mf_document", False)
                reason = result.get("reason", "No reason provided")

                if is_mf:
                    score += VALIDATION_SCORE_DELTA
                    print(
                        f"  Page {page_idx + 1}: "
                        f"+{VALIDATION_SCORE_DELTA} (MF content) → {score}  "
                        f"| {reason}"
                    )
                else:
                    score -= VALIDATION_SCORE_DELTA
                    print(
                        f"  Page {page_idx + 1}: "
                        f"-{VALIDATION_SCORE_DELTA} (not MF) → {score}  "
                        f"| {reason}"
                    )

                reasons.append(f"Page {page_idx + 1}: {reason}")

            except Exception as e:
                print(f"  Page {page_idx + 1}: Error — {e}")
                reasons.append(f"Page {page_idx + 1}: Error — {e}")

        is_valid = score >= VALIDATION_THRESHOLD
        summary = f"Score: {score}/100. " + " | ".join(reasons[:3])

        self.validation_result = {
            "is_valid": is_valid,
            "score": score,
            "threshold": VALIDATION_THRESHOLD,
            "pages_scanned": pages_to_scan,
            "reasons": reasons,
            "summary": summary,
        }

        tag = "VALID ✓" if is_valid else "REJECTED ✗"
        print(
            f"\nValidation: {tag}  "
            f"(score {score}, threshold {VALIDATION_THRESHOLD})"
        )
        return is_valid, score, summary

    # =================================================================
    # 2. SECTION DISCOVERY
    # =================================================================

    def discover_sections(self, pdf_path: str) -> Dict:
        """
        Scan every page with the vision model to build a dynamic
        section / subsection structure.

        Uses a ``SlidingWindowMemory`` (LangChain HumanMessage / AIMessage)
        of the last *PAGE_MEMORY_WINDOW* pages so the model can see
        continuity across page boundaries.

        Returns
        -------
        dict  –  ``{section_key: {title, description, subsections, keywords, pages}}``
                 (same shape the rest of the pipeline expects)
        """
        total_pages = self._get_page_count(pdf_path)
        memory = SlidingWindowMemory(window_size=PAGE_MEMORY_WINDOW)
        page_analyses: List[Dict] = []

        print(f"\nDiscovering sections across {total_pages} pages …")

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            print(f"  Scanning page {page_num}/{total_pages} …")

            try:
                # Build prompt with memory context
                memory_context = memory.get_context_string()
                prompt = PAGE_SECTION_SCAN_PROMPT.format(
                    memory_context=memory_context,
                    page_num=page_num,
                )

                # Render page & call vision
                image_b64 = self._render_page_to_base64(pdf_path, page_idx)
                response_text = self._call_vision(prompt, image_b64)
                result = self._parse_json_response(response_text)

                if result:
                    page_analyses.append(result)

                    # Save analysis into LangChain sliding-window memory
                    page_summary = result.get(
                        "page_summary", f"Page {page_num} analyzed"
                    )
                    section_titles = [
                        s.get("title", "")
                        for s in result.get("sections_found", [])
                    ]
                    memory.add_exchange(
                        user_input=f"Scan page {page_num}",
                        ai_output=(
                            f"Page {page_num}: {page_summary}. "
                            f"Sections: {', '.join(section_titles)}"
                        ),
                    )
                    print(
                        f"    Found {len(section_titles)} sections: "
                        f"{section_titles}"
                    )
                else:
                    print(f"    No structured response for page {page_num}")

            except Exception as e:
                print(f"    Error scanning page {page_num}: {e}")

        # Consolidate page-level findings into clean section tree
        print("\nConsolidating sections …")
        self.discovered_sections = self._consolidate_sections(page_analyses)

        print(f"Discovered {len(self.discovered_sections)} sections:")
        for key, sec in self.discovered_sections.items():
            n_sub = len(sec.get("subsections", []))
            n_kw = len(sec.get("keywords", []))
            print(f"  {sec['title']}  ({n_sub} subsections, {n_kw} keywords)")

        return self.discovered_sections

    # -----------------------------------------------------------------

    def _consolidate_sections(self, page_analyses: List[Dict]) -> Dict:
        """
        Use the text LLM to merge page-by-page findings into a single
        deduplicated section/subsection structure.
        """
        # Build readable text for all page analyses
        analyses_text = ""
        for analysis in page_analyses:
            page_num = analysis.get("page_number", "?")
            analyses_text += f"\n--- Page {page_num} ---\n"
            analyses_text += (
                f"Summary: {analysis.get('page_summary', 'N/A')}\n"
            )
            for sec in analysis.get("sections_found", []):
                analyses_text += f"  Section: {sec.get('title', 'Unknown')}\n"
                analyses_text += (
                    f"    Description: {sec.get('description', '')}\n"
                )
                kw = sec.get("keywords", [])
                analyses_text += f"    Keywords: {', '.join(kw)}\n"
                subs = sec.get("subsections", [])
                if subs:
                    analyses_text += f"    Subsections: {', '.join(subs)}\n"
                analyses_text += (
                    f"    Continuation: {sec.get('is_continuation', False)}\n"
                )
                analyses_text += (
                    f"    Content: {sec.get('content_summary', '')}\n"
                )

        prompt = SECTION_CONSOLIDATION_PROMPT.format(
            page_analyses=analyses_text
        )
        response = self.text_llm.invoke(prompt)
        result = self._parse_json_response(response.content)

        # Convert the list into a keyed dict
        sections_dict: Dict = {}
        for idx, sec in enumerate(result.get("sections", [])):
            key = sec.get("section_key", f"section_{idx}")
            sections_dict[key] = {
                "title": sec.get("title", f"Section {idx + 1}"),
                "description": sec.get("description", ""),
                "subsections": sec.get("subsections", []),
                "keywords": sec.get("keywords", []),
                "pages": sec.get("pages", []),
            }

        return sections_dict


# =====================================================================
# Quick smoke-test
# =====================================================================
if __name__ == "__main__":
    import os

    pdf_path = "../Fund Facts - HDFC Income Fund - December 2025 [a].pdf"
    if os.path.exists(pdf_path):
        scanner = DocumentScanner()

        # 1. Validate
        is_valid, score, summary = scanner.validate_document(pdf_path)
        print(f"\nValidation → valid={is_valid}, score={score}")

        if is_valid:
            # 2. Discover sections
            sections = scanner.discover_sections(pdf_path)
            print(f"\nDiscovered {len(sections)} sections:")
            for k, v in sections.items():
                print(f"  {k}: {v['title']}")
                for sub in v["subsections"]:
                    print(f"    - {sub}")
    else:
        print(f"PDF not found: {pdf_path}")
