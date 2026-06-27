# =============================================================================
# Section Extractor - Dynamic section extraction using discovered structure
# =============================================================================
#
# Sections are NO LONGER hardcoded.  They come from DocumentScanner
# (vision-based page scanning) and are passed in at construction time.
# Keywords discovered by the scanner are used to improve RAG queries.
# =============================================================================

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI

from config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SECTION_EXTRACTION_PROMPT,
)
from rag_engine import RAGEngine


@dataclass
class SubsectionContent:
    """Content for a subsection."""
    title: str
    content: str
    chunks: List[Dict] = field(default_factory=list)


@dataclass
class SectionContent:
    """Content for a main section."""
    section_key: str
    title: str
    description: str
    summary: str
    subsections: List[SubsectionContent] = field(default_factory=list)
    all_chunks: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "section_key": self.section_key,
            "title": self.title,
            "description": self.description,
            "summary": self.summary,
            "subsections": [
                {
                    "title": sub.title,
                    "content": sub.content,
                    "chunks": sub.chunks,
                }
                for sub in self.subsections
            ],
            "all_chunks": self.all_chunks,
        }


class SectionExtractor:
    """
    Extracts sections from a mutual fund prospectus using
    *dynamically discovered* section definitions + RAG retrieval + LLM.

    The constructor receives ``discovered_sections`` — a dict produced
    by ``DocumentScanner.discover_sections()`` with the shape::

        {
            "section_key": {
                "title": str,
                "description": str,
                "subsections": [str, ...],
                "keywords": [str, ...],
                "pages": [int, ...]
            }
        }
    """

    def __init__(self, rag_engine: RAGEngine,
                 discovered_sections: Dict[str, Dict]):
        self.rag = rag_engine
        self.sections = discovered_sections          # dynamic!
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        )
        self.extracted_sections: Dict[str, SectionContent] = {}

    # -----------------------------------------------------------------
    # Single section
    # -----------------------------------------------------------------

    def extract_section(self, section_key: str) -> SectionContent:
        """Extract content for a single section via RAG + LLM."""
        if section_key not in self.sections:
            raise ValueError(f"Unknown section: {section_key}")

        cfg = self.sections[section_key]
        title = cfg["title"]
        description = cfg.get("description", "")
        subsections = cfg.get("subsections", [])
        keywords = cfg.get("keywords", [])

        print(f"Extracting section: {title}")

        # Retrieve relevant chunks (keywords improve query)
        chunks = self.rag.search_for_section(
            title, description, subsections, keywords=keywords
        )

        # Build context string
        chunks_text = ""
        for idx, chunk in enumerate(chunks):
            page = chunk.get("page_number", "N/A")
            chunks_text += f"\n[Chunk {idx + 1} - Page {page}]:\n{chunk['text']}\n"

        # LLM summarization
        prompt = SECTION_EXTRACTION_PROMPT.format(
            section_title=title,
            section_description=description,
            subsections=", ".join(subsections),
            chunks=chunks_text,
        )
        response = self.llm.invoke(prompt)
        summary = response.content

        # Focused retrieval per subsection
        subsection_contents: List[SubsectionContent] = []
        for sub_title in subsections:
            sub_query = f"{title} - {sub_title}"
            if keywords:
                sub_query += f". Keywords: {', '.join(keywords[:5])}"
            sub_chunks = self.rag.retrieve_and_rerank(
                sub_query, initial_k=5, final_k=3
            )
            subsection_contents.append(
                SubsectionContent(
                    title=sub_title,
                    content=self._extract_subsection_content(
                        sub_title, sub_chunks
                    ),
                    chunks=sub_chunks,
                )
            )

        section_content = SectionContent(
            section_key=section_key,
            title=title,
            description=description,
            summary=summary,
            subsections=subsection_contents,
            all_chunks=chunks,
        )
        self.extracted_sections[section_key] = section_content
        print(f"Extracted {title} with {len(chunks)} chunks")
        return section_content

    # -----------------------------------------------------------------

    @staticmethod
    def _extract_subsection_content(subsection_title: str,
                                    chunks: List[Dict]) -> str:
        if not chunks:
            return "Not found in document"
        parts = [c["text"] for c in chunks[:2]]
        return " ".join(parts)[:500]

    # -----------------------------------------------------------------
    # All sections
    # -----------------------------------------------------------------

    def extract_all_sections(self) -> Dict[str, SectionContent]:
        """Extract every discovered section."""
        n = len(self.sections)
        print(f"Extracting all {n} discovered sections …")
        for section_key in self.sections:
            self.extract_section(section_key)
        print(f"Extracted {len(self.extracted_sections)} sections")
        return self.extracted_sections

    # -----------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------

    def get_section(self, section_key: str) -> Optional[SectionContent]:
        return self.extracted_sections.get(section_key)

    def get_chunks_for_section(self, section_key: str) -> List[Dict]:
        sec = self.extracted_sections.get(section_key)
        return sec.all_chunks if sec else []

    def get_chunks_for_subsection(self, section_key: str,
                                  subsection_index: int) -> List[Dict]:
        sec = self.extracted_sections.get(section_key)
        if sec and 0 <= subsection_index < len(sec.subsections):
            return sec.subsections[subsection_index].chunks
        return []

    def get_navigation_structure(self) -> List[Dict]:
        """Navigation tree for the UI (uses dynamic sections)."""
        nav = []
        for section_key, cfg in self.sections.items():
            section_data = {
                "key": section_key,
                "title": cfg["title"],
                "description": cfg.get("description", ""),
                "keywords": cfg.get("keywords", []),
                "subsections": [],
            }
            for idx, sub_title in enumerate(cfg.get("subsections", [])):
                sub_data = {
                    "index": idx,
                    "title": sub_title,
                    "has_content": False,
                    "chunks": [],
                }
                extracted = self.extracted_sections.get(section_key)
                if extracted and idx < len(extracted.subsections):
                    sub = extracted.subsections[idx]
                    sub_data["has_content"] = len(sub.chunks) > 0
                    sub_data["chunks"] = sub.chunks
                section_data["subsections"].append(sub_data)
            nav.append(section_data)
        return nav

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def export_to_json(self, filepath: str):
        data = {k: s.to_dict() for k, s in self.extracted_sections.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Exported sections to {filepath}")

    def load_from_json(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.extracted_sections = {}
        for key, sd in data.items():
            subs = [
                SubsectionContent(
                    title=s["title"], content=s["content"], chunks=s["chunks"]
                )
                for s in sd.get("subsections", [])
            ]
            self.extracted_sections[key] = SectionContent(
                section_key=sd["section_key"],
                title=sd["title"],
                description=sd["description"],
                summary=sd["summary"],
                subsections=subs,
                all_chunks=sd["all_chunks"],
            )
        print(f"Loaded {len(self.extracted_sections)} sections from {filepath}")


# =====================================================================
if __name__ == "__main__":
    import os
    from document_processor import DocumentProcessor
    from document_scanner import DocumentScanner

    pdf_path = "../Fund Facts - HDFC Income Fund - December 2025 [a].pdf"
    if os.path.exists(pdf_path):
        # 1. Discover sections dynamically
        scanner = DocumentScanner()
        is_valid, score, _ = scanner.validate_document(pdf_path)
        if not is_valid:
            print("Document rejected.")
        else:
            sections = scanner.discover_sections(pdf_path)

            # 2. Process + index
            processor = DocumentProcessor()
            chunks = processor.process_document(pdf_path)
            rag = RAGEngine()
            rag.index_chunks(chunks)

            # 3. Extract first section as demo
            extractor = SectionExtractor(rag, sections)
            first_key = next(iter(sections))
            sec = extractor.extract_section(first_key)
            print(f"\n--- {sec.title} ---")
            print(f"Summary: {sec.summary[:500]}…")
