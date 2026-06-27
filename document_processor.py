# =============================================================================
# Document Processor - Uses Docling for structured PDF extraction
# =============================================================================

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling.datamodel.document import TableItem


@dataclass
class ChunkWithMetadata:
    """Represents a document chunk with its metadata and location info"""
    chunk_id: int
    text: str
    page_number: int
    bbox: Optional[Dict[str, float]]  # Bounding box coordinates
    element_type: str  # 'text', 'table', 'heading', etc.
    headings: List[str]  # Section headings this chunk belongs to
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page_number": self.page_number,
            "bbox": self.bbox,
            "element_type": self.element_type,
            "headings": self.headings
        }


class DocumentProcessor:
    """
    Processes PDF documents using Docling to extract structured content
    with proper reading order, element types, and bounding boxes.
    """
    
    def __init__(self, tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_tokens: int = 512):
        """
        Initialize the document processor.
        
        Args:
            tokenizer_model: HuggingFace tokenizer for chunking
            max_tokens: Maximum tokens per chunk
        """
        self.tokenizer_model = tokenizer_model
        self.max_tokens = max_tokens
        self.converter = None
        self.chunker = None
        self.document = None
        self.chunks: List[ChunkWithMetadata] = []
        
    def _initialize_converter(self):
        """Initialize Docling converter (lazy loading)"""
        if self.converter is None:
            print("Initializing Docling DocumentConverter...")
            self.converter = DocumentConverter()
            print("DocumentConverter ready!")
            
    def _initialize_chunker(self):
        """Initialize the HybridChunker (lazy loading)"""
        if self.chunker is None:
            print("Initializing HybridChunker...")
            self.chunker = HybridChunker(
                tokenizer=self.tokenizer_model,
                max_tokens=self.max_tokens,
                merge_peers=True
            )
            print("HybridChunker ready!")
    
    def process_document(self, pdf_path: str) -> List[ChunkWithMetadata]:
        """
        Process a PDF document and extract structured chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of ChunkWithMetadata objects
        """
        # Initialize components
        self._initialize_converter()
        self._initialize_chunker()
        
        # Convert the PDF
        print(f"Converting document: {pdf_path}")
        result = self.converter.convert(pdf_path)
        self.document = result.document
        print(f"Document converted: {self.document.num_pages} pages")
        
        # Generate chunks
        print("Generating semantic chunks...")
        raw_chunks = list(self.chunker.chunk(self.document))
        print(f"Generated {len(raw_chunks)} chunks")
        
        # Process chunks to extract metadata
        self.chunks = []
        for idx, chunk in enumerate(raw_chunks):
            chunk_with_meta = self._extract_chunk_metadata(idx, chunk)
            self.chunks.append(chunk_with_meta)
        
        print(f"Processed {len(self.chunks)} chunks with metadata")
        return self.chunks
    
    def _extract_chunk_metadata(self, idx: int, chunk) -> ChunkWithMetadata:
        """
        Extract metadata from a Docling chunk.
        
        Args:
            idx: Chunk index
            chunk: Raw Docling chunk object
            
        Returns:
            ChunkWithMetadata object
        """
        # Get text content
        text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        
        # Default values
        page_number = 1
        bbox = None
        element_type = "text"
        headings = []
        
        # Extract metadata if available
        if hasattr(chunk, 'meta') and chunk.meta:
            meta = chunk.meta
            
            # Extract headings
            if hasattr(meta, 'headings') and meta.headings:
                headings = meta.headings
            
            # Extract from doc_items
            if hasattr(meta, 'doc_items') and meta.doc_items:
                for item in meta.doc_items:
                    # Get element type
                    if hasattr(item, 'label'):
                        label = item.label
                        element_type = str(label.value) if hasattr(label, 'value') else str(label)
                    
                    # Get page and bbox from provenance
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, 'page_no'):
                                page_number = prov.page_no
                            if hasattr(prov, 'bbox'):
                                b = prov.bbox
                                bbox = {
                                    'left': float(b.l),
                                    'top': float(b.t),
                                    'right': float(b.r),
                                    'bottom': float(b.b)
                                }
                            break  # Use first provenance
                    break  # Use first doc_item
        
        return ChunkWithMetadata(
            chunk_id=idx,
            text=text,
            page_number=page_number,
            bbox=bbox,
            element_type=element_type,
            headings=headings
        )
    
    def get_chunks_by_page(self, page_number: int) -> List[ChunkWithMetadata]:
        """Get all chunks from a specific page"""
        return [c for c in self.chunks if c.page_number == page_number]
    
    def get_chunks_by_type(self, element_type: str) -> List[ChunkWithMetadata]:
        """Get all chunks of a specific type (table, text, heading, etc.)"""
        return [c for c in self.chunks if c.element_type == element_type]
    
    def get_tables(self) -> List[Dict]:
        """Extract all tables from the document as DataFrames"""
        tables = []
        if self.document is None:
            return tables
            
        for idx, (item, level) in enumerate(self.document.iterate_items()):
            if isinstance(item, TableItem):
                try:
                    df = item.export_to_dataframe()
                    
                    # Get table location
                    page_no = 1
                    bbox = None
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, 'page_no'):
                                page_no = prov.page_no
                            if hasattr(prov, 'bbox'):
                                b = prov.bbox
                                bbox = {
                                    'left': float(b.l),
                                    'top': float(b.t),
                                    'right': float(b.r),
                                    'bottom': float(b.b)
                                }
                            break
                    
                    tables.append({
                        'dataframe': df,
                        'page_number': page_no,
                        'bbox': bbox
                    })
                except Exception as e:
                    print(f"Could not extract table: {e}")
                    
        return tables
    
    def export_to_markdown(self) -> str:
        """Export the entire document to markdown format"""
        if self.document is None:
            return ""
        return self.document.export_to_markdown()
    
    def save_chunks_to_json(self, filepath: str):
        """Save all chunks to a JSON file"""
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(chunks_data)} chunks to {filepath}")
    
    def load_chunks_from_json(self, filepath: str) -> List[ChunkWithMetadata]:
        """Load chunks from a JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        self.chunks = []
        for data in chunks_data:
            chunk = ChunkWithMetadata(
                chunk_id=data['chunk_id'],
                text=data['text'],
                page_number=data['page_number'],
                bbox=data['bbox'],
                element_type=data['element_type'],
                headings=data['headings']
            )
            self.chunks.append(chunk)
        
        print(f"Loaded {len(self.chunks)} chunks from {filepath}")
        return self.chunks


# Test the processor
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Test with sample PDF
    pdf_path = "../Fund Facts - HDFC Income Fund - December 2025 [a].pdf"
    if os.path.exists(pdf_path):
        chunks = processor.process_document(pdf_path)
        
        print("\n--- Sample Chunks ---")
        for chunk in chunks[:5]:
            print(f"\nChunk {chunk.chunk_id}:")
            print(f"  Page: {chunk.page_number}")
            print(f"  Type: {chunk.element_type}")
            print(f"  Headings: {chunk.headings}")
            print(f"  Text: {chunk.text[:100]}...")
            
        # Save to JSON
        processor.save_chunks_to_json("test_chunks.json")
