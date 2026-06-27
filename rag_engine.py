# =============================================================================
# RAG Engine - ChromaDB + LangChain + Re-ranking
# =============================================================================

import json
import os
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma

from document_processor import ChunkWithMetadata
from config import (
    OPENAI_API_KEY, 
    CHROMA_PERSIST_DIR, 
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RERANK_TOP_K,
    FINAL_TOP_K,
    CHAT_SYSTEM_PROMPT,
    RERANK_PROMPT
)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class RAGEngine:
    """
    RAG Engine using ChromaDB for vector storage and LangChain for orchestration.
    Implements re-ranking for improved retrieval quality.
    """
    
    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        """
        Initialize the RAG engine.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        
        # Initialize OpenAI embeddings
        print("Initializing OpenAI Embeddings...")
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Initialize LLM for chat and re-ranking
        print("Initializing ChatOpenAI...")
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Initialize ChromaDB
        self.vector_store = None
        self.chunks_metadata: Dict[int, ChunkWithMetadata] = {}
        
        print("RAG Engine initialized!")
    
    def index_chunks(self, chunks: List[ChunkWithMetadata], collection_name: str = COLLECTION_NAME):
        """
        Index document chunks into ChromaDB.
        
        Args:
            chunks: List of ChunkWithMetadata objects
            collection_name: Name for the ChromaDB collection
        """
        print(f"Indexing {len(chunks)} chunks into ChromaDB...")
        
        # Store chunks metadata for later retrieval
        self.chunks_metadata = {chunk.chunk_id: chunk for chunk in chunks}
        
        # Create LangChain documents with metadata
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "element_type": chunk.element_type,
                    "headings": json.dumps(chunk.headings),  # JSON serialize list
                    "bbox": json.dumps(chunk.bbox) if chunk.bbox else None
                }
            )
            documents.append(doc)
        
        # Create or update vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        
        print(f"Successfully indexed {len(documents)} chunks!")
        return self.vector_store
    
    def load_existing_index(self, collection_name: str = COLLECTION_NAME):
        """Load an existing ChromaDB index"""
        print(f"Loading existing index: {collection_name}")
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        print("Index loaded!")
        return self.vector_store
    
    def retrieve(self, query: str, top_k: int = RERANK_TOP_K) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            raise ValueError("No vector store loaded. Call index_chunks() first.")
        
        # Similarity search
        results = self.vector_store.similarity_search(query, k=top_k)
        return results
    
    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Re-rank documents using LLM-based scoring.
        
        Args:
            query: Original query
            documents: List of documents to re-rank
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not documents:
            return []
        
        # Prepare chunks for re-ranking prompt
        chunks_text = ""
        for idx, doc in enumerate(documents):
            chunks_text += f"\n[Chunk {idx}]:\n{doc.page_content[:500]}...\n"
        
        # Create re-ranking prompt
        prompt = RERANK_PROMPT.format(query=query, chunks=chunks_text)
        
        try:
            # Get LLM scores
            response = self.llm.invoke(prompt)
            scores_text = response.content.strip()
            
            # Parse JSON response
            # Clean up response if needed
            if "```json" in scores_text:
                scores_text = scores_text.split("```json")[1].split("```")[0]
            elif "```" in scores_text:
                scores_text = scores_text.split("```")[1].split("```")[0]
            
            scores = json.loads(scores_text)
            
            # Create scored list
            scored_docs = []
            for score_item in scores:
                chunk_id = score_item.get('chunk_id', 0)
                score = score_item.get('score', 0)
                if 0 <= chunk_id < len(documents):
                    scored_docs.append((documents[chunk_id], score))
            
            # Sort by score descending
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return scored_docs
            
        except Exception as e:
            print(f"Re-ranking failed: {e}")
            # Fallback: return original order with default scores
            return [(doc, 5.0) for doc in documents]
    
    def retrieve_and_rerank(self, query: str, 
                           initial_k: int = RERANK_TOP_K,
                           final_k: int = FINAL_TOP_K) -> List[Dict]:
        """
        Retrieve and re-rank documents for a query.
        
        Args:
            query: Search query
            initial_k: Number of documents for initial retrieval
            final_k: Number of documents to return after re-ranking
            
        Returns:
            List of dicts with document info, scores, and metadata
        """
        # Initial retrieval
        documents = self.retrieve(query, top_k=initial_k)
        
        # Re-rank
        reranked = self.rerank(query, documents)
        
        # Take top results
        results = []
        for doc, score in reranked[:final_k]:
            # Parse metadata
            metadata = doc.metadata
            bbox = json.loads(metadata.get('bbox')) if metadata.get('bbox') else None
            headings = json.loads(metadata.get('headings', '[]'))
            
            results.append({
                'text': doc.page_content,
                'score': score,
                'chunk_id': metadata.get('chunk_id'),
                'page_number': metadata.get('page_number'),
                'element_type': metadata.get('element_type'),
                'headings': headings,
                'bbox': bbox
            })
        
        return results
    
    def chat(self, query: str, context_chunks: Optional[List[Dict]] = None) -> Tuple[str, List[Dict]]:
        """
        Chat with the document using RAG.
        
        Args:
            query: User's question
            context_chunks: Pre-retrieved chunks (if None, will retrieve)
            
        Returns:
            Tuple of (LLM response text, list of source chunks used)
        """
        # Retrieve context if not provided
        if context_chunks is None:
            context_chunks = self.retrieve_and_rerank(query)
        
        # Build context string
        context = ""
        for idx, chunk in enumerate(context_chunks):
            page = chunk.get('page_number', 'N/A')
            context += f"\n[Chunk {idx+1} - Page {page}]:\n{chunk['text']}\n"
        
        # Create chat prompt
        prompt = CHAT_SYSTEM_PROMPT.format(context=context, query=query)
        
        # Get response
        response = self.llm.invoke(prompt)
        return response.content, context_chunks
    
    def search_for_section(self, section_title: str, section_description: str,
                          subsections: List[str],
                          keywords: Optional[List[str]] = None) -> List[Dict]:
        """
        Search for content related to a specific MF section.
        
        Args:
            section_title: Title of the section (e.g., "Risk Factors")
            section_description: Description of what to find
            subsections: List of subsections to look for
            keywords: Optional list of keywords to boost search relevance
            
        Returns:
            Relevant chunks for this section
        """
        # Build comprehensive query
        query = f"{section_title}: {section_description}. "
        query += f"Looking for: {', '.join(subsections)}"
        if keywords:
            query += f". Key terms: {', '.join(keywords)}"
        
        # Retrieve more chunks for section extraction
        return self.retrieve_and_rerank(query, initial_k=15, final_k=10)
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[ChunkWithMetadata]:
        """Get a specific chunk by its ID"""
        return self.chunks_metadata.get(chunk_id)
    
    def get_all_chunks_for_page(self, page_number: int) -> List[ChunkWithMetadata]:
        """Get all chunks from a specific page"""
        return [
            chunk for chunk in self.chunks_metadata.values() 
            if chunk.page_number == page_number
        ]


# Test the RAG engine
if __name__ == "__main__":
    from document_processor import DocumentProcessor
    
    # Process a document
    processor = DocumentProcessor()
    pdf_path = "../Fund Facts - HDFC Income Fund - December 2025 [a].pdf"
    
    if os.path.exists(pdf_path):
        chunks = processor.process_document(pdf_path)
        
        # Initialize RAG engine
        rag = RAGEngine()
        
        # Index chunks
        rag.index_chunks(chunks)
        
        # Test retrieval
        print("\n--- Testing Retrieval ---")
        results = rag.retrieve_and_rerank("What are the risks of this fund?")
        for r in results[:3]:
            print(f"\nScore: {r['score']}, Page: {r['page_number']}")
            print(f"Text: {r['text'][:200]}...")
        
        # Test chat
        print("\n--- Testing Chat ---")
        response = rag.chat("What is the expense ratio of this fund?")
        print(f"Response: {response}")
