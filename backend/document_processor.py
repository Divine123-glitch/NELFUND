"""
NELFUND Document Loader and Chunking System
Loads PDFs, processes them, and creates optimized chunks for RAG
"""

import os
from typing import List, Dict
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class NELFUNDDocumentProcessor:
    """
    Handles loading and chunking of NELFUND policy documents
    """
    
    def __init__(self, data_directory: str = "./data"):
        """
        Initialize the document processor
        
        Args:
            data_directory: Path to folder containing NELFUND PDFs
        """
        self.data_directory = data_directory
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        
    def load_documents(self) -> List[Document]:
        """
        Load all PDF documents from the data directory
        
        Returns:
            List of Document objects
        """
        print(f"Loading documents from {self.data_directory}...")
        
        # Check if directory exists
        if not os.path.exists(self.data_directory):
            raise FileNotFoundError(
                f"Data directory '{self.data_directory}' not found. "
                f"Please create it and add your NELFUND PDFs."
            )
        
        # Load all PDFs from directory
        loader = DirectoryLoader(
            self.data_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        self.documents = loader.load()
        
        print(f"Loaded {len(self.documents)} document pages")
        return self.documents
    
    def chunk_documents(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks (maintains context)
            
        Returns:
            List of chunked Document objects
        """
        print(f"Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
        
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        # RecursiveCharacterTextSplitter is best for maintaining semantic meaning
        # It tries to split on paragraphs, then sentences, then words
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Split on paragraph breaks first
                "\n",    # Then single line breaks
                ". ",    # Then sentences
                " ",     # Then words
                ""       # Then characters (last resort)
            ]
        )
        
        self.chunks = text_splitter.split_documents(self.documents)
        
        # Add chunk metadata for better tracking
        for i, chunk in enumerate(self.chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        print(f"Created {len(self.chunks)} chunks")
        return self.chunks
    
    def get_document_stats(self) -> Dict:
        """
        Get statistics about loaded documents and chunks
        
        Returns:
            Dictionary with document statistics
        """
        if not self.documents:
            return {"error": "No documents loaded"}
        
        total_chars = sum(len(doc.page_content) for doc in self.documents)
        
        stats = {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks) if self.chunks else 0,
            "total_characters": total_chars,
            "avg_doc_length": total_chars // len(self.documents) if self.documents else 0,
            "sources": list(set(doc.metadata.get("source", "unknown") for doc in self.documents))
        }
        
        return stats
    
    def preview_chunks(self, n: int = 3):
        """
        Preview first n chunks to verify chunking quality
        
        Args:
            n: Number of chunks to preview
        """
        if not self.chunks:
            print("No chunks available. Run chunk_documents() first.")
            return
        
        print(f"\nPreviewing first {n} chunks:\n")
        
        for i, chunk in enumerate(self.chunks[:n]):
            print(f"--- CHUNK {i+1} ---")
            print(f"Source: {chunk.metadata.get('source', 'unknown')}")
            print(f"Page: {chunk.metadata.get('page', 'unknown')}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"\nContent Preview:\n{chunk.page_content[:300]}...")
            print("\n" + "="*80 + "\n")


def main():
    """
    Example usage of the document processor
    """
    # Initialize processor
    processor = NELFUNDDocumentProcessor(data_directory="./data")
    
    # Load documents
    try:
        documents = processor.load_documents()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo fix this:")
        print("   1. Create a 'data' folder in your project directory")
        print("   2. Download NELFUND PDFs and place them in the 'data' folder")
        print("   3. Run this script again\n")
        return
    
    # Chunk documents
    chunks = processor.chunk_documents(
        chunk_size=1000,    # Adjust based on your needs
        chunk_overlap=200   # Maintains context between chunks
    )
    
    # Get statistics
    stats = processor.get_document_stats()
    print("\nDocument Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Preview chunks
    processor.preview_chunks(n=2)
    
    print("\nDocument processing complete!")
    print(f"   Ready to create vector database with {len(chunks)} chunks")


if __name__ == "__main__":
    main()