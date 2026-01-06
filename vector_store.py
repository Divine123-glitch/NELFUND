"""
NELFUND Vector Database System
Creates and manages ChromaDB vector store for semantic search
"""

import os
from typing import List, Optional
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv


class NELFUNDVectorStore:
    """
    Manages vector database for NELFUND document retrieval
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "nelfund_docs"
    ):
        """
        Initialize vector store
        
        Args:
            persist_directory: Where to save the vector database
            collection_name: Name for this collection of documents
        """
        # Load environment variables (API keys)
        load_dotenv()
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize OpenAI embeddings
        # Embeddings convert text to vectors (numbers) for semantic search
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Create a .env file with: OPENAI_API_KEY=your-key-here"
            )
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Fast and cost-effective
            openai_api_key=api_key
        )
        
        self.vectorstore: Optional[Chroma] = None
        
    def create_vectorstore(
        self,
        documents: List[Document],
        force_recreate: bool = False
    ) -> Chroma:
        """
        Create vector store from documents
        
        Args:
            documents: List of document chunks to embed
            force_recreate: If True, delete existing DB and create new one
            
        Returns:
            Chroma vectorstore instance
        """
        print(f"Creating vector store with {len(documents)} chunks...")
        
        # Check if vectorstore already exists
        if os.path.exists(self.persist_directory) and not force_recreate:
            print(f"Vector store already exists at {self.persist_directory}")
            response = input("Load existing store? (y/n): ").lower()
            if response == 'y':
                return self.load_vectorstore()
        
        # Delete existing store if force_recreate
        if force_recreate and os.path.exists(self.persist_directory):
            import shutil
            print(f"Deleting existing vector store...")
            shutil.rmtree(self.persist_directory)
        
        # Create new vector store
        print("Embedding documents (this may take a minute)...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        print(f"Vector store created and saved to {self.persist_directory}")
        return self.vectorstore
    
    def load_vectorstore(self) -> Chroma:
        """
        Load existing vector store from disk
        
        Returns:
            Chroma vectorstore instance
        """
        print(f"Loading vector store from {self.persist_directory}...")
        
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(
                f"Vector store not found at {self.persist_directory}. "
                "Create one first using create_vectorstore()"
            )
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        
        print("Vector store loaded successfully")
        return self.vectorstore
    
    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Search for relevant documents using semantic similarity
        
        Args:
            query: User's question
            k: Number of documents to retrieve
            
        Returns:
            List of relevant document chunks
        """
        if not self.vectorstore:
            raise ValueError("Vector store not loaded. Call load_vectorstore() first.")
        
        print(f"Searching for: '{query}'")
        results = self.vectorstore.similarity_search(query, k=k)
        
        print(f"Found {len(results)} relevant chunks")
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[tuple]:
        """
        Search with relevance scores
        
        Args:
            query: User's question
            k: Number of documents to retrieve
            
        Returns:
            List of (Document, score) tuples
        """
        if not self.vectorstore:
            raise ValueError("Vector store not loaded. Call load_vectorstore() first.")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def get_retriever(self, k: int = 4):
        """
        Get a retriever object for use in RAG chains
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever object
        """
        if not self.vectorstore:
            raise ValueError("Vector store not loaded. Call load_vectorstore() first.")
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def test_search(self, query: str):
        """
        Test search and display results
        
        Args:
            query: Test question
        """
        print(f"\n{'='*80}")
        print(f"TEST SEARCH: {query}")
        print(f"{'='*80}\n")
        
        results = self.similarity_search_with_score(query, k=3)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"--- Result {i} (Score: {score:.4f}) ---")
            print(f"Source: {doc.metadata.get('source', 'unknown')}")
            print(f"Page: {doc.metadata.get('page', 'unknown')}")
            print(f"\nContent:\n{doc.page_content[:400]}...")
            print("\n" + "="*80 + "\n")


def main():
    """
    Example usage: Create and test vector store
    """
    from document_processor import NELFUNDDocumentProcessor
    
    print("NELFUND Vector Store Setup\n")
    
    # Step 1: Load and chunk documents
    print("Step 1: Loading documents...")
    processor = NELFUNDDocumentProcessor(data_directory="./data")
    
    try:
        documents = processor.load_documents()
        chunks = processor.chunk_documents(chunk_size=1000, chunk_overlap=200)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have PDFs in the ./data folder")
        return
    
    # Step 2: Create vector store
    print("\nStep 2: Creating vector store...")
    vector_store = NELFUNDVectorStore(
        persist_directory="./chroma_db",
        collection_name="nelfund_docs"
    )
    
    try:
        vector_store.create_vectorstore(chunks, force_recreate=False)
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nCreate a .env file with your OpenAI API key:")
        print("   OPENAI_API_KEY=sk-your-key-here")
        return
    
    # Step 3: Test search
    print("\nStep 3: Testing search functionality...")
    
    test_queries = [
        "Am I eligible for NELFUND student loan?",
        "How do I apply for student loan?",
        "What documents do I need?",
        "When do I start paying back the loan?"
    ]
    
    for query in test_queries:
        vector_store.test_search(query)
        input("Press Enter for next query...")
    
    print("\nVector store setup complete!")
    print(f"   Database saved at: ./chroma_db")
    print(f"   Ready to build the Agentic RAG system!")


if __name__ == "__main__":
    main()