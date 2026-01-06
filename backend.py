"""
NELFUND FastAPI Backend
REST API with persistent chat storage using SQLite
"""

import os
import uuid
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
from dotenv import load_dotenv

from vector_store import NELFUNDVectorStore
from agentic_rag import NELFUNDAgenticRAG
from langchain_core.messages import HumanMessage, AIMessage


# Load environment variables
load_dotenv()


# ==================== DATABASE SETUP ====================

class ChatDatabase:
    """
    Manages SQLite database for persistent chat storage
    """
    
    def __init__(self, db_path: str = "./chats.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                title TEXT
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                sources TEXT,
                retrieved BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_conversation(self, conversation_id: str = None) -> str:
        """Create a new conversation"""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversations (conversation_id, title)
            VALUES (?, ?)
        """, (conversation_id, "New Conversation"))
        
        conn.commit()
        conn.close()
        
        return conversation_id
    
    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: str = "",
        retrieved: bool = False
    ):
        """Save a message to the database"""
        message_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO messages 
            (message_id, conversation_id, role, content, sources, retrieved)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (message_id, conversation_id, role, content, sources, retrieved))
        
        # Update conversation updated_at
        cursor.execute("""
            UPDATE conversations 
            SET updated_at = CURRENT_TIMESTAMP 
            WHERE conversation_id = ?
        """, (conversation_id,))
        
        conn.commit()
        conn.close()
    
    def get_conversation_messages(self, conversation_id: str) -> List[dict]:
        """Retrieve all messages from a conversation"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM messages 
            WHERE conversation_id = ? 
            ORDER BY timestamp ASC
        """, (conversation_id,))
        
        messages = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return messages
    
    def get_all_conversations(self) -> List[dict]:
        """Get list of all conversations"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.*, COUNT(m.message_id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.conversation_id = m.conversation_id
            GROUP BY c.conversation_id
            ORDER BY c.updated_at DESC
        """)
        
        conversations = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return conversations
    
    def delete_conversation(self, conversation_id: str):
        """Delete a conversation and its messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
        
        conn.commit()
        conn.close()
    
    def update_conversation_title(self, conversation_id: str, title: str):
        """Update conversation title"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE conversations 
            SET title = ? 
            WHERE conversation_id = ?
        """, (title, conversation_id))
        
        conn.commit()
        conn.close()


# ==================== PYDANTIC MODELS ====================

class ChatMessage(BaseModel):
    """Request model for chat endpoint"""
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str
    conversation_id: str
    retrieved: bool
    sources: List[str]
    timestamp: str


class ConversationHistory(BaseModel):
    """Model for conversation history"""
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class MessageHistory(BaseModel):
    """Model for individual message"""
    message_id: str
    role: str
    content: str
    sources: Optional[str]
    retrieved: bool
    timestamp: str


# ==================== FASTAPI APP ====================

# Global variables (initialized at startup)
agent = None
db = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events
    """
    global agent, db
    
    print("Starting NELFUND API...")
    
    # Initialize database
    print("Initializing database...")
    db = ChatDatabase()
    
    # Load vector store
    print("Loading vector store...")
    try:
        vectorstore = NELFUNDVectorStore()
        vectorstore.load_vectorstore()
    except FileNotFoundError:
        print("Vector store not found! Run vector_store.py first.")
        raise
    
    # Initialize agent
    print("Initializing RAG agent...")
    agent = NELFUNDAgenticRAG(vectorstore)
    
    print("API ready!")
    
    yield
    
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="NELFUND Student Loan Navigator API",
    description="AI-powered assistant for Nigerian student loans",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware (allows frontend to communicate)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "NELFUND API is running",
        "version": "1.0.0"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatMessage):
    """
    Main chat endpoint
    
    Accepts a message and optional conversation_id
    Returns AI response with sources
    """
    try:
        # Create or retrieve conversation
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = db.create_conversation()
        
        # Get conversation history
        messages = db.get_conversation_messages(conversation_id)
        
        # Convert to LangChain format
        conversation_history = []
        for msg in messages:
            if msg["role"] == "user":
                conversation_history.append(HumanMessage(content=msg["content"]))
            else:
                conversation_history.append(AIMessage(content=msg["content"]))
        
        # Get AI response
        response = agent.chat(request.message, conversation_history)
        
        # Save user message
        db.save_message(
            conversation_id=conversation_id,
            role="user",
            content=request.message,
            sources="",
            retrieved=False
        )
        
        # Save assistant message
        db.save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response["answer"],
            sources=", ".join(response["sources"]),
            retrieved=response["retrieved"]
        )
        
        # Update conversation title with first user message
        if len(messages) == 0:
            title = request.message[:50] + ("..." if len(request.message) > 50 else "")
            db.update_conversation_title(conversation_id, title)
        
        return ChatResponse(
            answer=response["answer"],
            conversation_id=conversation_id,
            retrieved=response["retrieved"],
            sources=response["sources"],
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations", response_model=List[ConversationHistory])
async def get_conversations():
    """
    Get list of all conversations
    """
    try:
        conversations = db.get_all_conversations()
        return conversations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}", response_model=List[MessageHistory])
async def get_conversation(conversation_id: str):
    """
    Get full message history for a conversation
    """
    try:
        messages = db.get_conversation_messages(conversation_id)
        if not messages:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation
    """
    try:
        db.delete_conversation(conversation_id)
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversations")
async def create_conversation():
    """
    Create a new conversation
    """
    try:
        conversation_id = db.create_conversation()
        return {"conversation_id": conversation_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting NELFUND API Server...")
    print("API will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes
    )