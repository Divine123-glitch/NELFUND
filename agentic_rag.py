"""
NELFUND Agentic RAG System
Intelligent agent that decides when to retrieve documents and maintains conversation memory
"""

import os
from typing import TypedDict, List, Annotated
from operator import add
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from vector_store import NELFUNDVectorStore


# Load environment variables
load_dotenv()


class AgentState(TypedDict):
    """
    State that flows through the agent graph
    """
    messages: Annotated[List, add]  # Conversation history
    question: str                    # Current user question
    context: str                     # Retrieved documents
    should_retrieve: bool            # Whether to retrieve documents
    answer: str                      # Final answer


class NELFUNDAgenticRAG:
    """
    Agentic RAG system with conditional retrieval and conversation memory
    """
    
    def __init__(self, vectorstore: NELFUNDVectorStore):
        """
        Initialize the agentic RAG system
        
        Args:
            vectorstore: Initialized vector store with loaded documents
        """
        self.vectorstore = vectorstore
        self.retriever = vectorstore.get_retriever(k=4)
        
        # Initialize LLM
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Fast and cost-effective
            temperature=0.3,       # Low temperature for factual answers
            openai_api_key=api_key
        )
        
        # Build the agent graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow
        
        Returns:
            StateGraph with agent logic
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("retriever", self._retrieve_documents)
        workflow.add_node("generator", self._generate_answer)
        
        # Define edges (flow)
        workflow.set_entry_point("classifier")
        
        # Conditional edge: retrieve only if needed
        workflow.add_conditional_edges(
            "classifier",
            self._should_retrieve_decision,
            {
                "retrieve": "retriever",
                "generate": "generator"
            }
        )
        
        workflow.add_edge("retriever", "generator")
        workflow.add_edge("generator", END)
        
        return workflow
    
    def _classify_query(self, state: AgentState) -> AgentState:
        """
        Classify if the query needs document retrieval
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with should_retrieve decision
        """
        question = state["question"]
        
        # System prompt for classification
        classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier for NELFUND student loan assistant.
            
Analyze the user's question and determine if it requires retrieving information from NELFUND policy documents.

RETRIEVE DOCUMENTS for:
- Questions about eligibility criteria
- Questions about application process
- Questions about repayment terms
- Questions about specific NELFUND policies
- Questions about required documents
- Questions about loan amounts or coverage
- Any factual questions about NELFUND

DO NOT RETRIEVE for:
- Greetings (hi, hello, hey)
- Thank you messages
- Casual conversation
- Follow-up acknowledgments ("okay", "thanks", "got it")
- Questions already answered in conversation history

Respond with ONLY one word: "RETRIEVE" or "GENERATE"
"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        # Get conversation history (last 5 messages for context)
        history = state.get("messages", [])[-5:]
        
        chain = classifier_prompt | self.llm | StrOutputParser()
        decision = chain.invoke({
            "question": question,
            "history": history
        }).strip().upper()
        
        should_retrieve = "RETRIEVE" in decision
        
        print(f"Classification: {'RETRIEVE' if should_retrieve else 'GENERATE'}")
        
        return {
            **state,
            "should_retrieve": should_retrieve
        }
    
    def _should_retrieve_decision(self, state: AgentState) -> str:
        """
        Decision function for conditional edge
        
        Args:
            state: Current agent state
            
        Returns:
            "retrieve" or "generate"
        """
        return "retrieve" if state["should_retrieve"] else "generate"
    
    def _retrieve_documents(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant documents from vector store
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with retrieved context
        """
        question = state["question"]
        
        print(f"Retrieving documents for: '{question}'")
        
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(question)
        
        # Format context with sources
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            context_parts.append(
                f"[Source {i}: {source}, Page {page}]\n{doc.page_content}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        print(f"Retrieved {len(docs)} relevant documents")
        
        return {
            **state,
            "context": context
        }
    
    def _generate_answer(self, state: AgentState) -> AgentState:
        """
        Generate final answer with or without retrieved context
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with generated answer
        """
        question = state["question"]
        context = state.get("context", "")
        history = state.get("messages", [])
        
        print(f"Generating answer...")
        
        # System prompt for answer generation
        if context:
            # Answer with retrieved documents
            system_message = """You are a helpful NELFUND student loan assistant for Nigerian students.

Your role:
- Answer questions about NELFUND student loans accurately
- Be friendly, encouraging, and supportive
- Use the provided context from official documents
- ALWAYS cite your sources using [Source X] format
- If information isn't in the context, say so honestly
- Keep answers clear and concise
- Use simple language that students can understand

Context from NELFUND documents:
{context}

IMPORTANT: Always cite which source you're using when providing information."""
        else:
            # Answer without retrieval (greetings, casual conversation)
            system_message = """You are a helpful NELFUND student loan assistant for Nigerian students.

This is a casual conversation (greeting, thanks, etc.) that doesn't require document retrieval.
Respond warmly and naturally. Keep it brief and friendly."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            "question": question,
            "context": context,
            "history": history[-5:]  # Last 5 messages for context
        })
        
        print(f"Answer generated")
        
        return {
            **state,
            "answer": answer
        }
    
    def chat(self, question: str, conversation_history: List = None) -> dict:
        """
        Main chat interface
        
        Args:
            question: User's question
            conversation_history: Previous messages in conversation
            
        Returns:
            Dictionary with answer and metadata
        """
        if conversation_history is None:
            conversation_history = []
        
        # Initial state
        initial_state = {
            "messages": conversation_history,
            "question": question,
            "context": "",
            "should_retrieve": False,
            "answer": ""
        }
        
        # Run the agent graph
        final_state = self.app.invoke(initial_state)
        
        # Update conversation history
        conversation_history.append(HumanMessage(content=question))
        conversation_history.append(AIMessage(content=final_state["answer"]))
        
        return {
            "answer": final_state["answer"],
            "retrieved": final_state["should_retrieve"],
            "sources": self._extract_sources(final_state.get("context", "")),
            "conversation_history": conversation_history
        }
    
    def _extract_sources(self, context: str) -> List[str]:
        """
        Extract source information from context
        
        Args:
            context: Retrieved context with source markers
            
        Returns:
            List of source descriptions
        """
        import re
        sources = re.findall(r'\[Source \d+: ([^\]]+)\]', context)
        return sources


def main():
    """
    Interactive demo of the agentic RAG system
    """
    print("NELFUND Agentic RAG System")
    print("="*80)
    
    # Load vector store
    print("\nLoading vector store...")
    try:
        vectorstore = NELFUNDVectorStore()
        vectorstore.load_vectorstore()
    except FileNotFoundError:
        print("\nVector store not found!")
        print("Run vector_store.py first to create the database")
        return
    
    # Initialize agent
    print("Initializing agent...")
    agent = NELFUNDAgenticRAG(vectorstore)
    
    print("\nAgent ready! Ask questions about NELFUND student loans.")
    print("Type 'quit' to exit\n")
    
    conversation_history = []
    
    while True:
        # Get user input
        question = input("\nYou: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not question:
            continue
        
        print("\nAssistant: ", end="", flush=True)
        
        # Get response
        response = agent.chat(question, conversation_history)
        
        # Print answer
        print(response["answer"])
        
        # Show metadata
        if response["retrieved"]:
            print(f"\nRetrieved from documents: {len(response['sources'])} sources")
            for i, source in enumerate(response['sources'], 1):
                print(f"   {i}. {source}")
        else:
            print("\nAnswered without document retrieval")


if __name__ == "__main__":
    main()