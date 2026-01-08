import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { 
  Send, 
  MessageSquare, 
  Trash2, 
  Loader2,
  BookOpen,
  CheckCircle
} from 'lucide-react'
import './App.css'

const API_BASE_URL = 'http://localhost:8000'

function App() {
  const [conversations, setConversations] = useState([])
  const [currentConversationId, setCurrentConversationId] = useState(null)
  const [messages, setMessages] = useState([])
  const [inputMessage, setInputMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const messagesEndRef = useRef(null)

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Load conversations on mount
  useEffect(() => {
    loadConversations()
  }, [])

  // Load all conversations
  const loadConversations = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/conversations`)
      setConversations(response.data)
    } catch (error) {
      console.error('Error loading conversations:', error)
    }
  }

  // Load specific conversation messages
  const loadConversation = async (conversationId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/conversations/${conversationId}`)
      const formattedMessages = response.data.map(msg => ({
        role: msg.role,
        content: msg.content,
        sources: msg.sources ? msg.sources.split(', ').filter(s => s) : [],
        retrieved: msg.retrieved,
        timestamp: msg.timestamp
      }))
      setMessages(formattedMessages)
      setCurrentConversationId(conversationId)
    } catch (error) {
      console.error('Error loading conversation:', error)
    }
  }

  // Start new conversation
  const startNewConversation = () => {
    setCurrentConversationId(null)
    setMessages([])
    setInputMessage('')
  }

  // Send message
  const sendMessage = async (e) => {
    e.preventDefault()
    
    if (!inputMessage.trim() || loading) return

    const userMessage = inputMessage.trim()
    setInputMessage('')
    setLoading(true)

    // Add user message to UI immediately
    const newUserMessage = {
      role: 'user',
      content: userMessage,
      timestamp: new Date().toISOString()
    }
    setMessages(prev => [...prev, newUserMessage])

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        message: userMessage,
        conversation_id: currentConversationId
      })

      // Add assistant message
      const assistantMessage = {
        role: 'assistant',
        content: response.data.answer,
        sources: response.data.sources || [],
        retrieved: response.data.retrieved,
        timestamp: response.data.timestamp
      }
      setMessages(prev => [...prev, assistantMessage])

      // Update conversation ID if new
      if (!currentConversationId) {
        setCurrentConversationId(response.data.conversation_id)
        loadConversations() // Refresh sidebar
      }

    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please make sure the backend is running on http://localhost:8000',
        timestamp: new Date().toISOString()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  // Delete conversation
  const deleteConversation = async (conversationId, e) => {
    e.stopPropagation()
    
    if (!window.confirm('Delete this conversation?')) return

    try {
      await axios.delete(`${API_BASE_URL}/conversations/${conversationId}`)
      
      if (conversationId === currentConversationId) {
        startNewConversation()
      }
      
      loadConversations()
    } catch (error) {
      console.error('Error deleting conversation:', error)
    }
  }

  return (
    <div className="flex h-screen bg-gradient-to-br from-purple-600 to-blue-600">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-80' : 'w-0'} bg-white shadow-xl transition-all duration-300 overflow-hidden`}>
        <div className="p-4 border-b bg-gradient-to-r from-green-500 to-green-600">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <BookOpen size={24} />
            NELFUND Navigator
          </h2>
          <p className="text-sm text-green-50 mt-1">Your Student Loan Assistant</p>
        </div>

        <div className="p-4">
          <button
            onClick={startNewConversation}
            className="w-full bg-green-500 hover:bg-green-600 text-white font-semibold py-3 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors"
          >
            <MessageSquare size={20} />
            New Chat
          </button>
        </div>

        <div className="overflow-y-auto h-[calc(100vh-200px)] px-4">
          <h3 className="text-sm font-semibold text-gray-500 mb-2">Chat History</h3>
          {conversations.length === 0 ? (
            <p className="text-sm text-gray-400 text-center py-8">No conversations yet</p>
          ) : (
            <div className="space-y-2">
              {conversations.map((conv) => (
                <div
                  key={conv.conversation_id}
                  onClick={() => loadConversation(conv.conversation_id)}
                  className={`p-3 rounded-lg cursor-pointer group hover:bg-gray-50 transition-colors ${
                    currentConversationId === conv.conversation_id ? 'bg-green-50 border-2 border-green-500' : 'bg-gray-50'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {conv.title || 'Untitled Chat'}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        {conv.message_count} messages
                      </p>
                    </div>
                    <button
                      onClick={(e) => deleteConversation(conv.conversation_id, e)}
                      className="opacity-0 group-hover:opacity-100 text-red-500 hover:text-red-700 transition-opacity"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-white">
        {/* Header */}
        <div className="bg-white border-b px-6 py-4 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-800">NELFUND Student Loan Navigator</h1>
              <p className="text-sm text-gray-600 mt-1">Ask me anything about Nigerian student loans</p>
            </div>
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden bg-gray-100 p-2 rounded-lg hover:bg-gray-200"
            >
              <MessageSquare size={20} />
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <BookOpen size={64} className="text-green-500 mb-4" />
              <h2 className="text-2xl font-bold text-gray-800 mb-2">Welcome to NELFUND Navigator!</h2>
              <p className="text-gray-600 max-w-md">
                I'm here to help you understand NELFUND student loans. Ask me about eligibility, 
                application process, repayment terms, or any other questions!
              </p>
              <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl">
                <button
                  onClick={() => setInputMessage("Am I eligible for NELFUND?")}
                  className="p-4 bg-green-50 hover:bg-green-100 rounded-lg text-left transition-colors"
                >
                  <p className="font-semibold text-green-800">Am I eligible?</p>
                  <p className="text-sm text-green-600 mt-1">Check eligibility criteria</p>
                </button>
                <button
                  onClick={() => setInputMessage("How do I apply for student loan?")}
                  className="p-4 bg-blue-50 hover:bg-blue-100 rounded-lg text-left transition-colors"
                >
                  <p className="font-semibold text-blue-800">How to apply?</p>
                  <p className="text-sm text-blue-600 mt-1">Learn application process</p>
                </button>
                <button
                  onClick={() => setInputMessage("What documents do I need?")}
                  className="p-4 bg-purple-50 hover:bg-purple-100 rounded-lg text-left transition-colors"
                >
                  <p className="font-semibold text-purple-800">Required documents?</p>
                  <p className="text-sm text-purple-600 mt-1">Check document requirements</p>
                </button>
                <button
                  onClick={() => setInputMessage("When do I start paying back?")}
                  className="p-4 bg-orange-50 hover:bg-orange-100 rounded-lg text-left transition-colors"
                >
                  <p className="font-semibold text-orange-800">Repayment terms?</p>
                  <p className="text-sm text-orange-600 mt-1">Understand repayment</p>
                </button>
              </div>
            </div>
          ) : (
            <>
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-3xl rounded-2xl px-6 py-4 ${
                      message.role === 'user'
                        ? 'bg-green-500 text-white'
                        : 'bg-gray-100 text-gray-800'
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{message.content}</p>
                    
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-300">
                        <div className="flex items-center gap-2 text-sm text-gray-600 mb-2">
                          <CheckCircle size={16} />
                          <span className="font-semibold">Sources:</span>
                        </div>
                        <ul className="text-sm text-gray-600 space-y-1 ml-6">
                          {message.sources.map((source, idx) => (
                            <li key={idx} className="list-disc">{source}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              
              {loading && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 rounded-2xl px-6 py-4">
                    <Loader2 className="animate-spin text-green-500" size={24} />
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t bg-white p-4">
          <form onSubmit={sendMessage} className="max-w-4xl mx-auto">
            <div className="flex gap-3">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Ask about NELFUND student loans..."
                disabled={loading}
                className="flex-1 px-6 py-4 border-2 border-gray-300 rounded-full focus:outline-none focus:border-green-500 disabled:bg-gray-100 disabled:cursor-not-allowed text-lg"
              />
              <button
                type="submit"
                disabled={loading || !inputMessage.trim()}
                className="bg-green-500 hover:bg-green-600 text-white px-8 py-4 rounded-full font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
              >
                {loading ? (
                  <Loader2 className="animate-spin" size={20} />
                ) : (
                  <Send size={20} />
                )}
                Send
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}

export default App