import streamlit as st
import os
import time
from datetime import datetime
from backend import MiniRAGBackend
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Mini RAG Application",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "rag_backend" not in st.session_state:
    st.session_state.rag_backend = None
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = 0
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Initialize backend
def init_backend():
    try:
        st.session_state.rag_backend = MiniRAGBackend()
        st.session_state.initialized = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize backend: {str(e)}")
        return False

# UI Components
def render_sidebar():
    with st.sidebar:
        st.title("🔍 Mini RAG")
        st.markdown("---")
        
        st.subheader("Configuration")
        st.info("""
        This RAG application uses:
        - Pinecone as vector database
        - Google Generative AI for embeddings and answers
        - Cohere for reranking
        """)
        
        # Show API key status
        st.markdown("---")
        st.subheader("API Status")
        pinecone_status = "✅" if os.getenv("PINECONE_API_KEY") else "❌"
        google_status = "✅" if os.getenv("GOOGLE_API_KEY") else "❌"
        cohere_status = "✅" if os.getenv("COHERE_API_KEY") else "❌"
        
        st.write(f"Pinecone: {pinecone_status}")
        st.write(f"Google AI: {google_status}")
        st.write(f"Cohere: {cohere_status}")
        
        st.markdown("---")
        st.subheader("Statistics")
        st.write(f"Processed documents: {st.session_state.processed_documents}")
        st.write(f"Query history: {len(st.session_state.query_history)}")
        
        if st.button("Clear History"):
            st.session_state.query_history = []
            st.rerun()

def render_file_upload():
    st.header("📁 Add Documents")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload a text or PDF file", 
            type=["txt", "pdf"],
            help="Supported formats: TXT, PDF"
        )
    
    with col2:
        pasted_text = st.text_area(
            "Or paste text directly",
            height=150,
            placeholder="Enter text here..."
        )
    
    process_btn = st.button("Process Document", type="primary")
    
    if process_btn and (uploaded_file or pasted_text):
        with st.spinner("Processing document..."):
            try:
                if uploaded_file:
                    if uploaded_file.type == "text/plain":
                        # Handle text files
                        text = str(uploaded_file.read(), "utf-8")
                        source = uploaded_file.name
                    elif uploaded_file.type == "application/pdf":
                        # Handle PDF files
                        if st.session_state.rag_backend is None:
                            if not init_backend():
                                return
                        text = st.session_state.rag_backend.extract_text_from_pdf(uploaded_file.read())
                        source = uploaded_file.name
                    else:
                        st.error("Unsupported file format")
                        return
                else:
                    text = pasted_text
                    source = "pasted_text"
                
                if st.session_state.rag_backend is None:
                    if not init_backend():
                        return
                
                chunks_count = st.session_state.rag_backend.process_input(text, source)
                st.session_state.processed_documents += 1
                
                st.success(f"Processed document into {chunks_count} chunks!")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

def render_query_section():
    st.header("❓ Ask Questions")
    
    query = st.text_input(
        "Enter your question:",
        placeholder="What would you like to know?",
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col2:
        retrieve_k = st.slider("Retrieve chunks", 5, 20, 10, key="retrieve_k")
    
    with col3:
        rerank_k = st.slider("Rerank chunks", 3, 10, 5, key="rerank_k")
    
    if st.button("Get Answer", type="primary") and query:
        if st.session_state.rag_backend is None:
            if not init_backend():
                return
        
        with st.spinner("Thinking..."):
            try:
                # Execute query with parameters
                start_time = time.time()
                result = st.session_state.rag_backend.query(
                    query, 
                    retrieve_k=retrieve_k, 
                    rerank_k=rerank_k
                )
                processing_time = time.time() - start_time
                
                # Store result
                result["query"] = query
                result["timestamp"] = datetime.now()
                result["processing_time"] = processing_time
                st.session_state.query_history.insert(0, result)
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

def render_answer_section():
    if not st.session_state.query_history:
        return
    
    result = st.session_state.query_history[0]
    
    st.header("📝 Answer")
    st.markdown("---")
    
    # Display formatted answer
    st.markdown(result["answer"], unsafe_allow_html=True)
    
    # Display timing information
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Time", f"{result['processing_time']:.2f}s")
    
    with col2:
        st.metric("Retrieve", f"{result['timing']['retrieve']:.2f}s")
    
    with col3:
        st.metric("Rerank", f"{result['timing']['rerank']:.2f}s")
    
    with col4:
        st.metric("Generate", f"{result['timing']['generate']:.2f}s")
    
    # Display detailed sources in an expander
    if result.get("detailed_sources"):
        with st.expander("📚 View Detailed Sources"):
            st.markdown(result["detailed_sources"], unsafe_allow_html=True)

def render_query_history():
    if len(st.session_state.query_history) <= 1:
        return
    
    st.header("📋 Query History")
    
    for i, past_query in enumerate(st.session_state.query_history[1:6]):
        with st.expander(f"Query {i+1}: {past_query['query'][:50]}..."):
            st.write(f"**Question:** {past_query['query']}")
            st.write(f"**Answer:** {past_query['answer'][:200]}...")
            st.write(f"**Time:** {past_query['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            if st.button("Load this query", key=f"load_{i}"):
                st.session_state.query_input = past_query["query"]
                st.rerun()

def main():
    # Custom CSS for better formatting
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton button {
        width: 100%;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 20px;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 3px solid #ff7f0e;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">Mini RAG Application</h1>', unsafe_allow_html=True)
    
    # Initialize backend if not already done
    if not st.session_state.initialized:
        with st.spinner("Initializing RAG backend..."):
            if not init_backend():
                st.error("Failed to initialize RAG backend. Please check your API keys.")
                return
    
    # Render components
    render_sidebar()
    render_file_upload()
    render_query_section()
    render_answer_section()
    render_query_history()

if __name__ == "__main__":
    main()