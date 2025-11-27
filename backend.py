import os
import time
import uuid
import asyncio
import re
from typing import List, Dict, Any, Tuple
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import cohere
from dotenv import load_dotenv
import nest_asyncio
import PyPDF2
import io

# Load environment variables from .env file
load_dotenv()

# Apply nest_asyncio to handle event loop issues
nest_asyncio.apply()

class ResponseFormatter:
    """Class to format the response with proper citations and structure"""
    
    @staticmethod
    def format_answer(answer: str, cited_chunks: List[Dict]) -> str:
        """Format the answer with proper citations and structure"""
        if not answer:
            return "I couldn't generate an answer based on the available information."
        
        # Clean up the answer
        formatted_answer = answer.strip()
        
        # Ensure citations are properly formatted
        formatted_answer = re.sub(r'\[(\d+)\]', r'[\1]', formatted_answer)
        
        # Add header
        formatted_response = "## Answer\n\n"
        formatted_response += formatted_answer
        
        # Add sources section if there are citations
        if cited_chunks:
            formatted_response += "\n\n## Sources\n\n"
            for i, chunk in enumerate(cited_chunks):
                source = chunk["metadata"].get("source", "Unknown")
                formatted_response += f"**[{i+1}]** {source}\n\n"
        
        return formatted_response
    
    @staticmethod
    def format_sources(cited_chunks: List[Dict]) -> str:
        """Format the sources section in detail"""
        if not cited_chunks:
            return ""
        
        formatted_sources = "## Detailed Sources\n\n"
        for i, chunk in enumerate(cited_chunks):
            source = chunk["metadata"].get("source", "Unknown")
            chunk_index = chunk["metadata"].get("chunk_index", 0)
            total_chunks = chunk["metadata"].get("total_chunks", 1)
            score = chunk.get("score", 0)
            rerank_score = chunk.get("rerank_score", 0)
            
            formatted_sources += f"### Source [{i+1}]: {source}\n\n"
            formatted_sources += f"**Relevance Score**: {score:.3f}\n\n"
            if rerank_score:
                formatted_sources += f"**Rerank Score**: {rerank_score:.3f}\n\n"
            formatted_sources += f"**Content**:\n{chunk['text']}\n\n"
            formatted_sources += "---\n\n"
        
        return formatted_sources

class MiniRAGBackend:
    def __init__(self):
        # Initialize all clients with environment variables
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        
        # Check if API keys are loaded
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        if not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = "mini-rag-index"
        self.dimension = 768  # For Google embedding model
        self.create_index()
        
        # Initialize Google embedding model
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.google_api_key
        )
        
        # Initialize Google Generative AI via LangChain - Use correct model name
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Updated to correct model name
            google_api_key=self.google_api_key,
            temperature=0.1
        )
        
        # Initialize Cohere for reranking
        self.cohere_client = cohere.Client(self.cohere_api_key)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
        )
        
        # Initialize response formatter
        self.formatter = ResponseFormatter()
    
    def create_index(self):
        """Create Pinecone index if it doesn't exist"""
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
        
        self.index = self.pc.Index(self.index_name)
    
    def extract_text_from_pdf(self, file_bytes):
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def chunk_document(self, text: str, source: str = "pasted_text") -> List[Dict]:
        """Split document into chunks with metadata"""
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid4())
            documents.append({
                "id": doc_id,
                "text": chunk,
                "metadata": {
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
        
        return documents
    
    def embed_and_upsert(self, documents: List[Dict]):
        """Generate embeddings and upsert to Pinecone"""
        texts = [doc["text"] for doc in documents]
        
        # Create a new event loop for the embedding operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            embeddings = self.embedding_model.embed_documents(texts)
        finally:
            loop.close()
        
        vectors = []
        for doc, embedding in zip(documents, embeddings):
            vectors.append({
                "id": doc["id"],
                "values": embedding,
                "metadata": {
                    "text": doc["text"],
                    "source": doc["metadata"]["source"],
                    "chunk_index": doc["metadata"]["chunk_index"],
                    "total_chunks": doc["metadata"]["total_chunks"]
                }
            })
        
        # Upsert in batches of 100
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i+100]
            self.index.upsert(vectors=batch)
    
    def process_input(self, text: str, source: str = "pasted_text"):
        """Process input text and store in vector database"""
        documents = self.chunk_document(text, source)
        self.embed_and_upsert(documents)
        return len(documents)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve relevant chunks from vector database"""
        # Generate query embedding
        # Create a new event loop for the embedding operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            query_embedding = self.embedding_model.embed_query(query)
        finally:
            loop.close()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        retrieved_chunks = []
        for match in results.matches:
            retrieved_chunks.append({
                "text": match.metadata.get("text", ""),
                "score": match.score,
                "metadata": match.metadata
            })
        
        return retrieved_chunks
    
    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank retrieved chunks using Cohere"""
        if not chunks:
            return []
        
        documents = [chunk["text"] for chunk in chunks]
        
        try:
            results = self.cohere_client.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
                model="rerank-english-v2.0"
            )
            
            reranked_chunks = []
            for result in results:
                original_chunk = chunks[result.index]
                original_chunk["rerank_score"] = result.relevance_score
                reranked_chunks.append(original_chunk)
            
            return reranked_chunks
        except Exception as e:
            print(f"Reranking failed: {e}")
            return chunks[:top_k]  # Fallback to original ranking
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> Tuple[str, List[Dict]]:
        """Generate answer using Google Generative AI with citations"""
        if not context_chunks:
            return "I couldn't find enough relevant information to answer your question. Please try a different query or add more documents to the knowledge base.", []
        
        # Prepare context with citations
        context_with_citations = ""
        for i, chunk in enumerate(context_chunks):
            source = chunk["metadata"].get("source", "Unknown")
            context_with_citations += f"[{i+1}] {chunk['text']}\nSource: {source}\n\n"
        
        # Create prompt with structured output instructions
        prompt = f"""
        Based on the context below, provide a concise answer to the question.
        Use inline citations like [1], [2] for any information from the context.
        If the context doesn't contain relevant information, say so.
        
        Format your response as follows:
        - Start with a clear, direct answer to the question
        - Provide additional context or explanation if needed
        - Always cite your sources using [number] notation
        - Keep the response concise but informative
        
        Question: {query}
        
        Context:
        {context_with_citations}
        
        Answer:
        """
        
        try:
            # Use LangChain's ChatGoogleGenerativeAI with optimized parameters
            messages = [
                SystemMessage(content="You are a helpful assistant that provides accurate, concise answers based on the given context. Always cite your sources using inline citations like [1], [2], etc."),
                HumanMessage(content=prompt)
            ]
            
            # Create a new event loop for the LLM operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = self.llm.invoke(messages)
                answer = response.content
            finally:
                loop.close()
            
            # Extract citations from answer
            cited_chunks = []
            for i, chunk in enumerate(context_chunks):
                if f"[{i+1}]" in answer:
                    cited_chunks.append(chunk)
            
            return answer, cited_chunks
        except Exception as e:
            return f"Error generating answer: {str(e)}", []
    
    def query(self, query: str, retrieve_k: int = 10, rerank_k: int = 5) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve, rerank, generate"""
        start_time = time.time()
        
        # Retrieve
        retrieve_time = time.time()
        retrieved_chunks = self.retrieve(query, top_k=retrieve_k)
        retrieve_time = time.time() - retrieve_time
        
        # Rerank
        rerank_time = time.time()
        reranked_chunks = self.rerank(query, retrieved_chunks, top_k=rerank_k)
        rerank_time = time.time() - rerank_time
        
        # Generate answer
        generate_time = time.time()
        answer, cited_chunks = self.generate_answer(query, reranked_chunks)
        generate_time = time.time() - generate_time
        
        # Format the response
        formatted_answer = self.formatter.format_answer(answer, cited_chunks)
        detailed_sources = self.formatter.format_sources(cited_chunks)
        
        total_time = time.time() - start_time
        
        return {
            "answer": formatted_answer,
            "detailed_sources": detailed_sources,
            "cited_chunks": cited_chunks,
            "retrieved_chunks": retrieved_chunks,
            "reranked_chunks": reranked_chunks,
            "timing": {
                "total": total_time,
                "retrieve": retrieve_time,
                "rerank": rerank_time,
                "generate": generate_time
            }
        }
