import os
import time
import uuid
import asyncio
import re
from typing import List, Dict, Any, Tuple
import io

# Third-party libraries (imports with fallbacks)
try:
    # Pinecone modern client (preferred)
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
except Exception:
    # If pinecone import fails, raise a clear error later during init
    pinecone = None
    Pinecone = None
    ServerlessSpec = None

# Text splitters: prefer dedicated package
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # Try old langchain path as fallback
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None

# Google GenAI + messages imports (try new split packages first)
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
except Exception:
    GoogleGenerativeAIEmbeddings = None
    ChatGoogleGenerativeAI = None

# Message types: new package name then old path
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:
    try:
        from langchain.schema import HumanMessage, SystemMessage
    except Exception:
        HumanMessage = None
        SystemMessage = None

# Output parsers: try preferred path; if not available provide small fallback
try:
    from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
except Exception:
    try:
        from langchain.output_parsers import StructuredOutputParser, ResponseSchema
    except Exception:
        StructuredOutputParser = None
        ResponseSchema = None

# Other deps
try:
    import cohere
except Exception:
    cohere = None

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    import nest_asyncio
except Exception:
    nest_asyncio = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# Load environment variables if python-dotenv is available
if load_dotenv:
    load_dotenv()

# Apply nest_asyncio if available
if nest_asyncio:
    try:
        nest_asyncio.apply()
    except Exception:
        pass

# --- Minimal fallback StructuredOutputParser ---
# This is a tiny, defensive parser that will attempt to parse JSON blocks
# from LLM output. It's a fallback so your app doesn't hard-fail when
# LangChain's StructuredOutputParser isn't available.
import json

class _SimpleStructuredOutputParser:
    def __init__(self, schema: List[Dict[str, Any]] = None):
        # schema is optional metadata; we don't enforce it in fallback
        self.schema = schema or []

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Try to extract the first JSON object from text. Returns parsed dict or raises ValueError.
        """
        # Try direct JSON
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try to find the first {...} block
        m = re.search(r'(\{(?:.|\n)*\})', text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        # Try to find first [...] block
        m = re.search(r'(\[(?:.|\n)*\])', text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        # As last resort, return raw text under a key
        return {"raw_text": text.strip()}

# Choose parser class: langchain's or fallback
_InternalStructuredOutputParser = StructuredOutputParser or _SimpleStructuredOutputParser

# --- ResponseFormatter (unchanged, minor safety tweaks) ---
class ResponseFormatter:
    """Class to format the response with proper citations and structure"""

    @staticmethod
    def format_answer(answer: str, cited_chunks: List[Dict]) -> str:
        if not answer:
            return "I couldn't generate an answer based on the available information."

        formatted_answer = str(answer).strip()
        formatted_answer = re.sub(r'\[(\d+)\]', r'[\1]', formatted_answer)

        formatted_response = "## Answer\n\n" + formatted_answer

        if cited_chunks:
            formatted_response += "\n\n## Sources\n\n"
            for i, chunk in enumerate(cited_chunks):
                source = chunk.get("metadata", {}).get("source", "Unknown")
                formatted_response += f"**[{i+1}]** {source}\n\n"

        return formatted_response

    @staticmethod
    def format_sources(cited_chunks: List[Dict]) -> str:
        if not cited_chunks:
            return ""

        formatted_sources = "## Detailed Sources\n\n"
        for i, chunk in enumerate(cited_chunks):
            source = chunk.get("metadata", {}).get("source", "Unknown")
            chunk_index = chunk.get("metadata", {}).get("chunk_index", 0)
            total_chunks = chunk.get("metadata", {}).get("total_chunks", 1)
            score = chunk.get("score", 0)
            rerank_score = chunk.get("rerank_score", 0)

            formatted_sources += f"### Source [{i+1}]: {source}\n\n"
            formatted_sources += f"**Relevance Score**: {score:.3f}\n\n"
            if rerank_score:
                formatted_sources += f"**Rerank Score**: {rerank_score:.3f}\n\n"
            formatted_sources += f"**Content**:\n{chunk.get('text','')}\n\n"
            formatted_sources += "---\n\n"

        return formatted_sources

# --- MiniRAGBackend (with defensive checks) ---
class MiniRAGBackend:
    def __init__(self):
        # Load API keys
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")

        # Basic checks for required libs, will raise helpful errors
        if Pinecone is None or pinecone is None:
            raise RuntimeError("Pinecone client not installed or not importable. Install `pinecone>=3.0.0`.")

        if GoogleGenerativeAIEmbeddings is None or ChatGoogleGenerativeAI is None:
            raise RuntimeError("Google GenAI LangChain integration not available. Install `langchain-google-genai` or check imports.")

        if RecursiveCharacterTextSplitter is None:
            raise RuntimeError("Text splitter package not available. Install `langchain-text-splitters` or fallback `langchain`.")

        if PyPDF2 is None:
            raise RuntimeError("PyPDF2 not available. Install `PyPDF2`.")

        # Check API keys presence
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        if not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")

        # Init Pinecone client (modern usage)
        # Note: if your pinecone package uses a different initialization pattern, update here.
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = "mini-rag-index"
        self.dimension = 768
        self.create_index()

        # Embeddings and LLM
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.google_api_key
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.google_api_key,
            temperature=0.1
        )

        # Cohere client
        if cohere is None:
            raise RuntimeError("Cohere SDK not installed. Install `cohere`.")
        self.cohere_client = cohere.Client(self.cohere_api_key)

        # Text splitter instance
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
        )

        # Response formatter and parser
        self.formatter = ResponseFormatter()
        # Use a structured output parser instance (fallback if missing)
        # If LangChain's StructuredOutputParser is a class, instantiate normally.
        try:
            if StructuredOutputParser:
                # If LangChain's parser expects a ResponseSchema list, try to create a trivial one
                self.structured_parser = StructuredOutputParser
            else:
                self.structured_parser = _SimpleStructuredOutputParser()
        except Exception:
            self.structured_parser = _SimpleStructuredOutputParser()

    def create_index(self):
        """Create Pinecone index if it doesn't exist"""
        idxs = []
        try:
            idxs = self.pc.list_indexes().names()
        except Exception:
            # Some pinecone clients return list directly
            try:
                idxs = self.pc.list_indexes()
            except Exception:
                idxs = []

        if self.index_name not in idxs:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for readiness (defensive)
            while True:
                try:
                    status = self.pc.describe_index(self.index_name).status
                    if isinstance(status, dict) and status.get('ready'):
                        break
                except Exception:
                    pass
                time.sleep(1)

        self.index = self.pc.Index(self.index_name)

    def extract_text_from_pdf(self, file_bytes):
        if PyPDF2 is None:
            raise RuntimeError("PyPDF2 not installed.")
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text_parts = []
            for page in pdf_reader.pages:
                ptext = page.extract_text() or ""
                text_parts.append(ptext)
            return "\n".join(text_parts)
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

    def chunk_document(self, text: str, source: str = "pasted_text") -> List[Dict]:
        chunks = self.text_splitter.split_text(text)
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "metadata": {"source": source, "chunk_index": i, "total_chunks": len(chunks)}
            })
        return documents

    def embed_and_upsert(self, documents: List[Dict]):
        texts = [doc["text"] for doc in documents]
        # Embedding calls might be sync; guard with event loop creation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            embeddings = self.embedding_model.embed_documents(texts)
        finally:
            try:
                loop.close()
            except Exception:
                pass

        vectors = []
        for doc, emb in zip(documents, embeddings):
            vectors.append({"id": doc["id"], "values": emb, "metadata": {"text": doc["text"], **doc["metadata"]}})
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i+100]
            self.index.upsert(vectors=batch)

    def process_input(self, text: str, source: str = "pasted_text"):
        documents = self.chunk_document(text, source)
        self.embed_and_upsert(documents)
        return len(documents)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            q_emb = self.embedding_model.embed_query(query)
        finally:
            try:
                loop.close()
            except Exception:
                pass

        results = self.index.query(vector=q_emb, top_k=top_k, include_metadata=True)
        matches = getattr(results, "matches", []) or results.get("matches", [])
        retrieved = []
        for m in matches:
            metadata = getattr(m, "metadata", {}) or m.get("metadata", {})
            score = getattr(m, "score", None) or m.get("score", None)
            retrieved.append({"text": metadata.get("text", ""), "score": score, "metadata": metadata})
        return retrieved

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        if not chunks:
            return []
        documents = [c["text"] for c in chunks]
        try:
            results = self.cohere_client.rerank(query=query, documents=documents, top_n=top_k, model="rerank-english-v2.0")
            reranked = []
            for res in results:
                original = chunks[res.index]
                original["rerank_score"] = res.relevance_score
                reranked.append(original)
            return reranked
        except Exception as e:
            print(f"Reranking failed: {e}")
            return chunks[:top_k]

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> Tuple[str, List[Dict]]:
        if not context_chunks:
            return "I couldn't find enough relevant information to answer your question.", []

        context_with_citations = ""
        for i, chunk in enumerate(context_chunks):
            source = chunk.get("metadata", {}).get("source", "Unknown")
            context_with_citations += f"[{i+1}] {chunk['text']}\nSource: {source}\n\n"

        prompt = f"""
Based on the context below, provide a concise answer to the question.
Use inline citations like [1], [2] for any information from the context.
If the context doesn't contain relevant information, say so.

Question: {query}

Context:
{context_with_citations}

Answer:
"""

        messages = []
        if SystemMessage:
            messages.append(SystemMessage(content="You are a helpful assistant that provides accurate, concise answers based on the given context. Always cite your sources using inline citations like [1], [2]."))
        if HumanMessage:
            messages.append(HumanMessage(content=prompt))
        else:
            # If message classes unavailable, pass the prompt in whatever shape ChatGoogleGenerativeAI expects
            messages = prompt

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = self.llm.invoke(messages)
            answer = getattr(response, "content", str(response))
        finally:
            try:
                loop.close()
            except Exception:
                pass

        # find cited chunks if answer contains [n]
        cited = []
        for i, chunk in enumerate(context_chunks):
            if f"[{i+1}]" in answer:
                cited.append(chunk)

        return answer, cited

    def query(self, query: str, retrieve_k: int = 10, rerank_k: int = 5) -> Dict[str, Any]:
        start_time = time.time()
        retrieved_chunks = self.retrieve(query, top_k=retrieve_k)
        retrieve_time = time.time() - start_time

        rerank_start = time.time()
        reranked_chunks = self.rerank(query, retrieved_chunks, top_k=rerank_k)
        rerank_time = time.time() - rerank_start

        gen_start = time.time()
        answer, cited_chunks = self.generate_answer(query, reranked_chunks)
        generate_time = time.time() - gen_start

        formatted_answer = self.formatter.format_answer(answer, cited_chunks)
        detailed_sources = self.formatter.format_sources(cited_chunks)

        total_time = time.time() - start_time

        return {
            "answer": formatted_answer,
            "detailed_sources": detailed_sources,
            "cited_chunks": cited_chunks,
            "retrieved_chunks": retrieved_chunks,
            "reranked_chunks": reranked_chunks,
            "timing": {"total": total_time, "retrieve": retrieve_time, "rerank": rerank_time, "generate": generate_time}
        }
