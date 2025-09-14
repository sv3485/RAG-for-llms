"""
RAG Pipeline Implementation
Handles data fetching, processing, embeddings, and retrieval-augmented generation
"""

import os
import re
import arxiv
import tiktoken
import chromadb
import requests
import PyPDF2
import io
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArXivDataFetcher:
    """Handles fetching and processing ArXiv papers"""
    
    def __init__(self, max_papers: int = 500):
        self.max_papers = max_papers
        self.client = arxiv.Client()
    
    def fetch_papers(self, query: str = "Large Language Models", max_results: int = None) -> List[Dict[str, Any]]:
        """Fetch papers from ArXiv"""
        if max_results is None:
            max_results = self.max_papers
            
        logger.info(f"Fetching {max_results} papers on '{query}' from ArXiv...")
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for paper in self.client.results(search):
            paper_data = {
                'id': paper.entry_id,
                'title': paper.title,
                'abstract': paper.summary,
                'authors': [author.name for author in paper.authors],
                'published': paper.published.isoformat(),
                'pdf_url': paper.pdf_url,
                'categories': paper.categories,
                'content': None  # Will be populated if PDF is available
            }
            papers.append(paper_data)
        
        logger.info(f"Successfully fetched {len(papers)} papers")
        return papers
    
    def extract_pdf_content(self, pdf_url: str) -> Optional[str]:
        """Extract text content from PDF URL"""
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            logger.warning(f"Failed to extract PDF content from {pdf_url}: {e}")
            return None

class TextProcessor:
    """Handles text preprocessing and chunking"""
    
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        # Clean the text first
        text = self.clean_text(text)
        
        # Split into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'tokens': current_tokens,
                    'metadata': metadata or {}
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self.count_tokens(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'tokens': current_tokens,
                'metadata': metadata or {}
            })
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        words = text.split()
        overlap_words = words[-self.chunk_overlap//4:]  # Approximate word-based overlap
        return " ".join(overlap_words)

class VectorStore:
    """Handles embeddings and vector storage with ChromaDB"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="arxiv_papers",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store"""
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        texts = [doc['text'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info("Documents added to vector store successfully")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results

class RAGPipeline:
    """Main RAG pipeline orchestrating all components"""
    
    def __init__(self):
        self.data_fetcher = ArXivDataFetcher()
        self.text_processor = TextProcessor()
        self.vector_store = VectorStore()
        
        # Initialize Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def build_knowledge_base(self, query: str = "Large Language Models"):
        """Build the complete knowledge base from ArXiv papers"""
        logger.info("Building knowledge base...")
        
        # Fetch papers
        papers = self.data_fetcher.fetch_papers(query)
        
        # Process papers and create chunks
        all_chunks = []
        for i, paper in enumerate(papers):
            logger.info(f"Processing paper {i+1}/{len(papers)}: {paper['title'][:50]}...")
            
            # Create metadata
            metadata = {
                'paper_id': paper['id'],
                'title': paper['title'],
                'authors': ', '.join(paper['authors']),
                'published': paper['published'],
                'categories': ', '.join(paper['categories']),
                'source': 'arxiv'
            }
            
            # Process abstract
            abstract_chunks = self.text_processor.chunk_text(
                paper['abstract'], 
                {**metadata, 'content_type': 'abstract'}
            )
            all_chunks.extend(abstract_chunks)
            
            # Try to extract PDF content
            pdf_content = self.data_fetcher.extract_pdf_content(paper['pdf_url'])
            if pdf_content:
                pdf_chunks = self.text_processor.chunk_text(
                    pdf_content,
                    {**metadata, 'content_type': 'pdf_content'}
                )
                all_chunks.extend(pdf_chunks)
        
        # Add to vector store
        self.vector_store.add_documents(all_chunks)
        logger.info(f"Knowledge base built with {len(all_chunks)} chunks")
        
        return len(all_chunks)
    
    def query(self, question: str, n_context: int = 5) -> Dict[str, Any]:
        """Process a query through the RAG pipeline"""
        logger.info(f"Processing query: {question}")
        
        # Retrieve relevant context
        context_docs = self.vector_store.search(question, n_results=n_context)
        
        # Prepare context for LLM
        context_text = "\n\n".join([
            f"Title: {doc['metadata'].get('title', 'Unknown')}\n"
            f"Content: {doc['text']}"
            for doc in context_docs
        ])
        
        # Create prompt
        prompt = f"""Based on the following research papers about Large Language Models, please answer the question: {question}

Context from research papers:
{context_text}

Please provide a comprehensive answer based on the provided context. If the context doesn't contain enough information to answer the question, please state that clearly."""

        # Generate response using Gemini
        try:
            response = self.model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            answer = "I apologize, but I encountered an error while generating a response. Please try again."
        
        return {
            'question': question,
            'answer': answer,
            'context_documents': context_docs,
            'context_text': context_text
        }

# Example usage
if __name__ == "__main__":
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Build knowledge base (this will take a while)
    print("Building knowledge base...")
    num_chunks = rag.build_knowledge_base()
    print(f"Knowledge base built with {num_chunks} chunks")
    
    # Example query
    result = rag.query("What are the main challenges in training large language models?")
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")

