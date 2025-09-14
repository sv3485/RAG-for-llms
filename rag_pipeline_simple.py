"""
Simplified RAG Pipeline Implementation
Handles data fetching, processing, embeddings, and retrieval-augmented generation
"""

import os
import re
import arxiv
import tiktoken
import requests
import PyPDF2
import io
import pickle
from typing import List, Dict, Any, Optional
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
        """Fetch papers from ArXiv with robust error handling"""
        if max_results is None:
            max_results = self.max_papers
            
        logger.info(f"Fetching {max_results} papers on '{query}' from ArXiv...")
        
        try:
            # Use smaller batch sizes to avoid API issues
            batch_size = 100
            papers = []
            start = 0
            
            while len(papers) < max_results:
                remaining = max_results - len(papers)
                current_batch_size = min(batch_size, remaining)
                
                logger.info(f"Fetching batch: {len(papers)+1}-{len(papers)+current_batch_size} of {max_results}")
                
                search = arxiv.Search(
                    query=query,
                    max_results=current_batch_size,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                
                batch_papers = []
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
                    batch_papers.append(paper_data)
                    
                    if len(batch_papers) >= current_batch_size:
                        break
                
                if not batch_papers:
                    logger.warning(f"No more papers found at batch starting from {start}")
                    break
                    
                papers.extend(batch_papers)
                start += current_batch_size
                
                # Add a small delay between batches to be respectful to the API
                if len(papers) < max_results:
                    import time
                    time.sleep(1)
            
            logger.info(f"Successfully fetched {len(papers)} papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching papers from ArXiv: {e}")
            # Return some sample papers if ArXiv fails
            return self._get_sample_papers()
    
    def _get_sample_papers(self) -> List[Dict[str, Any]]:
        """Return sample papers if ArXiv API fails"""
        logger.info("Using sample papers due to ArXiv API issues")
        return [
            {
                'id': 'sample-1',
                'title': 'Large Language Models: A Survey',
                'abstract': 'Large Language Models (LLMs) have revolutionized natural language processing. This survey covers the architecture, training methods, and applications of LLMs including GPT, BERT, and T5 models.',
                'authors': ['John Doe', 'Jane Smith'],
                'published': '2024-01-01T00:00:00Z',
                'pdf_url': 'https://example.com/paper1.pdf',
                'categories': ['cs.CL', 'cs.AI'],
                'content': None
            },
            {
                'id': 'sample-2',
                'title': 'Transformer Architecture in Large Language Models',
                'abstract': 'The transformer architecture forms the backbone of modern large language models. This paper explores the attention mechanism, positional encoding, and multi-head attention in transformer-based LLMs.',
                'authors': ['Alice Johnson', 'Bob Wilson'],
                'published': '2024-01-02T00:00:00Z',
                'pdf_url': 'https://example.com/paper2.pdf',
                'categories': ['cs.CL', 'cs.LG'],
                'content': None
            },
            {
                'id': 'sample-3',
                'title': 'Training Challenges in Large Language Models',
                'abstract': 'Training large language models presents numerous challenges including computational requirements, data quality, optimization difficulties, and ethical considerations. This paper discusses these challenges and potential solutions.',
                'authors': ['Charlie Brown', 'Diana Prince'],
                'published': '2024-01-03T00:00:00Z',
                'pdf_url': 'https://example.com/paper3.pdf',
                'categories': ['cs.CL', 'cs.AI'],
                'content': None
            }
        ]
    
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

class SimpleVectorStore:
    """Simple vector store using file-based storage"""
    
    def __init__(self, persist_directory: str = "./faiss_db"):
        self.persist_directory = persist_directory
        self.documents = []
        self.metadata = []
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Try to load existing data
        try:
            if os.path.exists(os.path.join(persist_directory, "documents.pkl")):
                with open(os.path.join(persist_directory, "documents.pkl"), "rb") as f:
                    self.documents = pickle.load(f)
                    
                with open(os.path.join(persist_directory, "metadata.pkl"), "rb") as f:
                    self.metadata = pickle.load(f)
                    
                logger.info(f"Loaded existing documents: {len(self.documents)} documents")
        except Exception as e:
            logger.warning(f"Could not load documents: {e}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store"""
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        try:
            texts = [doc['text'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            
            # Store documents and metadata
            self.documents.extend(texts)
            self.metadata.extend(metadatas)
            
            # Save to disk
            with open(os.path.join(self.persist_directory, "documents.pkl"), "wb") as f:
                pickle.dump(self.documents, f)
                
            with open(os.path.join(self.persist_directory, "metadata.pkl"), "wb") as f:
                pickle.dump(self.metadata, f)
                
            logger.info("Documents added successfully")
        except Exception as e:
            logger.warning(f"Failed to add documents: {e}")
            # Fallback to simple storage without saving
            self.documents.extend([doc['text'] for doc in documents])
            self.metadata.extend([doc['metadata'] for doc in documents])
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using simple text matching"""
        query_lower = query.lower()
        scored_docs = []
        
        for i, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            # Simple scoring based on word overlap
            query_words = set(query_lower.split())
            doc_words = set(doc_lower.split())
            overlap = len(query_words.intersection(doc_words))
            score = overlap / len(query_words) if query_words else 0
            
            scored_docs.append({
                'text': doc,
                'metadata': self.metadata[i] if i < len(self.metadata) else {},
                'distance': 1 - score
            })
        
        # Sort by score and return top results
        scored_docs.sort(key=lambda x: x['distance'])
        return scored_docs[:n_results]

class RAGPipeline:
    """Main RAG pipeline orchestrating all components"""
    
    def __init__(self):
        self.data_fetcher = ArXivDataFetcher()
        self.text_processor = TextProcessor()
        self.vector_store = SimpleVectorStore()
        
        # Initialize Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        # Use the more efficient flash model to reduce quota usage
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
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
        
        # Check if knowledge base has documents
        if not self.vector_store.documents and not self.vector_store.collection:
            return {
                'question': question,
                'answer': "The knowledge base is empty. Please build the knowledge base first by clicking 'Rebuild Knowledge Base' in the sidebar.",
                'context_documents': [],
                'context_text': ""
            }
        
        # Retrieve relevant context
        context_docs = self.vector_store.search(question, n_results=n_context)
        
        # Check if we found any context
        if not context_docs:
            return {
                'question': question,
                'answer': "I couldn't find any relevant context in the knowledge base to answer your question. The knowledge base might be empty or the question might be outside the scope of the available documents.",
                'context_documents': [],
                'context_text': ""
            }
        
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
            if "quota" in str(e).lower() or "429" in str(e):
                answer = "I apologize, but I've reached the API quota limit. Please wait a few minutes and try again, or check your Gemini API billing settings."
            elif "api key" in str(e).lower():
                answer = "I apologize, but there's an issue with the API key configuration. Please check your Gemini API key."
            else:
                answer = f"I apologize, but I encountered an error while generating a response: {str(e)[:200]}..."
        
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
