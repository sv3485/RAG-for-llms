"""
Streamlit RAG Application
Main entry point for the RAG system with web interface
"""

import streamlit as st
import os
import time
import json
import pickle
from typing import Dict, Any
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG System - Large Language Models Research",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #2d3748;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00ff88;
        margin: 1rem 0;
        color: #ffffff;
        font-size: 16px;
        line-height: 1.6;
        box-shadow: 0 4px 8px rgba(0,255,136,0.3);
        border: 1px solid #4a5568;
    }
    .context-box {
        background-color: #1a202c;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
        margin: 0.5rem 0;
        color: #ffffff;
        font-size: 14px;
        line-height: 1.5;
        box-shadow: 0 4px 8px rgba(255,107,107,0.3);
        border: 1px solid #4a5568;
    }
    .metric-box {
        background-color: #d1ecf1;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
        text-align: center;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
    /* Dark theme - Ensure all text is visible */
    .stMarkdown {
        color: #ffffff;
    }
    .stTextArea > div > div > textarea {
        color: #ffffff;
        background-color: #2d3748;
        border: 1px solid #4a5568;
    }
    .stSelectbox > div > div {
        color: #ffffff;
        background-color: #2d3748;
        border: 1px solid #4a5568;
    }
    /* Improve answer display */
    .answer-text {
        color: #ffffff;
        font-size: 16px;
        line-height: 1.6;
        margin: 0;
    }
    /* Dark theme background */
    .main .block-container {
        background-color: #1a202c;
        padding: 2rem;
    }
    /* Dark theme for entire app */
    .stApp > div {
        background-color: #1a202c;
    }
    /* Dark theme for sidebar */
    .css-1d391kg {
        background-color: #2d3748;
    }
    /* Context section styling - Dark theme */
    .context-section {
        background-color: #2d3748;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #4a5568;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .context-header {
        color: #00ff88;
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 0.5rem;
    }
    .context-item {
        background-color: #1a202c;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        border-left: 3px solid #ff6b6b;
        border: 1px solid #4a5568;
        color: #ffffff;
        box-shadow: 0 2px 4px rgba(255,107,107,0.2);
    }
    /* Additional dark theme elements */
    .stExpander {
        background-color: #2d3748;
        border: 1px solid #4a5568;
    }
    .stExpander > div > div {
        background-color: #2d3748;
        color: #ffffff;
    }
    /* Headers and titles */
    h1, h2, h3, h4, h5, h6 {
        color: #00ff88 !important;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2d3748 !important;
    }
    /* Button styling for dark theme */
    .stButton > button {
        background-color: #00ff88;
        color: #1a202c;
        border: none;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #00cc6a;
        color: #1a202c;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_pipeline():
    """Load and cache the RAG pipeline"""
    try:
        from rag_pipeline_simple import RAGPipeline
        return RAGPipeline()
    except Exception as e:
        st.error(f"Failed to load RAG pipeline: {e}")
        return None

@st.cache_resource
def load_evaluator():
    """Load and cache the evaluator"""
    try:
        from evaluation import RAGASEvaluator
        rag = load_rag_pipeline()
        if rag:
            return RAGASEvaluator(rag)
        return None
    except Exception as e:
        st.error(f"Failed to load evaluator: {e}")
        return None

def display_sidebar():
    """Display sidebar with system information and controls"""
    st.sidebar.title("🤖 RAG System")
    
    st.sidebar.markdown("### System Information")
    st.sidebar.info("""
    This RAG system is built with:
    - **Data Source**: ArXiv papers on Large Language Models
    - **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
    - **Vector Store**: FAISS
    - **LLM**: Google Gemini Pro
    - **Evaluation**: RAGAS framework
    """)
    
    st.sidebar.markdown("### Configuration")
    
    # API Key status
    gemini_key = os.getenv('GEMINI_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if gemini_key:
        st.sidebar.success("✅ Gemini API Key configured")
    else:
        st.sidebar.error("❌ Gemini API Key missing")
    
    if openai_key:
        st.sidebar.success("✅ OpenAI API Key configured")
    else:
        st.sidebar.warning("⚠️ OpenAI API Key missing (needed for evaluation)")
    
    # Knowledge base status
    if os.path.exists("./faiss_db"):
        try:
            # Using pickle imported at the global level
            # Check if the documents pickle file exists and has content
            if os.path.exists("./faiss_db/documents.pkl"):
                with open("./faiss_db/documents.pkl", "rb") as f:
                    documents = pickle.load(f)
                count = len(documents)
                if count > 0:
                    st.sidebar.success(f"✅ Knowledge base ready ({count} documents)")
                else:
                    st.sidebar.warning("⚠️ Knowledge base is empty")
            else:
                st.sidebar.warning("⚠️ Knowledge base exists but may be corrupted")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Knowledge base exists but may be corrupted: {str(e)}")
    else:
        st.sidebar.warning("⚠️ Knowledge base not built yet")
    
    st.sidebar.markdown("### Quick Actions")
    
    if st.sidebar.button("🔄 Rebuild Knowledge Base"):
        st.session_state.rebuild_kb = True
    
    if st.sidebar.button("📊 Run Evaluation"):
        st.session_state.run_evaluation = True

def display_main_interface():
    """Display the main query interface"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG System for Large Language Models Research</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about Large Language Models based on recent ArXiv research papers</p>', unsafe_allow_html=True)
    
    # Load RAG pipeline
    rag = load_rag_pipeline()
    if not rag:
        st.error("RAG pipeline could not be loaded. Please check your configuration.")
        return
    
    # Check if knowledge base is empty
    try:
        # Using pickle and os imported at the global level
        if not os.path.exists("./faiss_db") or not os.path.exists("./faiss_db/documents.pkl"):
            st.warning("⚠️ **Knowledge base not found!** Please build the knowledge base first by clicking 'Rebuild Knowledge Base' in the sidebar.")
            return
            
        with open("./faiss_db/documents.pkl", "rb") as f:
            documents = pickle.load(f)
        
        if len(documents) == 0:
            st.warning("⚠️ **Knowledge base is empty!** Please build the knowledge base first by clicking 'Rebuild Knowledge Base' in the sidebar.")
            st.info("💡 **Tip:** The knowledge base building process will fetch 500 papers from ArXiv about Large Language Models. This may take 5-10 minutes.")
            return
    except Exception as e:
        st.warning(f"⚠️ **Knowledge base error!** Please rebuild the knowledge base: {str(e)}")
        return
    
    # Query input
    st.markdown("### 💬 Ask a Question")
    
    # Example questions
    example_questions = [
        "What are the main challenges in training large language models?",
        "How do transformer architectures work in LLMs?",
        "What are the latest developments in prompt engineering?",
        "How can we evaluate the performance of large language models?",
        "What are the ethical considerations in deploying LLMs?"
    ]
    
    selected_example = st.selectbox("Or choose an example question:", ["Select an example..."] + example_questions)
    
    if selected_example != "Select an example...":
        default_question = selected_example
    else:
        default_question = ""
    
    question = st.text_area(
        "Enter your question about Large Language Models:",
        value=default_question,
        height=100,
        placeholder="e.g., What are the main challenges in training large language models?"
    )
    
    # Query parameters
    col1, col2 = st.columns(2)
    with col1:
        n_context = st.slider("Number of context documents:", min_value=3, max_value=10, value=5)
    with col2:
        show_context = st.checkbox("Show retrieved context", value=True)
    
    # Submit button
    if st.button("🔍 Get Answer", type="primary"):
        if question.strip():
            with st.spinner("Processing your question..."):
                try:
                    # Get RAG response
                    result = rag.query(question, n_context=n_context)
                    
                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.markdown(f'<div class="answer-box"><p class="answer-text">{result["answer"]}</p></div>', unsafe_allow_html=True)
                    
                    # Display context if requested
                    if show_context and result['context_documents']:
                        st.markdown("### 📚 Retrieved Context")
                        st.markdown('<div class="context-section">', unsafe_allow_html=True)
                        
                        for i, doc in enumerate(result['context_documents'], 1):
                            with st.expander(f"Context {i}: {doc['metadata'].get('title', 'Unknown Title')[:50]}..."):
                                st.markdown(f"**Source:** {doc['metadata'].get('title', 'Unknown')}")
                                st.markdown(f"**Authors:** {doc['metadata'].get('authors', 'Unknown')}")
                                st.markdown(f"**Published:** {doc['metadata'].get('published', 'Unknown')}")
                                st.markdown(f"**Content Type:** {doc['metadata'].get('content_type', 'Unknown')}")
                                st.markdown(f"**Similarity Score:** {1 - doc['distance']:.3f}")
                                st.markdown("**Content:**")
                                st.markdown(f'<div class="context-item">{doc["text"]}</div>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Store result in session state for evaluation
                    st.session_state.last_result = result
                    
                except Exception as e:
                    st.error(f"Error processing question: {e}")
                    logger.error(f"Error in query processing: {e}")
        else:
            st.warning("Please enter a question.")

def display_evaluation_interface():
    """Display the evaluation interface"""
    
    st.markdown("### 📊 System Evaluation")
    
    evaluator = load_evaluator()
    if not evaluator:
        st.error("Evaluator could not be loaded. Please check your configuration.")
        return
    
    st.markdown("""
    The RAGAS evaluation framework measures:
    - **Faithfulness**: Factual consistency of generated answers
    - **Answer Relevancy**: How relevant answers are to questions
    - **Context Precision**: How precise retrieved context is
    - **Context Recall**: How well context covers the answer
    - **Context Relevancy**: How relevant context is to questions
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        num_questions = st.slider("Number of test questions:", min_value=3, max_value=20, value=5)
    with col2:
        if st.button("🚀 Run Evaluation", type="primary"):
            with st.spinner("Running evaluation (this may take a few minutes)..."):
                try:
                    scores = evaluator.evaluate_rag_system(num_test_questions=num_questions)
                    
                    if 'error' in scores:
                        st.error(f"Evaluation failed: {scores['error']}")
                    else:
                        # Display results
                        st.markdown("### 📈 Evaluation Results")
                        
                        # Create columns for metrics
                        cols = st.columns(5)
                        metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'context_relevancy']
                        
                        for i, metric in enumerate(metrics):
                            with cols[i]:
                                score = scores[metric]
                                color = "green" if score >= 0.7 else "orange" if score >= 0.5 else "red"
                                st.markdown(f"""
                                <div class="metric-box">
                                    <h4>{metric.replace('_', ' ').title()}</h4>
                                    <h2 style="color: {color}">{score:.3f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Overall score
                        overall_score = scores['overall']
                        st.markdown(f"""
                        <div class="metric-box" style="background-color: #e8f5e8; margin-top: 1rem;">
                            <h3>Overall Score</h3>
                            <h1 style="color: {'green' if overall_score >= 0.7 else 'orange' if overall_score >= 0.5 else 'red'}">{overall_score:.3f}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Interpretation
                        if overall_score >= 0.8:
                            st.success("🟢 Excellent performance - RAG system is working very well!")
                        elif overall_score >= 0.6:
                            st.info("🟡 Good performance - RAG system is working well with room for improvement")
                        elif overall_score >= 0.4:
                            st.warning("🟠 Fair performance - RAG system needs significant improvements")
                        else:
                            st.error("🔴 Poor performance - RAG system requires major improvements")
                        
                        # Save results
                        evaluator.save_evaluation_results(scores)
                        st.success("Evaluation results saved to evaluation_results.json")
                        
                except Exception as e:
                    st.error(f"Error during evaluation: {e}")
                    logger.error(f"Error in evaluation: {e}")

def handle_knowledge_base_rebuild():
    """Handle knowledge base rebuilding"""
    if hasattr(st.session_state, 'rebuild_kb') and st.session_state.rebuild_kb:
        st.markdown("### 🔄 Rebuilding Knowledge Base")
        
        rag = load_rag_pipeline()
        if rag:
            with st.spinner("Rebuilding knowledge base (this will take several minutes)..."):
                try:
                    num_chunks = rag.build_knowledge_base()
                    st.success(f"Knowledge base rebuilt successfully with {num_chunks} chunks!")
                except Exception as e:
                    st.error(f"Error rebuilding knowledge base: {e}")
                    logger.error(f"Error rebuilding knowledge base: {e}")
        
        st.session_state.rebuild_kb = False

def main():
    """Main application function"""
    
    # Display sidebar
    display_sidebar()
    
    # Handle knowledge base rebuild
    handle_knowledge_base_rebuild()
    
    # Main interface tabs
    tab1, tab2 = st.tabs(["🔍 Query Interface", "📊 Evaluation"])
    
    with tab1:
        display_main_interface()
    
    with tab2:
        display_evaluation_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Built with Streamlit, ChromaDB, SentenceTransformers, and Google Gemini</p>
        <p>Data source: ArXiv papers on Large Language Models</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
