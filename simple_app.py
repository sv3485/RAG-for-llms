"""
Simple RAG App for testing
This is a minimal version to test the basic functionality
"""

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG System - Large Language Models Research",
    page_icon="🤖",
    layout="wide"
)

def main():
    st.title("🤖 RAG System for Large Language Models Research")
    st.markdown("Ask questions about Large Language Models based on recent ArXiv research papers")
    
    # Check API keys
    gemini_key = os.getenv('GEMINI_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    col1, col2 = st.columns(2)
    with col1:
        if gemini_key:
            st.success("✅ Gemini API Key configured")
        else:
            st.error("❌ Gemini API Key missing")
    
    with col2:
        if openai_key:
            st.success("✅ OpenAI API Key configured")
        else:
            st.warning("⚠️ OpenAI API Key missing (needed for evaluation)")
    
    # Simple interface
    st.markdown("### 💬 Ask a Question")
    
    question = st.text_area(
        "Enter your question about Large Language Models:",
        placeholder="e.g., What are the main challenges in training large language models?",
        height=100
    )
    
    if st.button("🔍 Get Answer", type="primary"):
        if question.strip():
            st.info("🚧 The full RAG system is being set up. This is a demo interface.")
            st.write(f"**Your question:** {question}")
            st.write("**Demo response:** The RAG system would process your question by:")
            st.write("1. 📚 Retrieving relevant papers from ArXiv")
            st.write("2. 🔍 Finding similar content using embeddings")
            st.write("3. 🤖 Generating an answer using Gemini Pro")
            st.write("4. 📊 Providing context from the retrieved papers")
        else:
            st.warning("Please enter a question.")
    
    # Instructions
    st.markdown("---")
    st.markdown("### 📋 Setup Instructions")
    st.markdown("""
    1. **Add API Keys**: Edit the `.env` file and add your API keys:
       ```
       GEMINI_API_KEY=your_gemini_api_key_here
       OPENAI_API_KEY=your_openai_api_key_here
       ```
    
    2. **Install Dependencies**: Run `pip install -r requirements.txt`
    
    3. **Run Full App**: Use `streamlit run app.py` for the complete system
    
    4. **Build Knowledge Base**: The system will automatically fetch 500 ArXiv papers on first run
    """)
    
    # Status
    st.markdown("### 🔧 System Status")
    if os.path.exists("./chroma_db"):
        st.success("✅ Knowledge base exists")
    else:
        st.info("ℹ️ Knowledge base will be built on first run")
    
    # Example questions
    st.markdown("### 💡 Example Questions")
    examples = [
        "What are the main challenges in training large language models?",
        "How do transformer architectures work in LLMs?",
        "What are the latest developments in prompt engineering?",
        "How can we evaluate the performance of large language models?",
        "What are the ethical considerations in deploying LLMs?"
    ]
    
    for example in examples:
        if st.button(f"📝 {example}", key=f"example_{example}"):
            st.session_state.example_question = example
            st.rerun()
    
    if hasattr(st.session_state, 'example_question'):
        st.text_area("Selected example:", value=st.session_state.example_question, height=50)

if __name__ == "__main__":
    main()

