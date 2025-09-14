#!/usr/bin/env python3
"""
Script to build the knowledge base for the RAG system
"""

import os
from dotenv import load_dotenv
from rag_pipeline_simple import RAGPipeline

def main():
    print("🚀 Building RAG Knowledge Base")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("Please add your Gemini API key to the .env file")
        return
    
    print("✅ API key found")
    
    try:
        # Initialize RAG pipeline
        print("📚 Initializing RAG pipeline...")
        rag = RAGPipeline()
        
        # Build knowledge base with a smaller dataset of 100 papers
        print("🔍 Fetching papers from ArXiv...")
        print("This may take 5-10 minutes depending on your internet connection...")
        
        # Fetch 100 papers for knowledge base as requested
        rag.data_fetcher.max_papers = 100  # Reduced dataset of 100 papers
        num_chunks = rag.build_knowledge_base(query="Large Language Models")
        
        print(f"✅ Knowledge base built successfully!")
        print(f"📊 Total chunks created: {num_chunks}")
        print("\n🎉 You can now use the RAG system to ask questions!")
        print("Run: streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Error building knowledge base: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify your Gemini API key is valid")
        print("3. Check if you have sufficient API quota")

if __name__ == "__main__":
    main()
