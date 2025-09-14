"""
Test script for the RAG system
Verifies that all components work correctly
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import arxiv
        print("✅ ArXiv imported successfully")
    except ImportError as e:
        print(f"❌ ArXiv import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✅ SentenceTransformers imported successfully")
    except ImportError as e:
        print(f"❌ SentenceTransformers import failed: {e}")
        return False
    
    try:
        import chromadb
        print("✅ ChromaDB imported successfully")
    except ImportError as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False
    
    try:
        import google.generativeai
        print("✅ Google Generative AI imported successfully")
    except ImportError as e:
        print(f"❌ Google Generative AI import failed: {e}")
        return False
    
    try:
        import ragas
        print("✅ RAGAS imported successfully")
    except ImportError as e:
        print(f"❌ RAGAS import failed: {e}")
        return False
    
    return True

def test_environment():
    """Test environment variables"""
    print("\nTesting environment variables...")
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        print("✅ GEMINI_API_KEY is set")
    else:
        print("❌ GEMINI_API_KEY is not set")
        return False
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("✅ OPENAI_API_KEY is set")
    else:
        print("⚠️ OPENAI_API_KEY is not set (needed for evaluation)")
    
    return True

def test_rag_pipeline():
    """Test RAG pipeline initialization"""
    print("\nTesting RAG pipeline initialization...")
    
    try:
        from rag_pipeline import RAGPipeline
        rag = RAGPipeline()
        print("✅ RAG pipeline initialized successfully")
        return True
    except Exception as e:
        print(f"❌ RAG pipeline initialization failed: {e}")
        return False

def test_evaluation():
    """Test evaluation framework"""
    print("\nTesting evaluation framework...")
    
    try:
        from evaluation import RAGASEvaluator, GoldTestSetGenerator
        print("✅ Evaluation framework imported successfully")
        return True
    except Exception as e:
        print(f"❌ Evaluation framework import failed: {e}")
        return False

def test_streamlit_app():
    """Test Streamlit app"""
    print("\nTesting Streamlit app...")
    
    try:
        # Check if app.py exists and can be imported
        if os.path.exists('app.py'):
            print("✅ app.py exists")
            return True
        else:
            print("❌ app.py not found")
            return False
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running RAG System Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_environment,
        test_rag_pipeline,
        test_evaluation,
        test_streamlit_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nTo run the application:")
        print("streamlit run app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\nTo install missing dependencies:")
        print("pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

