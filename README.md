# RAG System for Large Language Models Research

A comprehensive Retrieval-Augmented Generation (RAG) application that answers questions about Large Language Models using recent ArXiv research papers.

## 🚀 Features

- **Data Source**: Fetches 500 recent papers on "Large Language Models" from ArXiv API
- **Text Processing**: Cleans and chunks papers into 500-800 token passages
- **Embeddings**: Uses SentenceTransformers ("all-MiniLM-L6-v2") for vector embeddings
- **Vector Store**: ChromaDB for efficient similarity search
- **LLM Integration**: Google Gemini Pro for answer generation
- **Evaluation**: RAGAS framework for comprehensive system evaluation
- **Web Interface**: Streamlit-based UI for easy interaction
- **Deployment Ready**: Configured for Streamlit Community Cloud

## 📋 Prerequisites

- Python 3.8+
- Google Gemini API key
- OpenAI API key (for RAGAS evaluation)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd RAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env_example.txt .env
   ```
   
   Edit `.env` and add your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🚀 Usage

### Local Development

1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - Open your browser to `http://localhost:8501`
   - The app will automatically build the knowledge base on first run

### Building Knowledge Base

The knowledge base is built automatically when you first run the application. This process:
- Fetches 500 papers from ArXiv on "Large Language Models"
- Extracts text from abstracts and PDFs
- Chunks text into 500-800 token passages
- Generates embeddings and stores in ChromaDB

**Note**: Building the knowledge base takes 10-15 minutes depending on your internet connection.

### Querying the System

1. Enter a question about Large Language Models
2. Choose the number of context documents to retrieve (3-10)
3. Click "Get Answer" to see the response
4. View retrieved context documents for transparency

### Evaluation

Use the Evaluation tab to:
- Run RAGAS evaluation with 3-20 test questions
- View detailed metrics (Faithfulness, Answer Relevancy, etc.)
- Get overall system performance score

## 📊 Evaluation Metrics

The system uses RAGAS to evaluate:

- **Faithfulness**: Factual consistency of generated answers
- **Answer Relevancy**: How relevant answers are to questions  
- **Context Precision**: How precise retrieved context is
- **Context Recall**: How well context covers the answer
- **Context Relevancy**: How relevant context is to questions

## 🏗️ Project Structure

```
RAG/
├── app.py                 # Streamlit web interface
├── rag_pipeline.py        # Core RAG pipeline implementation
├── evaluation.py          # RAGAS evaluation framework
├── requirements.txt       # Python dependencies
├── env_example.txt        # Environment variables template
├── README.md             # This file
└── chroma_db/            # ChromaDB vector store (created automatically)
```

## 🔧 Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `OPENAI_API_KEY`: Your OpenAI API key (required for evaluation)
- `CHROMA_PERSIST_DIRECTORY`: ChromaDB storage directory (default: ./chroma_db)
- `MAX_PAPERS`: Maximum number of papers to fetch (default: 500)
- `CHUNK_SIZE`: Text chunk size in tokens (default: 600)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100)

### Customization

You can modify the system by:
- Changing the ArXiv query in `rag_pipeline.py`
- Adjusting chunk size and overlap parameters
- Using different embedding models
- Modifying the evaluation metrics

## 🚀 Deployment

### Streamlit Community Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial RAG system implementation"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set environment variables in the Streamlit Cloud dashboard
   - Deploy!

### Environment Variables for Deployment

Set these in your Streamlit Cloud deployment:
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`

## 🔍 Example Queries

Try these example questions:
- "What are the main challenges in training large language models?"
- "How do transformer architectures work in LLMs?"
- "What are the latest developments in prompt engineering?"
- "How can we evaluate the performance of large language models?"
- "What are the ethical considerations in deploying LLMs?"

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure your API keys are correctly set in the environment variables
   - Check that you have sufficient API quota

2. **Knowledge Base Building Fails**
   - Check your internet connection
   - Verify ArXiv API is accessible
   - Ensure sufficient disk space for ChromaDB

3. **Evaluation Fails**
   - Verify OpenAI API key is set
   - Check that you have sufficient OpenAI API quota

### Performance Tips

- The knowledge base is cached after first build
- Use fewer context documents for faster responses
- Consider reducing the number of papers for faster initial setup

## 📈 Future Improvements

- [ ] Hybrid retrieval (dense + BM25)
- [ ] Microsoft GraphRAG integration
- [ ] Multi-modal support (images, tables)
- [ ] Real-time paper updates
- [ ] Advanced filtering and search
- [ ] Export functionality for results

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

# RAG-for-llms
