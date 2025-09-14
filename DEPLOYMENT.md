# Deployment Guide

This guide will help you deploy the RAG system to Streamlit Community Cloud.

## 🚀 Quick Deployment Steps

### 1. Prepare Your Repository

1. **Initialize Git repository** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial RAG system implementation"
   ```

2. **Push to GitHub**:
   ```bash
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

### 2. Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**:
   - Click "New app"
   - Select your GitHub repository
   - Choose the main branch
   - Set the main file path to `streamlit_app.py`

3. **Configure Environment Variables**:
   In the Streamlit Cloud dashboard, add these secrets:
   ```
   GEMINI_API_KEY = your_gemini_api_key_here
   OPENAI_API_KEY = your_openai_api_key_here
   ```

4. **Deploy**:
   - Click "Deploy!"
   - Wait for the deployment to complete (5-10 minutes)

### 3. Post-Deployment

1. **First Run**:
   - The app will automatically build the knowledge base on first access
   - This process takes 10-15 minutes
   - You'll see a loading indicator during this time

2. **Monitor Performance**:
   - Check the Streamlit Cloud logs for any issues
   - Monitor API usage in your Gemini and OpenAI dashboards

## 🔧 Configuration Options

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key | - |
| `OPENAI_API_KEY` | Yes | OpenAI API key for evaluation | - |
| `CHROMA_PERSIST_DIRECTORY` | No | ChromaDB storage path | `./chroma_db` |
| `MAX_PAPERS` | No | Max papers to fetch | `500` |
| `CHUNK_SIZE` | No | Text chunk size in tokens | `600` |
| `CHUNK_OVERLAP` | No | Overlap between chunks | `100` |

### Streamlit Cloud Settings

- **Python Version**: 3.8+
- **Main File**: `streamlit_app.py`
- **Requirements**: `requirements.txt`

## 🐛 Troubleshooting

### Common Issues

1. **Build Failures**:
   - Check that all dependencies are in `requirements.txt`
   - Verify Python version compatibility
   - Check Streamlit Cloud logs for specific errors

2. **API Key Issues**:
   - Ensure API keys are correctly set in Streamlit Cloud secrets
   - Verify API keys have sufficient quota
   - Check that keys are not expired

3. **Memory Issues**:
   - Reduce `MAX_PAPERS` if hitting memory limits
   - Consider using smaller chunk sizes
   - Monitor Streamlit Cloud resource usage

4. **Timeout Issues**:
   - Knowledge base building can take 10-15 minutes
   - Consider implementing progress indicators
   - Use smaller paper sets for testing

### Performance Optimization

1. **Reduce Initial Load Time**:
   - Start with fewer papers (e.g., 100 instead of 500)
   - Use smaller chunk sizes
   - Implement caching strategies

2. **Optimize API Usage**:
   - Monitor API quotas
   - Implement rate limiting
   - Cache responses when possible

## 📊 Monitoring

### Streamlit Cloud Dashboard

- Monitor app performance
- Check error logs
- View usage statistics

### API Dashboards

- **Google AI Studio**: Monitor Gemini API usage
- **OpenAI Platform**: Monitor OpenAI API usage

## 🔄 Updates and Maintenance

### Updating the App

1. **Make Changes**:
   - Update your local code
   - Test locally with `streamlit run app.py`

2. **Deploy Updates**:
   ```bash
   git add .
   git commit -m "Update description"
   git push origin main
   ```
   - Streamlit Cloud will automatically redeploy

### Regular Maintenance

1. **Monitor API Usage**:
   - Check quotas regularly
   - Update API keys if needed

2. **Update Dependencies**:
   - Keep packages up to date
   - Test compatibility before deploying

3. **Data Refresh**:
   - Consider rebuilding knowledge base periodically
   - Monitor for new papers in your domain

## 🚀 Advanced Deployment Options

### Custom Domain

- Streamlit Cloud supports custom domains
- Configure in your Streamlit Cloud dashboard

### Multiple Environments

- Create separate apps for development and production
- Use different API keys for each environment

### Scaling Considerations

- Monitor concurrent users
- Consider upgrading Streamlit Cloud plan if needed
- Implement caching for better performance

## 📞 Support

- **Streamlit Cloud**: [docs.streamlit.io](https://docs.streamlit.io/streamlit-community-cloud)
- **GitHub Issues**: Create issues in your repository
- **Community**: [Streamlit Community Forum](https://discuss.streamlit.io/)

## 🎉 Success!

Once deployed, your RAG system will be available at:
`https://your-app-name.streamlit.app`

Share the link with users and start answering questions about Large Language Models!

