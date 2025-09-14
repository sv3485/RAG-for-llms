"""
Streamlit Cloud deployment entry point
This file is used specifically for Streamlit Community Cloud deployment
"""

import streamlit as st
import os

# Set page config for deployment
st.set_page_config(
    page_title="RAG System - Large Language Models Research",
    page_icon="🤖",
    layout="wide"
)

# Import and run the main app
from app import main

if __name__ == "__main__":
    main()

