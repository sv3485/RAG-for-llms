import streamlit as st

st.title("Package Import Check")

try:
    import ragas
    st.success(f"RAGAS is available: {ragas.__version__}")
except ImportError as e:
    st.error(f"RAGAS import error: {e}")

try:
    from datasets import Dataset
    st.success("Datasets package is available")
except ImportError as e:
    st.error(f"Datasets import error: {e}")

st.write("Check complete!")