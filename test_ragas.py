import os
import sys

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import ragas
    print(f"RAGAS is available: {ragas.__version__}")
except ImportError as e:
    print(f"RAGAS import error: {e}")

try:
    from datasets import Dataset
    print("Datasets package is available")
except ImportError as e:
    print(f"Datasets import error: {e}")

print("\nEnvironment variables:")
for key, value in os.environ.items():
    if key.startswith('PYTHON') or key.startswith('PATH'):
        print(f"{key}: {value}")