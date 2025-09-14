import sys
import importlib

# Remove ragas and datasets from sys.modules if they exist
if 'ragas' in sys.modules:
    del sys.modules['ragas']
    print("Removed ragas from sys.modules")
else:
    print("ragas not in sys.modules")

if 'datasets' in sys.modules:
    del sys.modules['datasets']
    print("Removed datasets from sys.modules")
else:
    print("datasets not in sys.modules")

# Try to import them again
try:
    import ragas
    print(f"Successfully imported ragas version {ragas.__version__}")
except ImportError as e:
    print(f"Failed to import ragas: {e}")

try:
    from datasets import Dataset
    print("Successfully imported datasets")
except ImportError as e:
    print(f"Failed to import datasets: {e}")

print("\nModule paths:")
if 'ragas' in sys.modules:
    print(f"ragas: {sys.modules['ragas'].__file__}")
if 'datasets' in sys.modules:
    print(f"datasets: {sys.modules['datasets'].__file__}")