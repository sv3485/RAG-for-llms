"""
Setup script for the RAG system
Helps users configure the system and run initial tests
"""

import os
import sys
import subprocess
from pathlib import Path

def create_env_file():
    """Create .env file from template"""
    env_file = Path('.env')
    env_example = Path('env_example.txt')
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        # Copy template to .env
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ Created .env file from template")
        print("⚠️ Please edit .env file and add your API keys")
        return True
    else:
        print("❌ env_example.txt not found")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_tests():
    """Run system tests"""
    print("Running system tests...")
    try:
        result = subprocess.run([sys.executable, 'test_system.py'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 RAG System Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('requirements.txt').exists():
        print("❌ requirements.txt not found. Please run this script from the RAG project directory.")
        return False
    
    # Create .env file
    if not create_env_file():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Run tests
    print("\n" + "=" * 50)
    if run_tests():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and add your API keys")
        print("2. Run: streamlit run app.py")
        print("3. Open your browser to http://localhost:8501")
    else:
        print("\n⚠️ Setup completed with some issues.")
        print("Please check the errors above and fix them before running the app.")
    
    return True

if __name__ == "__main__":
    main()

