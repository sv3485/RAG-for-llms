#!/usr/bin/env python3

import chromadb
import os

def check_knowledge_base():
    print("Checking knowledge base...")
    
    if not os.path.exists("./chroma_db"):
        print("❌ chroma_db directory does not exist")
        return False
    
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        
        try:
            collection = client.get_collection("arxiv_papers")
            count = collection.count()
            print(f"✅ Collection exists with {count} documents")
            return count > 0
        except Exception as e:
            print(f"❌ Error getting collection: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to ChromaDB: {e}")
        return False

if __name__ == "__main__":
    check_knowledge_base()