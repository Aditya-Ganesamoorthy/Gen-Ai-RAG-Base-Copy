import chromadb
import numpy as np
import os
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

class VectorStore:
    def __init__(self, persist_directory="data/vector_db"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with new API
        try:
            # For ChromaDB >= 0.4.0
            self.client = chromadb.PersistentClient(path=persist_directory)
            print("✓ Using ChromaDB PersistentClient")
        except Exception as e:
            print(f"Error with PersistentClient: {e}. Trying old API...")
            try:
                from chromadb.config import Settings
                self.client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=persist_directory
                ))
                print("✓ Using ChromaDB Client (old API)")
            except Exception as e2:
                print(f"Error initializing ChromaDB: {e2}")
                raise
    
    def create_vector_store(self, chunks: List[str], embeddings: np.ndarray, 
                           metadata: List[Dict] = None, collection_name: str = "video_transcripts") -> chromadb.Collection:
        """Create and store embeddings in vector database"""
        
        # Try to delete existing collection if it exists
        try:
            self.client.delete_collection(name=collection_name)
            print(f"✓ Deleted existing collection: {collection_name}")
        except:
            pass  # Collection doesn't exist
        
        # Create new collection
        try:
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "description": "Video transcript chunks"}
            )
            print(f"✓ Created new collection: {collection_name}")
        except Exception as e:
            print(f"Error creating collection: {e}. Trying get_or_create...")
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        # Prepare IDs
        ids = [f"chunk_{i:04d}" for i in range(len(chunks))]
        
        # Prepare metadata if not provided
        if metadata is None:
            metadata = [{"chunk_index": i, "text_length": len(chunk), "word_count": len(chunk.split())} 
                       for i, chunk in enumerate(chunks)]
        
        # Convert embeddings to list if numpy array
        if isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = embeddings
        
        # Add to collection in batches to handle large collections
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            
            collection.add(
                embeddings=embeddings_list[i:end_idx],
                documents=chunks[i:end_idx],
                metadatas=metadata[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        print(f"✓ Added {len(chunks)} chunks to collection '{collection_name}'")
        return collection
    
    def search_similar(self, query: str, collection_name: str = "video_transcripts", k: int = 3) -> List[Dict]:
        """Search for similar chunks"""
        
        try:
            # Get collection
            collection = self.client.get_collection(name=collection_name)
            
            # Query the collection
            results = collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and i < len(results['metadatas'][0]) else {},
                        'distance': results['distances'][0][i] if results['distances'] and i < len(results['distances'][0]) else 0.0
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
    
    def get_collection(self, collection_name: str = "video_transcripts"):
        """Get a collection by name"""
        try:
            return self.client.get_collection(name=collection_name)
        except:
            return None
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        try:
            self.client.get_collection(name=collection_name)
            return True
        except:
            return False
    
    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        try:
            self.client.delete_collection(name=collection_name)
            print(f"✓ Deleted collection: {collection_name}")
            return True
        except Exception as e:
            print(f"Collection {collection_name} not found or error: {e}")
            return False
    
    def list_collections(self):
        """List all collections"""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except:
            return []
    
    def count_chunks(self, collection_name: str = "video_transcripts") -> int:
        """Count number of chunks in collection"""
        try:
            collection = self.client.get_collection(name=collection_name)
            return collection.count()
        except:
            return 0
    
    def clear_all(self):
        """Clear all collections"""
        collections = self.list_collections()
        for col_name in collections:
            self.delete_collection(col_name)
        print(f"✓ Cleared all {len(collections)} collections")