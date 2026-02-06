"""
Cache Manager for storing and retrieving processed video data
"""
import os
import json
import hashlib
import pickle
from datetime import datetime
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

class CacheManager:
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_video_hash(self, youtube_url: str) -> str:
        """Generate unique hash for YouTube URL"""
        return hashlib.md5(youtube_url.encode()).hexdigest()
    
    def get_cache_path(self, youtube_url: str, data_type: str) -> str:
        """Get cache file path for specific data type"""
        video_hash = self.get_video_hash(youtube_url)
        return os.path.join(self.cache_dir, f"{video_hash}_{data_type}.pkl")
    
    def is_cached(self, youtube_url: str) -> bool:
        """Check if video is already cached"""
        video_hash = self.get_video_hash(youtube_url)
        
        # Check for essential cache files
        required_files = [
            f"{video_hash}_metadata.json",
            f"{video_hash}_transcript.pkl",
            f"{video_hash}_chunks.pkl",
            f"{video_hash}_embeddings.npy"
        ]
        
        for file in required_files:
            if not os.path.exists(os.path.join(self.cache_dir, file)):
                return False
        return True
    
    def save_to_cache(self, youtube_url: str, data_type: str, data: Any):
        """Save data to cache"""
        cache_file = self.get_cache_path(youtube_url, data_type)
        
        # Create cache directory if not exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # Save based on file type
        if data_type in ['transcript', 'chunks', 'metadata']:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        elif data_type == 'embeddings':
            import numpy as np
            np.save(cache_file, data)
        elif data_type == 'cleaned_text':
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(data)
        
        print(f"✓ Cached {data_type} for video")
    
    def load_from_cache(self, youtube_url: str, data_type: str) -> Any:
        """Load data from cache"""
        cache_file = self.get_cache_path(youtube_url, data_type)
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            if data_type in ['transcript', 'chunks', 'metadata']:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            elif data_type == 'embeddings':
                import numpy as np
                return np.load(cache_file, allow_pickle=True)
            elif data_type == 'cleaned_text':
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    
    def get_cache_info(self, youtube_url: str) -> Optional[Dict]:
        """Get cache metadata"""
        metadata_file = self.get_cache_path(youtube_url, 'metadata')
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return None
    
    def clear_cache(self, youtube_url: str = None):
        """Clear cache for specific video or all"""
        if youtube_url:
            video_hash = self.get_video_hash(youtube_url)
            pattern = f"{video_hash}_*"
            import glob
            for file in glob.glob(os.path.join(self.cache_dir, pattern)):
                os.remove(file)
            print(f"✓ Cleared cache for video")
        else:
            # Clear all cache
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
            print("✓ Cleared all cache")