import yt_dlp
import os
from datetime import datetime
import tempfile
import warnings
warnings.filterwarnings("ignore")
import re

class YouTubeProcessor:
    def __init__(self, output_dir="data/audio"):
        self.output_dir = output_dir
        self.processing_time = 0
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_video_id(self, youtube_url: str) -> str:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=)([^&]+)',
            r'(?:youtu\.be\/)([^?]+)',
            r'(?:youtube\.com\/embed\/)([^?]+)',
            r'(?:youtube\.com\/v\/)([^?]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        
        # If no pattern matches, use hash of URL
        import hashlib
        return hashlib.md5(youtube_url.encode()).hexdigest()[:12]
    
    def extract_audio(self, youtube_url, quality="best", force_download=False):
        """Extract audio from YouTube video"""
        
        video_id = self.extract_video_id(youtube_url)
        audio_filename = os.path.join(self.output_dir, f"{video_id}.mp3")
        
        # Check if audio already exists
        if os.path.exists(audio_filename) and not force_download:
            print(f"✓ Audio file already exists: {audio_filename}")
            self.processing_time = 0.1  # Minimal time for cache hit
            return audio_filename
        
        start_time = datetime.now()
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(self.output_dir, f'{video_id}.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                # Calculate processing time
                self.processing_time = (datetime.now() - start_time).total_seconds()
                
                return audio_filename
                
        except Exception as e:
            raise Exception(f"Failed to extract audio: {str(e)}")
    
    def get_video_info(self, youtube_url):
        """Get video metadata"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            
            return {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'channel': info.get('channel', 'Unknown'),
                'view_count': info.get('view_count', 0),
                'upload_date': info.get('upload_date', 'Unknown'),
                'description': info.get('description', ''),
                'video_id': self.extract_video_id(youtube_url)
            }