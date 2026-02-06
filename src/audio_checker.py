import whisper
from transformers import pipeline
import numpy as np
from pydub import AudioSegment
import tempfile
import warnings
warnings.filterwarnings("ignore")

class AudioChecker:
    def __init__(self):
        # Load a small Whisper model for quick checking
        self.model = whisper.load_model("tiny")
        
        # Load text classification model
        self.classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=True
        )
    
    def is_educational(self, audio_path, threshold=0.6):
        """Check if audio content is educational"""
        
        # Transcribe first 60 seconds for quick check
        audio = AudioSegment.from_mp3(audio_path)
        first_minute = audio[:60000]  # First 60 seconds
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            first_minute.export(tmp.name, format="mp3")
            
            # Transcribe
            result = self.model.transcribe(tmp.name)
            transcription = result["text"]
        
        # Check for educational keywords
        educational_keywords = [
            'lecture', 'tutorial', 'explain', 'theory', 'concept',
            'learning', 'study', 'education', 'course', 'class',
            'mathematics', 'science', 'history', 'physics', 'chemistry',
            'biology', 'programming', 'computer', 'algorithm', 'data',
            'analysis', 'research', 'method', 'process', 'system'
        ]
        
        # Simple keyword matching
        transcript_lower = transcription.lower()
        keyword_matches = sum(1 for keyword in educational_keywords 
                            if keyword in transcript_lower)
        
        # Use ML classifier
        classification = self.classifier(transcription[:512])
        educational_score = 0
        
        for item in classification[0]:
            if item['label'] == 'POSITIVE':
                educational_score = item['score']
                break
        
        # Combined decision
        is_educational = (educational_score > threshold) or (keyword_matches >= 3)
        
        return is_educational