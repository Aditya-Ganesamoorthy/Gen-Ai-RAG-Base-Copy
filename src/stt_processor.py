import whisper
import torch
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

class STTProcessor:
    def __init__(self, model_size="base"):
        """
        Initialize Whisper model for speech-to-text
        Available models: tiny, base, small, medium, large
        Use 'tiny' or 'base' for faster processing
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model ({model_size}) on {self.device}...")
        
        # Load with optimization
        self.model = whisper.load_model(
            model_size, 
            device=self.device,
            download_root="models/whisper"  # Cache models
        )
        
    def transcribe(self, audio_path, language=None):
        """Transcribe audio file to text with optimization"""
        
        start_time = datetime.now()
        
        try:
            # Load and transcribe audio with optimized settings
            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=(self.device == "cuda"),
                verbose=False,
                task="transcribe",
                # Faster decoding options
                beam_size=1,  # Reduced from default 5
                best_of=1,    # Reduced from default 5
                temperature=0.0,  # Deterministic output
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False  # Faster but slightly less accurate
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"Transcription completed in {processing_time:.2f} seconds")
            
            return result["text"]
            
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")