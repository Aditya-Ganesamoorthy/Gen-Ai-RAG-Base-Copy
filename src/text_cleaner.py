import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import spacy
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def clean_transcript(self, text):
        """Clean and preprocess transcript text"""
        
        # Remove timestamps and special characters
        text = re.sub(r'\d+:\d+', '', text)  # Remove timestamps like 00:00
        text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
        text = re.sub(r'\(.*?\)', '', text)  # Remove text in parentheses
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Sentence segmentation using spaCy for better accuracy
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        
        # Join back with proper spacing
        cleaned_text = ' '.join(sentences)
        
        return cleaned_text
    
    def remove_fillers(self, text):
        """Remove common filler words"""
        filler_words = [
            'um', 'uh', 'ah', 'er', 'hm', 'like', 'you know',
            'i mean', 'actually', 'basically', 'literally'
        ]
        
        pattern = r'\b(' + '|'.join(filler_words) + r')\b'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def normalize_text(self, text):
        """Normalize text for consistency"""
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'ve": " have",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text