from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class TextChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def create_chunks(self, text):
        """Split text into chunks"""
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def create_semantic_chunks(self, text, similarity_threshold=0.85):
        """Create chunks based on semantic similarity"""
        
        # First create sentence-level chunks
        sentences = text.split('. ')
        
        # Load sentence transformer for semantic similarity
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentence_embeddings = model.encode(sentences)
        
        chunks = []
        current_chunk = []
        current_embedding = None
        
        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            if not current_chunk:
                current_chunk.append(sentence)
                current_embedding = embedding
            else:
                # Calculate cosine similarity
                similarity = self._cosine_similarity(current_embedding, embedding)
                
                if similarity > similarity_threshold and len(' '.join(current_chunk)) < self.chunk_size:
                    current_chunk.append(sentence)
                    # Update embedding as average
                    current_embedding = (current_embedding + embedding) / 2
                else:
                    chunks.append('. '.join(current_chunk) + '.')
                    current_chunk = [sentence]
                    current_embedding = embedding
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def create_topic_based_chunks(self, text, nlp):
        """Create chunks based on topic changes"""
        doc = nlp(text)
        
        chunks = []
        current_chunk = []
        current_topic = None
        
        for sent in doc.sents:
            # Simple topic detection (can be enhanced)
            sent_topic = self._extract_main_topic(str(sent))
            
            if current_topic is None:
                current_topic = sent_topic
            
            if sent_topic == current_topic or len(' '.join(current_chunk)) < 500:
                current_chunk.append(str(sent))
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [str(sent)]
                current_topic = sent_topic
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _extract_main_topic(self, sentence):
        """Extract main topic from sentence (simplified)"""
        # This can be enhanced with NER or keyword extraction
        important_words = ['introduction', 'conclusion', 'example', 'theorem',
                          'definition', 'algorithm', 'code', 'experiment']
        
        sentence_lower = sentence.lower()
        for word in important_words:
            if word in sentence_lower:
                return word
        
        return "general"