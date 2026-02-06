import streamlit as st
import os
import time
from dotenv import load_dotenv
from src.youtube_processor import YouTubeProcessor
from src.audio_checker import AudioChecker
from src.stt_processor import STTProcessor
from src.text_cleaner import TextCleaner
from src.chunking import TextChunker
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.llm_handler import LLMHandler
from src.features import VideoLearningAssistant
from src.cache_manager import CacheManager
from web_ui.components.sidebar import render_sidebar
from web_ui.components.chat_interface import render_chat_interface
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Video Learning Assistant",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'current_video_url' not in st.session_state:
    st.session_state.current_video_url = None
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'video_info' not in st.session_state:
    st.session_state.video_info = None
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'whisper_model': 'base',
        'chunk_size': 800,
        'chunk_overlap': 100,
        'llm_model': 'gemini-pro',
        'temperature': 0.3,
        'embedding_model': 'all-MiniLM-L6-v2',
        'force_reprocess': False
    }

def main():
    """Main application"""
    
    # Sidebar
    render_sidebar()
    
    # Main content
    st.title("🎓 Multimodal RAG-based AI Video Learning Assistant")
    st.markdown("---")
    
    # URL Input
    col1, col2 = st.columns([3, 1])
    with col1:
        youtube_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            key="youtube_url_input"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        process_button = st.button("Process Video", type="primary", key="process_button", use_container_width=True)
    
    if process_button and youtube_url:
        if youtube_url == st.session_state.get('current_video_url') and st.session_state.get('assistant'):
            st.info("✅ This video is already processed. You can use the features below.")
        else:
            process_video_with_cache(youtube_url)
    
    # Show current video info if available
    if st.session_state.get('video_info'):
        display_video_info()
    
    # Features section
    if st.session_state.assistant:
        display_features()
    
    # Chat interface
    if st.session_state.assistant:
        render_chat_interface()

def process_video_with_cache(youtube_url):
    """Process video with caching mechanism"""
    
    # Initialize cache manager
    cache_manager = CacheManager()
    
    # Get current settings
    settings = st.session_state.settings
    
    # Generate cache signature based on settings
    cache_signature = f"{settings['whisper_model']}_{settings['chunk_size']}_{settings['embedding_model']}"
    
    # Check if video is already cached with same settings
    if cache_manager.is_cached(youtube_url) and not settings['force_reprocess']:
        # Check if cached with same settings
        cache_info = cache_manager.get_cache_info(youtube_url)
        if cache_info and cache_info.get('settings_signature') == cache_signature:
            st.info("📁 Loading from cache...")
            load_from_cache(youtube_url, cache_manager)
            return
        else:
            st.info("⚙️ Settings changed, reprocessing with new settings...")
    
    # Process from scratch
    process_video_from_scratch(youtube_url, cache_manager, settings)

def load_from_cache(youtube_url, cache_manager):
    """Load video data from cache"""
    
    with st.spinner("Loading cached data..."):
        try:
            # Load cached data
            transcript = cache_manager.load_from_cache(youtube_url, 'transcript')
            cleaned_text = cache_manager.load_from_cache(youtube_url, 'cleaned_text')
            chunks = cache_manager.load_from_cache(youtube_url, 'chunks')
            embeddings = cache_manager.load_from_cache(youtube_url, 'embeddings')
            metadata = cache_manager.load_from_cache(youtube_url, 'metadata')
            
            if not all([transcript, cleaned_text, chunks, embeddings]):
                raise ValueError("Incomplete cache data")
            
            # Get video info
            youtube_processor = YouTubeProcessor()
            video_info = youtube_processor.get_video_info(youtube_url)
            video_id = video_info.get('video_id', 'unknown')
            
            # Initialize vector store
            vector_store = VectorStore()
            
            # Create or get collection
            collection_name = f"video_{video_id}"
            db = vector_store.create_vector_store(chunks, embeddings, collection_name=collection_name)
            
            # Initialize LLM handler with current settings
            llm_handler = LLMHandler(
                model_name=st.session_state.settings['llm_model'],
                temperature=st.session_state.settings['temperature']
            )
            
            # Initialize assistant
            assistant = VideoLearningAssistant(
                vector_db=vector_store,
                llm_handler=llm_handler,
                transcript=cleaned_text,
                video_id=video_id
            )
            
            # Update session state
            st.session_state.assistant = assistant
            st.session_state.transcript = cleaned_text
            st.session_state.vector_db = vector_store
            st.session_state.current_video_url = youtube_url
            st.session_state.video_info = video_info
            
            st.success("✅ Video loaded from cache successfully!")
            
        except Exception as e:
            st.error(f"Error loading from cache: {str(e)}")
            st.info("Processing from scratch...")
            process_video_from_scratch(youtube_url, cache_manager, st.session_state.settings)

def process_video_from_scratch(youtube_url, cache_manager, settings):
    """Process YouTube video from scratch"""
    
    st.session_state.is_processing = True
    start_time = time.time()
    
    try:
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            # Step 1: Get video info
            st.info("📋 Getting video information...")
            youtube_processor = YouTubeProcessor()
            video_info = youtube_processor.get_video_info(youtube_url)
            video_id = video_info.get('video_id', 'unknown')
            
            # Display estimated time
            estimated_time = estimate_processing_time(
                video_info.get('duration', 0),
                settings['whisper_model'],
                settings['chunk_size']
            )
            st.info(f"⏱️ Estimated processing time: {estimated_time}")
            
            # Step 2: Extract audio
            st.info("🎵 Step 1/6: Extracting audio from YouTube...")
            audio_path = youtube_processor.extract_audio(youtube_url)
            
            # Step 3: Check content type (optional, can skip for speed)
            st.info("🔍 Step 2/6: Checking content type...")
            audio_checker = AudioChecker()
            is_educational = audio_checker.is_educational(audio_path)
            
            if not is_educational:
                st.warning("⚠️ This may not be highly educational content. Processing anyway...")
            
            # Step 4: Transcribe audio
            st.info("🗣️ Step 3/6: Transcribing audio (this may take a while)...")
            stt_processor = STTProcessor(model_size=settings['whisper_model'])
            transcript = stt_processor.transcribe(audio_path)
            
            if not transcript or len(transcript.strip()) < 50:
                st.error("Transcription failed or produced very short text. The video might have no speech.")
                return
            
            # Step 5: Clean and chunk text
            st.info("🧹 Step 4/6: Cleaning and chunking text...")
            text_cleaner = TextCleaner()
            cleaned_text = text_cleaner.clean_transcript(transcript)
            
            chunker = TextChunker(
                chunk_size=settings['chunk_size'],
                chunk_overlap=settings['chunk_overlap']
            )
            chunks = chunker.create_chunks(cleaned_text)
            
            # Step 6: Create embeddings
            st.info("🔢 Step 5/6: Creating embeddings...")
            embedding_gen = EmbeddingGenerator(model_name=settings['embedding_model'])
            embeddings = embedding_gen.generate_embeddings(chunks)
            
            # Step 7: Create vector store
            st.info("🗄️ Step 6/6: Setting up vector database and assistant...")
            vector_store = VectorStore()
            
            # Use video_id for collection name
            collection_name = f"video_{video_id}"
            db = vector_store.create_vector_store(chunks, embeddings, collection_name=collection_name)
            
            # Initialize LLM handler
            llm_handler = LLMHandler(
                model_name=settings['llm_model'],
                temperature=settings['temperature']
            )
            
            # Initialize assistant
            assistant = VideoLearningAssistant(
                vector_db=vector_store,
                llm_handler=llm_handler,
                transcript=cleaned_text,
                video_id=video_id
            )
            
            # Cache all data
            cache_signature = f"{settings['whisper_model']}_{settings['chunk_size']}_{settings['embedding_model']}"
            metadata = {
                'video_id': video_id,
                'title': video_info.get('title', 'Unknown'),
                'duration': video_info.get('duration', 0),
                'processed_at': time.time(),
                'chunk_count': len(chunks),
                'word_count': len(cleaned_text.split()),
                'settings_signature': cache_signature,
                'whisper_model': settings['whisper_model'],
                'chunk_size': settings['chunk_size'],
                'embedding_model': settings['embedding_model']
            }
            
            cache_manager.save_to_cache(youtube_url, 'transcript', transcript)
            cache_manager.save_to_cache(youtube_url, 'cleaned_text', cleaned_text)
            cache_manager.save_to_cache(youtube_url, 'chunks', chunks)
            cache_manager.save_to_cache(youtube_url, 'embeddings', embeddings)
            cache_manager.save_to_cache(youtube_url, 'metadata', metadata)
            
            # Update session state
            st.session_state.assistant = assistant
            st.session_state.transcript = cleaned_text
            st.session_state.vector_db = vector_store
            st.session_state.current_video_url = youtube_url
            st.session_state.video_info = video_info
            
            processing_time = time.time() - start_time
            
            st.success(f"✅ Video processed successfully in {processing_time:.1f} seconds!")
            
            # Store processing stats
            st.session_state.processing_stats = {
                'time': processing_time,
                'chunks': len(chunks),
                'words': len(cleaned_text.split()),
                'whisper_model': settings['whisper_model']
            }
            
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        import traceback
        st.error(f"Error details: {traceback.format_exc()}")
        
    finally:
        st.session_state.is_processing = False
        st.rerun()

def estimate_processing_time(duration_seconds, whisper_model, chunk_size):
    """Estimate processing time based on settings"""
    if not duration_seconds or duration_seconds == 0:
        return "Unknown"
    
    duration_minutes = duration_seconds / 60
    
    # Base transcription time (minutes per minute of audio)
    whisper_speeds = {
        'tiny': 0.15,    # 0.15 minutes per minute (9 seconds per minute)
        'base': 0.25,    # 0.25 minutes per minute (15 seconds per minute)
        'small': 0.4,    # 0.4 minutes per minute (24 seconds per minute)
        'medium': 0.7,   # 0.7 minutes per minute (42 seconds per minute)
        'large': 1.2     # 1.2 minutes per minute (72 seconds per minute)
    }
    
    # Estimated transcription time
    trans_time = duration_minutes * whisper_speeds.get(whisper_model, 0.25)
    
    # Estimated embedding time (depends on chunk count)
    approx_words = duration_minutes * 150  # 150 WPM
    approx_chars = approx_words * 5
    num_chunks = max(1, approx_chars / chunk_size)
    embed_time = num_chunks * 0.05 / 60  # 0.05 seconds per chunk
    
    # Other processing time
    other_time = 1.0  # 1 minute for other steps
    
    total_minutes = trans_time + embed_time + other_time
    
    if total_minutes < 1:
        return f"{int(total_minutes*60)} seconds"
    elif total_minutes < 60:
        return f"{int(total_minutes)} minutes"
    else:
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        return f"{hours} hour{'' if hours == 1 else 's'} {minutes} minutes"

def display_video_info():
    """Display video information"""
    
    video_info = st.session_state.video_info
    if not video_info:
        return
    
    st.subheader("📊 Video Information")
    
    # Create columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Display title (truncated if too long)
        title = video_info.get('title', 'Unknown Title')
        if len(title) > 40:
            title = title[:37] + "..."
        st.metric("Title", title)
    
    with col2:
        # Display duration
        duration = video_info.get('duration', 0)
        if duration:
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            seconds = duration % 60
            
            if hours > 0:
                duration_str = f"{hours}h {minutes}m"
            elif minutes > 0:
                duration_str = f"{minutes}m {seconds}s"
            else:
                duration_str = f"{seconds}s"
            st.metric("Duration", duration_str)
        else:
            st.metric("Duration", "Unknown")
    
    with col3:
        # Display channel
        channel = video_info.get('channel', 'Unknown Channel')
        if len(channel) > 25:
            channel = channel[:22] + "..."
        st.metric("Channel", channel)
    
    # Second row of metrics
    col4, col5, col6 = st.columns(3)
    
    with col4:
        # Display views
        views = video_info.get('view_count', 0)
        if views:
            if views >= 1000000:
                views_str = f"{views/1000000:.1f}M"
            elif views >= 1000:
                views_str = f"{views/1000:.1f}K"
            else:
                views_str = str(views)
            st.metric("Views", views_str)
        else:
            st.metric("Views", "Unknown")
    
    with col5:
        # Display processing stats if available
        if hasattr(st.session_state, 'processing_stats'):
            stats = st.session_state.processing_stats
            st.metric("Processing Time", f"{stats['time']:.1f}s")
        else:
            st.metric("Status", "Cached")
    
    with col6:
        # Display chunk count
        if hasattr(st.session_state, 'processing_stats'):
            stats = st.session_state.processing_stats
            st.metric("Chunks", stats['chunks'])
        elif st.session_state.assistant:
            st.metric("Status", "Ready")

def display_features():
    """Display available features"""
    
    if not st.session_state.assistant:
        return
    
    st.markdown("---")
    st.subheader("✨ Available Features")
    
    # Create feature buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📚 Topic-wise Summary", 
                    use_container_width=True, 
                    key="summary_btn",
                    help="Generate a comprehensive topic-wise summary of the video"):
            with st.spinner("Generating summary... This may take a moment."):
                try:
                    summary = st.session_state.assistant.generate_topic_summary()
                    
                    # Display summary in an expandable section
                    with st.expander("📖 Topic-wise Summary", expanded=True):
                        st.markdown(summary)
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
    
    with col2:
        if st.button("🧠 Generate Quiz", 
                    use_container_width=True, 
                    key="quiz_btn",
                    help="Create a quiz based on video content"):
            with st.spinner("Creating quiz... This may take a moment."):
                try:
                    quiz = st.session_state.assistant.generate_quiz()
                    
                    # Display quiz
                    st.subheader("📝 Generated Quiz")
                    
                    if isinstance(quiz, dict) and 'questions' in quiz:
                        st.success(f"**{quiz.get('quiz_title', 'Video Content Quiz')}**")
                        
                        for i, q in enumerate(quiz['questions'], 1):
                            st.markdown(f"**Q{i}: {q.get('question', 'No question text')}**")
                            
                            # Display options
                            options = q.get('options', [])
                            if isinstance(options, list):
                                for opt_idx, opt in enumerate(options):
                                    if isinstance(opt, str) and opt.strip():
                                        st.write(f"  {chr(65+opt_idx)}. {opt}")
                            
                            # Show answer with expander
                            with st.expander(f"Show Answer for Q{i}"):
                                correct_answer = q.get('correct_answer', 'N/A')
                                st.success(f"**Correct Answer: {correct_answer}**")
                                
                                explanation = q.get('explanation', '')
                                if explanation:
                                    st.info(f"**Explanation:** {explanation}")
                            
                            st.markdown("---")
                    else:
                        st.warning("Could not generate quiz in proper format. Here's what we got:")
                        st.write(quiz)
                        
                except Exception as e:
                    st.error(f"Error generating quiz: {str(e)}")
    
    with col3:
        if st.button("📈 Learning Analytics", 
                    use_container_width=True, 
                    key="analytics_btn",
                    help="View learning analytics and recommendations"):
            with st.spinner("Analyzing... This may take a moment."):
                try:
                    analytics = st.session_state.assistant.get_learning_analytics()
                    
                    # Display analytics in a nice format
                    st.subheader("📊 Learning Analytics")
                    
                    # Video Analytics
                    st.markdown("### Video Analytics")
                    video_stats = analytics.get('video_analytics', {})
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Words", video_stats.get('transcript_length_words', 0))
                    with col_b:
                        st.metric("Topics", video_stats.get('topics_covered', 0))
                    with col_c:
                        duration = video_stats.get('estimated_duration_minutes', 0)
                        st.metric("Est. Duration", f"{duration:.0f} min")
                    
                    # Topics covered
                    topics = video_stats.get('main_topics', [])
                    if topics:
                        st.markdown("**Main Topics:**")
                        for topic in topics[:5]:
                            st.write(f"• {topic}")
                    
                    # Learning Analytics
                    st.markdown("### Learning Analytics")
                    learning_stats = analytics.get('learning_analytics', {})
                    
                    col_d, col_e = st.columns(2)
                    with col_d:
                        st.metric("Chat Interactions", learning_stats.get('total_chat_interactions', 0))
                    with col_e:
                        st.metric("Quiz Ready", "✅" if learning_stats.get('quiz_ready') else "❌")
                    
                    # Recommendations
                    recommendations = analytics.get('recommendations', [])
                    if recommendations:
                        st.markdown("### 💡 Recommendations")
                        for rec in recommendations:
                            st.write(f"• {rec}")
                    
                except Exception as e:
                    st.error(f"Error generating analytics: {str(e)}")
    
    with col4:
        if st.button("🗑️ Clear Cache", 
                    use_container_width=True, 
                    key="clear_cache_btn",
                    help="Clear cache for this video (will need to reprocess)"):
            try:
                from src.cache_manager import CacheManager
                cache_manager = CacheManager()
                cache_manager.clear_cache(st.session_state.current_video_url)
                
                # Clear vector store collection
                if st.session_state.video_info:
                    video_id = st.session_state.video_info.get('video_id')
                    if video_id:
                        vector_store = VectorStore()
                        collection_name = f"video_{video_id}"
                        vector_store.delete_collection(collection_name)
                
                st.success("✅ Cache cleared for this video!")
                st.info("⚠️ You'll need to reprocess the video to use it again.")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing cache: {str(e)}")

if __name__ == "__main__":
    main()