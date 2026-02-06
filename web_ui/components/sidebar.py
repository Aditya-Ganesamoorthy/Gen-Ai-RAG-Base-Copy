import streamlit as st
import os
import warnings
warnings.filterwarnings("ignore")

def render_sidebar():
    """Render the sidebar with project information and controls"""
    
    with st.sidebar:
        st.title("🎓 Video Learning Assistant")
        st.markdown("---")
        
        # Project Info
        st.subheader("📋 Project Info")
        st.markdown("""
        **Multimodal RAG-based AI Video Learning Assistant**
        
        - Topic-wise understanding
        - Interactive chat with video
        - Personalized summaries
        - Quiz generation
        - Learning analytics
        """)
        
        st.markdown("---")
        
        # Performance Settings (affects processing speed)
        st.subheader("⚡ Performance Settings")
        
        # Whisper Model Selection - AFFECTS TRANSCRIPTION SPEED
        whisper_model = st.selectbox(
            "Whisper Model Size",
            ["tiny", "base", "small", "medium", "large"],
            index=1,  # default is "base"
            help="""
            🤔 **What this does:** Controls speech-to-text accuracy & speed
            • tiny: Fastest (3x faster), 40% less accurate
            • base: Balanced (default), good accuracy
            • large: Slowest, most accurate
            
            ⚡ **For 3-hour video:** Use 'tiny' or 'base' for faster processing
            """
        )
        
        # LLM Model Selection - AFFECTS RESPONSE GENERATION
        llm_model = st.selectbox(
            "LLM Model",
            ["gemini-pro", "gpt-3.5-turbo", "gpt-4", "local-model"],
            index=0,
            help="""
            🤔 **What this does:** Controls chat/summary/quiz quality
            • gemini-pro: Fast, good quality (free tier available)
            • gpt-3.5-turbo: Good balance
            • gpt-4: Best quality but slower
            • local-model: Offline but limited
            
            ⚠️ **Note:** Changing this requires re-processing if cache exists
            """
        )
        
        # Chunk Size - AFFECTS EMBEDDING GENERATION
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=200,
            max_value=2000,
            value=800,  # Reduced for faster processing
            step=100,
            help="""
            🤔 **What this does:** Controls text segmentation size
            • Smaller chunks: More embeddings, better search, slower processing
            • Larger chunks: Fewer embeddings, faster processing
            
            ⚡ **For 3-hour video:** Use 800-1200 for balance
            """
        )
        
        # Temperature - AFFECTS RESPONSE CREATIVITY
        temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="""
            🤔 **What this does:** Controls response randomness
            • 0.0: Factual, deterministic (good for summaries)
            • 0.3: Balanced (default)
            • 0.7-1.0: Creative, varied responses
            
            ⚡ **No effect on processing speed**
            """
        )
        
        # Advanced Settings
        with st.expander("⚙️ Advanced Settings"):
            # Overlap Size
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=200,
                value=100,
                step=10,
                help="Helps maintain context between chunks"
            )
            
            # Force Reprocess
            force_reprocess = st.checkbox(
                "Force Reprocess (ignore cache)",
                value=False,
                help="Always process video from scratch"
            )
            
            # Embedding Model
            embedding_model = st.selectbox(
                "Embedding Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L3-v2"],
                index=0,
                help="Smaller models are faster"
            )
        
        st.markdown("---")
        
        # Cache Management
        st.subheader("🗃️ Cache")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Cache", use_container_width=True):
                from src.cache_manager import CacheManager
                cache_manager = CacheManager()
                cache_manager.clear_all()
                st.success("Cache cleared!")
                st.rerun()
        
        with col2:
            if st.button("View Cache Stats", use_container_width=True):
                from src.cache_manager import CacheManager
                cache_manager = CacheManager()
                collections = cache_manager.list_collections()
                st.info(f"**Cached Videos:** {len(collections)}")
                for col in collections:
                    st.write(f"- {col}")
        
        st.markdown("---")
        
        # Session Management
        st.subheader("🔄 Session")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.success("Chat cleared!")
                st.rerun()
        
        with col2:
            if st.button("Reset All", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("Session reset!")
                st.rerun()
        
        # Status
        st.markdown("---")
        st.subheader("📊 Status")
        
        if st.session_state.get('assistant'):
            st.success("✅ Assistant Ready")
            st.metric("Chats", len(st.session_state.chat_history))
            if st.session_state.get('current_video_url'):
                # Show if cached
                from src.cache_manager import CacheManager
                cache_manager = CacheManager()
                if cache_manager.is_cached(st.session_state.current_video_url):
                    st.info("📁 Loaded from cache")
        else:
            st.info("🔍 Waiting for video...")