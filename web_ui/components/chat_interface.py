import streamlit as st
from streamlit_chat import message
import warnings
warnings.filterwarnings("ignore")

def render_chat_interface():
    """Render the chat interface"""
    
    st.markdown("---")
    st.subheader("💬 Chat with Video")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, chat in enumerate(st.session_state.chat_history):
            if "question" in chat and "answer" in chat:
                message(chat["question"], is_user=True, key=f"user_{i}")
                message(chat["answer"], key=f"ai_{i}")
    
    # Chat input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        question = st.text_input(
            "Ask a question about the video:",
            placeholder="What was explained about...",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    if send_button and question:
        # Generate response
        with st.spinner("Thinking..."):
            answer = st.session_state.assistant.chat_with_video(question)
            
            # Store in chat history
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer
            })
            
            # Rerun to update display
            st.rerun()
    
    # Suggested questions
    st.markdown("### 💡 Suggested Questions")
    suggested_questions = [
        "What are the main topics covered?",
        "Can you summarize the key points?",
        "Explain the most important concept",
        "What examples were given?",
        "What are the practical applications?"
    ]
    
    cols = st.columns(len(suggested_questions))
    for col, question in zip(cols, suggested_questions):
        with col:
            if st.button(question, use_container_width=True):
                # Auto-fill the question
                st.session_state.last_suggested = question
                st.rerun()