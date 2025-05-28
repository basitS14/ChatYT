# app.py
import streamlit as st
from helpers import (create_chunks, document_retriever,
                     get_transcript, create_vector_store, ConversationalYouTubeChat)
import sys

# Configure page
st.set_page_config(
    page_title="YouTube Chat Bot",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'chat_bot' not in st.session_state:
    st.session_state.chat_bot = None
if 'video_link' not in st.session_state:
    st.session_state.video_link = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def process_video(link):
    """Process the YouTube video and set up the chatbot"""
    try:
        # Get transcript
        transcript = get_transcript(link=link)
        if not transcript:
            return False, "Could not get transcript for this video."
        
        # Create chunks
        chunks = create_chunks(transcript=transcript)
        
        # Create vector store
        vector_store = create_vector_store(chunks, link=link)
        if not vector_store:
            return False, "Could not create vector store."
        
        # Create retriever
        retriever = document_retriever(vector_store=vector_store)
        if not retriever:
            return False, "Could not create document retriever."
        
        # Create chatbot
        chat_bot = ConversationalYouTubeChat(retriever, memory_window=5)
        
        return True, chat_bot
    
    except Exception as e:
        return False, f"Error processing video: {str(e)}"

def reset_session():
    """Reset the session to start over"""
    st.session_state.processed = False
    st.session_state.chat_bot = None
    st.session_state.video_link = ""
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.title("🎥 ChatYT")
    st.write("---")
    
    if st.button("🔄 New Conversation", use_container_width=True):
        reset_session()
        st.rerun()
    
    if st.session_state.processed:
        st.success("✅ Video processed successfully!")
        st.write(f"**Video:** {st.session_state.video_link[:50]}...")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.chat_bot:
                st.session_state.chat_bot.clear_history()
            st.rerun()

# Main content
if not st.session_state.processed:
    # Page 1: Video Input and Processing
    st.title("🎥 Conversational YouTube Chat Bot")
    st.write("#### Don't have time to watch the full video? No problem! Just chat with it and get the information you want.")
    st.write("---")
    
    # Video input form
    with st.form("video_form", clear_on_submit=True):
        video_link = st.text_input(
            "📹 Paste YouTube video link:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter a valid YouTube video URL"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_button = st.form_submit_button("🚀 Start Processing", use_container_width=True)
    
    if submit_button and video_link:
        if video_link.strip():
            st.session_state.video_link = video_link.strip()
            
            # Show processing status
            with st.spinner("🔄 Processing video... This may take a few moments."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate progress updates
                status_text.text("📹 Extracting transcript...")
                progress_bar.progress(25)
                
                success, result = process_video(st.session_state.video_link)
                
                if success:
                    status_text.text("📝 Creating chunks...")
                    progress_bar.progress(50)
                    
                    status_text.text("🔍 Building vector store...")
                    progress_bar.progress(75)
                    
                    status_text.text("🤖 Setting up chatbot...")
                    progress_bar.progress(100)
                    
                    # Store chatbot in session state
                    st.session_state.chat_bot = result
                    st.session_state.processed = True
                    
                    st.success("✅ Video processed successfully! Redirecting to chat...")
                    st.balloons()
                    
                    # Auto-rerun to switch to chat page
                    st.rerun()
                else:
                    st.error(f"❌ {result}")
                    progress_bar.empty()
                    status_text.empty()
        else:
            st.warning("⚠️ Please enter a valid YouTube video link.")
    
    elif submit_button and not video_link:
        st.warning("⚠️ Please enter a YouTube video link.")

else:
    # Page 2: Chat Interface
    st.title("🤖 Chat with Your Video")
    st.write(f"**Video:** {st.session_state.video_link}")
    st.write("---")
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        if st.session_state.chat_history:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                # User message
                with st.chat_message("user"):
                    st.write(question)
                
                # Bot response
                with st.chat_message("assistant"):
                    st.write(answer)
        else:
            st.info("💡 Start chatting by typing your question below! I can remember our conversation context.")
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask a question about the video:",
                placeholder="What is this video about?",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.form_submit_button("Send 📤", use_container_width=True)
    
    # Process chat input
    if send_button and user_input:
        if user_input.strip():
            question = user_input.strip()
            
            # Show thinking spinner
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get response from chatbot
                    response = st.session_state.chat_bot.chat(question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, response))
                    
                    # Rerun to update chat display
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question.")
    
    # Quick action buttons
    st.write("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📝 Summarize Video", use_container_width=True):
            with st.spinner("📝 Generating summary..."):
                try:
                    summary = st.session_state.chat_bot.chat("Please provide a comprehensive summary of this video using the transcript of the video")
                    st.session_state.chat_history.append(("Summarize Video", summary))
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error generating summary: {str(e)}")
    
    with col2:
        if st.button("🔑 Key Points", use_container_width=True):
            with st.spinner("🔑 Extracting key points..."):
                try:
                    key_points = st.session_state.chat_bot.chat("What are the main key points or takeaways from this video? check transcript for your refrence")
                    st.session_state.chat_history.append(("Key Points", key_points))
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error extracting key points: {str(e)}")
    
    with col3:
        if st.button("❓ Suggest Questions", use_container_width=True):
            with st.spinner("❓ Generating questions..."):
                try:
                    questions = st.session_state.chat_bot.chat("Suggest some interesting questions I could ask about this video content.")
                    st.session_state.chat_history.append(("Suggest Questions", questions))
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error generating questions: {str(e)}")

# Footer
st.write("---")
st.write("Made with ❤️ using Streamlit")