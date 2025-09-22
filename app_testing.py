import streamlit as st
from streamlit_chat import message  # Optional: pip install streamlit-chat

st.set_page_config(page_title="Chat Interface", layout="wide")

# Inject custom CSS for styling
st.markdown("""
    <style>
    .chat-input-container {
        display: flex;
        align-items: center;
        border: 1px solid #ccc;
        border-radius: 30px;
        padding: 0.4em 1em;
        margin-top: 1em;
        background-color: #f9f9f9;
    }
    .chat-input-container input[type='text'] {
        border: none;
        flex: 1;
        padding: 0.5em;
        font-size: 1em;
        outline: none;
        background: transparent;
    }
    .chat-icon-button {
        background: none;
        border: none;
        cursor: pointer;
        font-size: 1.2em;
        margin-left: 0.5em;
    }
    </style>
""", unsafe_allow_html=True)

# Display message history (optional)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    message(msg["content"], is_user=msg["is_user"])

# File uploader via "+"
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a file", label_visibility="collapsed")

# Custom chat input layout
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)

# Create a text_input inside a form so we can control submission
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("", placeholder="Ask anything...", label_visibility="collapsed")
    submit_button = st.form_submit_button("➤")

st.markdown('</div>', unsafe_allow_html=True)

# Handle send
if submit_button and user_input:
    st.session_state.chat_history.append({"content": user_input, "is_user": True})
    st.rerun()  # Rerun to show new message

# Display uploaded file name
if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
