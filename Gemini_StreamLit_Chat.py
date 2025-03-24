import streamlit as st
import os
import re
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
import fitz  # PyMuPDF
from io import BytesIO
import tempfile
import base64

# Import Gemini client libraries.
from google import genai
from google.genai import types

# ----------------------------------------------------------------
# Configuration and settings

# Your Gemini API key (set this in your Streamlit secrets)
API_KEY = st.secrets["api_key"]
SECRET_KEY = st.secrets["secret_key"]


#For quickly running local put your API keys and comment out the above area
#API_KEY = "KEY"
#SECRET_KEY = "SECRET"

#.env
# API_KEY = os.environ.get("api_key")
# SECRET_KEY = os.environ.get("secret_key")

SYSTEM_PROMPT = ""

#Requires the secret key as part of the Parameters to access the app
requireKey = False

# Feature flags
disable_youtube = False
disable_scraping = False
disable_fileUpload = False

# This will upload and attach instead of processing for PDF/Txt Files
only_attach = False

# Avatars and images (customize as needed)
userAvatar = None
aiAvatar = None

#Images chosen for the sidebar
side_bar_image = "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExOW1wNnZlNTl4Y2Vlazkzd2Mzd2QwY2JvcnN0cHBxZjllN3ZvdzFpZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/XXsmHq395d7XNBDoP6/giphy.gif"
error_image = "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExcGlqcGxmaHE2ajM3YnBrMGV0dDdwbTF6NXd5aWM2MXJzMWZubWpqayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/273P92MBOqLiU/giphy.gif"

# ----------------------------------------------------------------
# Utility functions (URL transcription, scraping, and file reading, System Prompt)
def get_transcript_from_url(url):
    if "youtube.com" in url or "youtu.be" in url:
        return get_youtube_transcript(url)
    else:
        return scrape_website(url)

def get_youtube_transcript(url):
    if disable_youtube:
        return "Youtube Processing is Disabled"
    video_id = extract_youtube_video_id(url)
    if not video_id:
        return "Invalid YouTube URL"
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([entry['text'] for entry in transcript_list])
        return transcript
    except Exception as e:
        print(f"Error retrieving YouTube transcript: {e}")
        return f"Error retrieving YouTube transcript: {e}"

def extract_youtube_video_id(url):
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.match(pattern, url)
    return match.group(1) if match else None

def scrape_website(url):
    if disable_scraping:
        return "Scraping is Disabled"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join(soup.stripped_strings)
        return text
    except Exception as e:
        return f"❌Error scraping website: {e}"

def read_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"❌Error reading PDF file: {e}"

def read_txt(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        return f"❌Error reading TXT file: {e}"



#Builds the contents for Gemini to process the conversation history
def build_contents():
    contents = []
    for entry in st.session_state.conversation:
        parts = []
        # If there are file attachments, add them as from_uri parts.
        if "files" in entry:
            for f in entry["files"]:
                parts.append(
                    types.Part.from_uri(
                        file_uri=f.uri,
                        mime_type=f.mime_type,
                    )
                )
        # If there is an inline image stored as a base64 string, decode it.
        if "image" in entry:
            parts.append(
                types.Part.from_bytes(
                    mime_type="image/png",
                    data=base64.b64decode(entry["image"])
                )
            )
        # Always add the text part.
        parts.append(types.Part.from_text(text=entry["text"]))
        contents.append(types.Content(role=entry["role"], parts=parts))
        #output conversation for debugging
        #save_text_as_file(str(contents))
        #print(contents)
    return contents

# Picks the conversation settings note that gemini-2.0-flash-exp-image-generation does not accept system instructions
def get_gemini_settings(model):
    if model == "gemini-2.0-flash-exp-image-generation":
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
            response_modalities=[
                "image",
                "text",
            ],
            response_mime_type="text/plain",
        )
        return generate_content_config
    else:
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
            response_mime_type="text/plain",
            system_instruction=st.session_state.system_prompt,
        )
        return generate_content_config

# ----------------------------------------------------------------
# Streamlit UI setup
st.set_page_config(page_title="Gemini Chat", page_icon=":speech_balloon:", layout="wide")

# Initialize conversation history in session state.

#Conversation is the internal behind scenes we send to gemini.
if "conversation" not in st.session_state:
    st.session_state.conversation = []
#Thread is what we display to the user, it is a simplified version of the conversation
if "thread" not in st.session_state:
    st.session_state.thread = []

# Sidebar: display title and a button to clear the conversation.
st.sidebar.image(side_bar_image)

# Model Choices
model_choice = st.sidebar.selectbox(
    "Model",
    ("gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash-thinking-exp-01-21","gemini-2.0-flash-exp-image-generation")
)

def clear_conversation():
    """Clears the conversation history."""
    st.session_state.conversation.clear()
    st.session_state.thread.clear()
    st.toast("Memory Cleared")

# System Prompt Stuff
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = SYSTEM_PROMPT

#Make the Bar
system_bar = st.sidebar.text_area(
    "System Prompt", 
    value=st.session_state.system_prompt, 
    key="system_prompt"
)

# button to clear conversation
if st.sidebar.button("Clear Conversation", use_container_width=True):
    clear_conversation()

# Display thread in SD chat
for msg in st.session_state.thread:
    if msg["role"] == "user":
        with st.chat_message("user", avatar=userAvatar):
            st.markdown(msg["text"])
    else:
        with st.chat_message("assistant", avatar=aiAvatar):
            st.markdown(msg["text"])

# ----------------------------------------------------------------
# Chat input and response generation

def chat_loop():
    if prompt := st.chat_input(
        placeholder="What is up?",
        accept_file=not disable_fileUpload,
        file_type=["txt", "csv", "py", "cs", "ts", "js", "pdf", "html", "md", "xml", "yaml", "yml", "sql", "css", "php", "png", "jpeg", "jpg"]
    ):
        # User Input is what we show the user
        user_input = prompt.text
        # Processed creates the extra data
        processed_input = user_input

        # Process URLs: replace any URL with its transcript or scraped content.
        urls = re.findall(r'(https?://\S+)', user_input)
        for url in urls:
            transcript = get_transcript_from_url(url)
            processed_input = processed_input.replace(url, transcript)

        # Prepare a list for file attachments to include in the conversation.
        file_attachments = []
        # Initialize Gemini client for file uploads.
        client = genai.Client(api_key=API_KEY)
    
        # Process any uploaded files.
        if prompt.get("files"):
            for file in prompt["files"]:
                if only_attach:
                    file_obj = client.files.upload(file=tmp_path)
                    file_attachments.append(file_obj)
                    processed_input += f"\nFile uploaded: {file.name}"
                    user_input += " File:" + file.name
                    continue
                elif file.type == "application/pdf":
                    file_text = read_pdf(file)
                    processed_input += f"\n{file_text}\nFile: {file.name}"
                    user_input += " PDF:" + file.name
                elif file.type in ["image/jpeg", "image/png"]:
                    suffix = os.path.splitext(file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name
                    try:
                        file_obj = client.files.upload(file=tmp_path)
                        file_attachments.append(file_obj)
                        processed_input += f"\nImage uploaded: {file.name}"
                        user_input += " Image:" + file.name
                    except Exception as e:
                        processed_input += f"\nError uploading image {file.name}: {e}"
                    finally:
                        os.remove(tmp_path)
                else:
                    file_text = file.read().decode("utf-8")
                    processed_input += f"\n{file_text}\nFile: {file.name}"
                    user_input += " File:" + file.name

        # Append the user's message along with any file attachments to the conversation.
        conversation_entry = {"role": "user", "text": processed_input}
        if file_attachments:
            conversation_entry["files"] = file_attachments
        st.session_state.conversation.append(conversation_entry)
        thread_entry = {"role": "user", "text": user_input}
        st.session_state.thread.append(thread_entry)
        with st.chat_message("user", avatar=userAvatar):
            st.markdown(user_input)

        # Build the conversation contents for Gemini.
        contents = build_contents()
        # Stream the response from Gemini.
        response_text = ""
        response_thread = ""
        response_image = None
        with st.spinner('Processing...', show_time=True):
            with st.chat_message("assistant", avatar=aiAvatar):
                placeholder = st.empty()
                response_image = None
                for chunk in client.models.generate_content_stream(
                    model=model_choice,
                    contents=contents,
                    config=get_gemini_settings(model_choice),
                ):
                    if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                        continue
                    # Gemini Pro image introduced a new inline_data for images
                    if chunk.candidates[0].content.parts[0].inline_data:
                        inline_data = chunk.candidates[0].content.parts[0].inline_data
                        inline_bytes = inline_data.data if hasattr(inline_data, "data") else inline_data
                        base64_image = base64.b64encode(inline_bytes).decode("utf-8")
                        # Generate markdown for inline display
                        markdown_image = f"![image](data:image/png;base64,{base64_image})"
                        response_thread += markdown_image
                        # Save the base64 image data for storage
                        response_image = base64_image
                    else:                       
                        response_text += chunk.text
                        response_thread += chunk.text
                    placeholder.markdown(response_thread)
        if model_choice == "gemini-2.0-flash-exp-image-generation": # Gemini!! changed the term from assistant to model 
            entry = {"role": "model", "text": response_text}
            entryThread = {"role": "model", "text": response_thread}
        else:
            entry = {"role": "assistant", "text": response_text}
            entryThread = {"role": "model", "text": response_thread}
        if response_image is not None:
            entry["image"] = response_image
        st.session_state.conversation.append(entry)
        st.session_state.thread.append(entryThread)

def displayExtra():
    st.sidebar.markdown("♊ Gemini can make mistakes.");
    st.sidebar.markdown("ℹ️ gemini-2.0-flash-exp-image-generation does not accept system instructions.");
    st.sidebar.markdown("⚠️ Please note that chats are not saved. If you navigate away from this page, you will lose your conversation.")

if requireKey:
    if st.query_params.__contains__("secretkey") and st.query_params["secretkey"] == SECRET_KEY:
        chat_loop()
        displayExtra()
    else:
        st.header('Access is Denied', divider='rainbow')
        st.image(error_image)
else:
    chat_loop()
    displayExtra()
