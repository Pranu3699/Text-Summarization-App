import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from dotenv import load_dotenv
import os

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Text Summarization App",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠Text Summarization App")
st.write("Upload a text file or paste your text below to generate a concise summary using OpenAI GPT models.")

# -----------------------------
# Sidebar Options
# -----------------------------
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ["gpt-3.5-turbo", "gpt-4"]
)
chunk_size = st.sidebar.slider("Text Chunk Size", 500, 3000, 1500, step=100)

# -----------------------------
# Input: File Upload or Text Box
# -----------------------------
uploaded_file = st.file_uploader("📄 Upload a .txt or .pdf file", type=["txt", "pdf"])
text_input = st.text_area("Or paste your text here:", height=200)

# -----------------------------
# Extract text from file
# -----------------------------
text = ""
if uploaded_file:
    if uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text()

elif text_input:
    text = text_input

# -----------------------------
# Summarization Logic
# -----------------------------
if text:
    st.subheader("🔍 Original Text Preview")
    st.write(text[:1000] + "..." if len(text) > 1000 else text)

    if st.button("🪄 Generate Summary"):
        with st.spinner("Summarizing... please wait ⏳"):

            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=100,
                length_function=len
            )
            chunks = splitter.split_text(text)
            docs = [Document(page_content=t) for t in chunks]

            # Load LLM
            llm = ChatOpenAI(model=model_choice, temperature=0, openai_api_key=OPENAI_API_KEY)

            # Create summarization chain
            chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)

            # Run summarization
            summary = chain.run(docs)

            # Display Summary
            st.subheader("📝 Summary Output")
            st.success(summary)

            # Option to Download Summary
            st.download_button(
                label="⬇️ Download Summary as Text File",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )
else:
    st.info("Please upload a file or paste text to begin.")
