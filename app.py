import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import PyPDF2
import docx
from io import StringIO

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def fetch_website_content(url, max_pages=5):
    def get_links(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith(url)]
    
    visited = set()
    to_visit = [url]
    content = ""
    
    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url not in visited:
            try:
                response = requests.get(current_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                content += soup.get_text() + "\n\n"
                visited.add(current_url)
                to_visit.extend([link for link in get_links(current_url) if link not in visited])
            except Exception as e:
                st.error(f"Error fetching {current_url}: {str(e)}")
    
    return content

def read_pdf(file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
    return text

def read_docx(file):
    text = ""
    try:
        doc = docx.Document(file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
    return text

def read_txt(file):
    text = ""
    try:
        stringio = StringIO(file.getvalue().decode("utf-8"))
        text = stringio.read()
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
    return text

def create_vector_store(content):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_text(content)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    
    vector_store = FAISS.from_texts(splits, embeddings)
    return vector_store

def answer_question(vector_store, question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    
    return qa.run(question)

# Streamlit UI
st.title("Advanced Document Q&A Chatbot (Gemini)")

# Input method selection
input_method = st.radio(
    "Choose input method:",
    ("Website URL", "Upload Document")
)

content = ""

if input_method == "Website URL":
    url = st.text_input("Enter website URL:")
    if url:
        with st.spinner("Fetching website content..."):
            content = fetch_website_content(url)

else:
    uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'docx', 'txt'])
    if uploaded_file:
        with st.spinner("Processing document..."):
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                content = read_pdf(uploaded_file)
            elif file_extension == 'docx':
                content = read_docx(uploaded_file)
            elif file_extension == 'txt':
                content = read_txt(uploaded_file)

question = st.text_input("Ask a question about the content:")

if content and question:
    with st.spinner("Creating vector store..."):
        vector_store = create_vector_store(content)
    
    with st.spinner("Analyzing and answering..."):
        answer = answer_question(vector_store, question)
    
    st.write("Answer:", answer)

    # Chat History
    st.markdown("---")
    st.write("Chat History:")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append((question, answer))
    
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        st.write(f"Q{i}: {q}")
        st.write(f"A{i}: {a}")
