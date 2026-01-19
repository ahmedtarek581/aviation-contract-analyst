import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub # Changed for Cloud
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PAGE CONFIG
st.set_page_config(page_title="SGHA Analyst", layout="wide")
st.title("✈️ IATA SGHA Contract Analyst (Cloud Edition)")

# GET API TOKEN FROM SECRETS
# This keeps your key safe!
api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# 1. THE BRAIN (Now uses API, no download needed!)
def load_llm():
    return HuggingFaceHub(
        repo_id="MBZUAI/LaMini-Flan-T5-783M",
        model_kwargs={"temperature": 0.1, "max_length": 512},
        huggingfacehub_api_token=api_token
    )

llm = load_llm()

# 2. DOCUMENT PROCESSING
@st.cache_resource
def process_document(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    pages = text_splitter.split_documents(documents)
    
    # We use a lighter embedding model for the cloud
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(pages, embeddings)
    return vector_store

# 3. UI LAYOUT
with st.sidebar:
    st.header("Upload Contract")
    uploaded_file = st.file_uploader("Choose PDF", type="pdf")

if uploaded_file is not None:
    # Save temp file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing (Cloud CPU)..."):
        vector_store = process_document("temp.pdf")
        st.success("✅ Ready")

    user_question = st.text_input("Ask a question:")

    if user_question:
        docs = vector_store.similarity_search(user_question, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        prompt = f"Question: {user_question}\nContext: {context}\nAnswer:"
        
        response = llm.invoke(prompt)
        
        st.write("### 🤖 Answer:")
        st.info(response)