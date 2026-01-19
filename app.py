import streamlit as st
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
# Page Config
st.set_page_config(page_title="IAB Contract Analyzer", layout="wide")

# Valid API Key Check
if "HF_TOKEN" not in st.secrets:
    st.error("⚠️ Error: HF_TOKEN not found in secrets. Please add it to your Streamlit secrets.")
    st.stop()

# Initialize Models
# We use a good, free instruction model for the logic
REPO_ID = "HuggingFaceH4/zephyr-7b-beta" 
client = InferenceClient(model=REPO_ID, token=st.secrets["HF_TOKEN"])

@st.cache_resource
def load_embedding_model():
    # This runs once and saves time
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def get_keywords(text):
    """
    Restored Functionality: Extracts 3 keywords using the LLM.
    """
    # Take the first 2000 characters to find the main topic
    sample_text = text[:2000] 
    
    prompt = f"""
    Analyze the following text and extract exactly 3 main keywords or topics. 
    Return ONLY the keywords separated by commas.
    
    Text: {sample_text}
    """
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        # UPDATED: Replaced client.post with client.chat_completion
        response = client.chat_completion(messages, max_tokens=50)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error extracting keywords: {str(e)}"

def get_answer_from_llm(context, question):
    """
    Generates an answer based on the context.
    """
    system_prompt = """You are an expert contract analyst for IAB (International Aviation Business). 
    Answer the user's question strictly based on the provided context chunks from the SGHA.
    If the answer is not in the text, say "I cannot find this information in the document."
    """
    
    user_message = f"""
    Context from Document:
    {context}
    
    Question: 
    {question}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    try:
        # UPDATED: Replaced client.post with client.chat_completion
        response = client.chat_completion(messages, max_tokens=500, temperature=0.3)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# ---------------------------------------------------------
# 3. MAIN APP INTERFACE
# ---------------------------------------------------------
st.title("✈️ IAB SGHA Smart Search")
st.markdown("Upload a Standard Ground Handling Agreement (SGHA) to analyze it.")

# File Uploader
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    # 1. Process PDF
    with st.spinner("Processing document..."):
        full_text = extract_text_from_pdf(uploaded_file)
        chunks = split_text_into_chunks(full_text)
        
        # Create Embeddings for chunks
        chunk_embeddings = embed_model.encode(chunks)
        
        # 2. Extract Keywords (Restored Feature)
        keywords = get_keywords(full_text)
        st.success("Document Processed Successfully!")
        
        # Display Keywords
        st.subheader("🔑 Key Topics Detected")
        st.info(keywords)

    st.divider()

    # 3. Q&A Section
    query = st.text_input("Ask a question about this contract:")

    if query:
        with st.spinner("Searching for answer..."):
            # Embed the query
            query_embedding = embed_model.encode([query])
            
            # Find closest chunks (Vector Search)
            similarities = cosine_similarity(query_embedding, chunk_embeddings)
            top_k_indices = np.argsort(similarities[0])[-3:][::-1] # Get top 3 chunks
            
            # Combine top chunks into context
            retrieved_context = "\n\n".join([chunks[i] for i in top_k_indices])
            
            # Generate Answer
            answer = get_answer_from_llm(retrieved_context, query)
            
            # Display Result
            st.markdown("### 🤖 Analysis")
            st.write(answer)
            
            # Optional: Show source chunks for verification
            with st.expander("View Source Context"):
                st.text(retrieved_context)
