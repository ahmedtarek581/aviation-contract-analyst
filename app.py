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
st.set_page_config(page_title="IAB SGHA Smart Search", layout="wide", page_icon="✈️")

# Valid API Key Check
if "HF_TOKEN" not in st.secrets:
    st.error("⚠️ Error: HF_TOKEN not found in secrets. Please add it to your Streamlit secrets.")
    st.stop()

# Initialize Llama 3 Model (Serverless API)
REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct" 
client = InferenceClient(model=REPO_ID, token=st.secrets["HF_TOKEN"])

@st.cache_resource
def load_embedding_model():
    # This runs once on startup to handle semantic chunk searches
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()

# ---------------------------------------------------------
# 2. IATA SGHA MAIN AGREEMENT BASELINE KNOWLEDGE
# ---------------------------------------------------------
IATA_BASELINES = {
    "IATA SGHA 2023": """
    IATA SGHA 2023 MAIN AGREEMENT BASELINE:
    - Article 1-3: Defines management of services, subcontracting rules, and operational safety obligations.
    - Article 4 (Carrier's Representation): Carrier can maintain its own representative to oversee operations.
    - Article 5 (Standard of Work): Services must comply with IATA/ICAO and the IATA Ground Operations Manual (IGOM). Mandates SMS (Safety Management System).
    - Article 7 & 8 (Remuneration and Accounting): Governs standard payment timelines, billing currencies, and rights to suspend service or demand cash-basis operations/pre-payments if airline financial defaults occur.
    - Article 9 (Liability and Indemnity): Default liability for aircraft physical damage caps at standard IATA limits (calculated on a percentage of handling fees or indexed fixed max limits per occurrence) unless caused by gross negligence or willful misconduct. Indemnity clauses favor mutual protection except in proven negligence.
    - Article 11 (Duration and Termination): Default termination notice period is 60 days for cause/insolvency, or 60/90 days standard bilateral notice without cause unless modified by Annex B.
    """,
    "IATA SGHA 2018": """
    IATA SGHA 2018 MAIN AGREEMENT BASELINE:
    - Article 4: Carrier representation allowances at local stations.
    - Article 5: Standards of performance explicitly tied to safe industry practices and standard IGOM manuals.
    - Article 7 & 8: Remuneration structures, accounting practices, and credit defaults.
    - Article 9 (Liability and Indemnity): Standardized limitations of liability for aircraft physical damage and cargo losses. Protects the handler from consequential losses unless clear gross misconduct is established.
    - Article 11: Governs duration, 60-day notice baseline for alterations or general termination timelines.
    """,
    "IATA SGHA 2013": """
    IATA SGHA 2013 MAIN AGREEMENT BASELINE:
    - Baseline features similar legal mechanics to 2018 but has older formatting definitions for specific sub-services in Annex A.
    - Article 9: Classic structural liability caps for ground handling accidents based on structural aircraft weights or preset pricing coefficients.
    - Article 11: Rigid duration framework requiring formal structural notification periods.
    """
}

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
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
    """ Extracts 3 primary keywords/topics using Llama 3 """
    sample_text = text[:2000] 
    
    prompt = f"""
    Analyze the following text from an aviation ground handling document and extract exactly 3 main keywords or topics (e.g., Ramp Handling, Passenger Services, Payment Terms). 
    Return ONLY the keywords separated by commas.
    
    Text: {sample_text}
    """
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat_completion(messages, max_tokens=50)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error extracting keywords: {str(e)}"

def get_answer_from_llm(context, question, sgha_version):
    """
    Generates an answer combining the static IATA Main Agreement 
    baseline template with the uploaded custom Annex B data.
    """
    baseline_text = IATA_BASELINES.get(sgha_version, IATA_BASELINES["IATA SGHA 2023"])

    system_prompt = f"""You are an expert contract analyst for IAB (International Aviation Business). 
    You are analyzing an SGHA contract package consisting of an uploaded Annex B file and the background IATA Main Agreement.
    
    CRITICAL BASELINE RULES TO UNDERSTAND:
    {baseline_text}
    
    OPERATIONAL INSTRUCTION:
    Answer the user's question accurately. 
    1. If the question relates to high-level legal frameworks, liability caps, or baseline definitions not physically typed in Annex B, use the Main Agreement Baseline context above.
    2. If the user asks about local station variations, modifications, handovers, handling fees, or technical service choices, prioritize the provided Annex B text context below.
    3. If information is missing from both the baseline and the uploaded text, state cleanly: "I cannot find specific details for this query in the provided Annex B chunks or the standard framework rules."
    """
    
    user_message = f"""
    Context extracted from the uploaded Annex B PDF:
    {context}
    
    User's Question: 
    {question}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    try:
        response = client.chat_completion(messages, max_tokens=600, temperature=0.2)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# ---------------------------------------------------------
# 4. MAIN APP INTERFACE
# ---------------------------------------------------------
st.title("✈️ IAB SGHA Smart Search")
st.markdown("Upload a Standard Ground Handling Agreement (SGHA) Annex B document to analyze it alongside IATA Main Agreement frameworks.")

# Sidebar Controls for Versioning Context
st.sidebar.header("Contract Context Settings")
selected_version = st.sidebar.selectbox(
    "Select Governing IATA SGHA Version:", 
    ["IATA SGHA 2023", "IATA SGHA 2018", "IATA SGHA 2013"],
    help="Annex B contracts modify an underlying IATA Main Agreement version. Select the year specified in your contract preamble."
)

# File Uploader
uploaded_file = st.file_uploader("Upload Annex B PDF File", type="pdf")

if uploaded_file is not None:
    # 1. Process PDF and Vector Embeddings
    with st.spinner("Analyzing and parsing Annex B PDF text..."):
        full_text = extract_text_from_pdf(uploaded_file)
        chunks = split_text_into_chunks(full_text)
        
        # Create Vector Embeddings
        chunk_embeddings = embed_model.encode(chunks)
        
        # 2. Extract and Display Topic Identifiers
        keywords = get_keywords(full_text)
        st.success("Document Processed and Aligned with IATA Baseline Framework Successfully!")
        
        st.subheader("🔑 Key Topics Detected in Annex B")
        st.info(keywords)

    st.divider()

    # 3. Hybrid AI Query Engine
    st.subheader("🤖 Smart Contract Analysis Q&A")
    query = st.text_input("Ask about handling fees, liability, terminal deviations, or settlement cycles:")

    if query:
        with st.spinner("Cross-referencing Annex B context with IATA baseline..."):
            # Embed the query
            query_embedding = embed_model.encode([query])
            
            # Semantic Vector Search across the Annex B chunks
            similarities = cosine_similarity(query_embedding, chunk_embeddings)
            top_k_indices = np.argsort(similarities[0])[-4:][::-1] # Retrieve top 4 most relevant chunks
            
            # Assemble Context
            retrieved_context = "\n\n".join([chunks[i] for i in top_k_indices])
            
            # Send context combo to Llama 3
            answer = get_answer_from_llm(retrieved_context, query, selected_version)
            
            # Display Final Analysis
            st.markdown("### Analysis Results")
            st.write(answer)
            
            # Source Verification Expandable Panel
            with st.expander("View Isolated Annex B Context Segments"):
                st.text(retrieved_context)
