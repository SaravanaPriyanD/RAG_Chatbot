import streamlit as st
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import re
import pdfplumber
import docx
from bs4 import BeautifulSoup
import requests
import base64

CHUNK_SIZE = 250
CHUNK_OVERLAP = 50

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_html(file):
    html = file.read()
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")

def extract_text(file):
    fname = file.name.lower()
    try:
        if fname.endswith(".pdf"):
            return extract_text_from_pdf(file)
        elif fname.endswith(".docx"):
            return extract_text_from_docx(file)
        elif fname.endswith(".html") or fname.endswith(".htm"):
            return extract_text_from_html(file)
        else:
            return file.read().decode("utf-8")
    except Exception as e:
        return ""

def load_and_chunk(files):
    all_chunks = []
    sources = []
    for file in files:
        text = extract_text(file)
        file_chunks = chunk_text(text)
        all_chunks.extend(file_chunks)
        sources.extend([file.name]*len(file_chunks))
        file.seek(0)  # reset pointer for later
    return all_chunks, sources

def fetch_confluence_pages(base_url, api_token, email, page_ids):
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{email}:{api_token}'.encode()).decode()}",
        "Accept": "application/json",
    }
    all_chunks = []
    sources = []
    for pid in page_ids:
        url = f"{base_url}/wiki/rest/api/content/{pid}?expand=body.storage"
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            title = data.get("title", f"Page {pid}")
            html = data["body"]["storage"]["value"]
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n")
            file_chunks = chunk_text(text)
            all_chunks.extend(file_chunks)
            sources.extend([f"Confluence: {title} ({pid})"] * len(file_chunks))
        else:
            st.warning(f"Could not fetch page {pid}: {resp.status_code}")
    return all_chunks, sources

st.title("RAG Chatbot: PDF, DOCX, HTML, TXT, MD & Confluence API")
st.markdown(
    "Upload docs or fetch Confluence pages. Ask a question to get RAG-powered streaming answers."
)

tab1, tab2 = st.tabs(["Upload Files", "Fetch from Confluence API"])
uploaded_files, api_chunks, api_sources = [], [], []

with tab1:
    uploaded_files = st.file_uploader(
        "Upload your documentation files",
        type=['txt', 'md', 'pdf', 'docx', 'html', 'htm'],
        accept_multiple_files=True
    )

with tab2:
    st.markdown("**Enter your Confluence API details:**")
    conf_base_url = st.text_input("Confluence Base URL (e.g., https://yourorg.atlassian.net)")
    conf_email = st.text_input("Email (for Atlassian API Auth)")
    conf_api_token = st.text_input("API Token (create at id.atlassian.com/manage-profile/security/api-tokens)", type="password")
    conf_page_ids = st.text_input("Comma-separated Page IDs (e.g., 123456,654321)")
    fetch_btn = st.button("Fetch Pages")

    if fetch_btn and conf_base_url and conf_api_token and conf_page_ids and conf_email:
        with st.spinner("Fetching Confluence pages..."):
            page_id_list = [pid.strip() for pid in conf_page_ids.split(",") if pid.strip().isdigit()]
            api_chunks, api_sources = fetch_confluence_pages(
                conf_base_url.strip(), conf_api_token.strip(), conf_email.strip(), page_id_list
            )
        if api_chunks:
            st.success(f"Fetched {len(api_chunks)} chunks from {len(page_id_list)} pages.")

chunks, chunk_sources = [], []
if uploaded_files:
    f_chunks, f_sources = load_and_chunk(uploaded_files)
    chunks.extend(f_chunks)
    chunk_sources.extend(f_sources)
if api_chunks:
    chunks.extend(api_chunks)
    chunk_sources.extend(api_sources)

if not chunks:
    st.warning("No extractable text found in the uploaded or fetched docs.")
    st.stop()

@st.cache_resource(show_spinner=True)
def get_models_and_vectors(text_chunks):
    tokenized_corpus = [re.findall(r'\w+', chunk.lower()) for chunk in text_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(text_chunks)
    return bm25, model, embeddings, tokenized_corpus

bm25, embed_model, doc_embeddings, tokenized_corpus = get_models_and_vectors(chunks)

def ensemble_retrieve(query, k=4, bm25_weight=0.5, vector_weight=0.5):
    tokenized_query = re.findall(r'\w+', query.lower())
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    query_emb = embed_model.encode([query])
    sim_scores = 1 - cdist(query_emb, doc_embeddings, metric="cosine")
    sim_scores = sim_scores.flatten()
    bm25_norm = (bm25_scores - bm25_scores.min()) / (np.ptp(bm25_scores) + 1e-8)
    sim_norm = (sim_scores - sim_scores.min()) / (np.ptp(sim_scores) + 1e-8)
    final_score = bm25_weight * bm25_norm + vector_weight * sim_norm
    top_idx = final_score.argsort()[::-1][:k]
    results = [{"chunk": chunks[i], "score": float(final_score[i]), "source": chunk_sources[i]} for i in top_idx]
    return results

def stream_llm_answer(prompt, api_key, model="gpt-3.5-turbo"):
    import openai
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
        stream=True,
    )
    full = ""
    for chunk in response:
        content = getattr(chunk.choices[0].delta, "content", None)
        if content:
            full += content
            yield full

# --- Chat history and Clear button ---
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Clear Chat", type="primary"):
    st.session_state.history = []
    st.rerun()

# === Top Matches Sidebar ===
with st.sidebar:
    st.markdown("**Streaming LLM (OpenAI):**")
    api_key = st.text_input("OpenAI API Key", type="password")
    model = st.selectbox("Model", options=["gpt-3.5-turbo", "gpt-4"], index=0)
    st.write(f"{len(chunks)} chunks loaded from {len(uploaded_files)} uploaded files and Confluence.")

    # Display top matches in the sidebar if there was a retrieval
    if st.session_state.get("top_matches"):
        st.markdown("---")
        st.markdown("### Top Matches")
        for i, doc in enumerate(st.session_state["top_matches"], 1):
            st.markdown(
                f"**{i}. {doc['source']} (score {doc['score']:.2f})**\n\n{doc['chunk'][:400]}{'...' if len(doc['chunk']) > 400 else ''}",
                unsafe_allow_html=True
            )

user_input = st.text_input("Ask a question about your docs (RAG):")

if user_input:
    st.session_state.history.append(("user", user_input))
    with st.spinner("Retrieving relevant chunks..."):
        top_docs = ensemble_retrieve(user_input, k=4)
    # Save top matches to session state for sidebar display
    st.session_state["top_matches"] = top_docs

    # Only display in chat window: user and LLM messages
    if api_key:
        context = "\n\n".join([f"Source: {doc['source']}\nContent: {doc['chunk']}" for doc in top_docs])
        rag_prompt = (
            "You are a helpful DevOps assistant. Use ONLY the documentation below to answer.\n"
            f"Question: {user_input}\n"
            f"Documentation:\n{context}\n"
            "Answer:"
        )
        st.session_state.history.append(("llm", ""))  # Placeholder for streaming
        st.markdown("**AI Synthesized Answer (Streaming):**")
        resp_placeholder = st.empty()
        answer = ""
        for partial in stream_llm_answer(rag_prompt, api_key, model):
            answer = partial
            resp_placeholder.markdown(answer + "▍")
        st.session_state.history[-1] = ("llm", answer)
    else:
        st.session_state.history.append(("llm", "Please enter your OpenAI API key in the sidebar for LLM-powered answers."))

# Display Chat History (user and LLM only)
for who, msg in st.session_state.history:
    if who == "user":
        st.markdown(f"**You:** {msg}")
    elif who == "llm" and msg:
        st.markdown(f"**AI Synthesized Answer:**\n{msg}")

st.markdown("---")
st.markdown(
    "**How it works:** Upload docs or fetch Confluence ➔ Ask a question ➔ Top chunks retrieved (BM25 + vector) are shown on the left ➔ Passed as context to OpenAI LLM ➔ Streaming answer!"
)
