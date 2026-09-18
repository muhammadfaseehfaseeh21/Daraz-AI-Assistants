import os
import re
import shutil
from pathlib import Path

import streamlit as st
import numpy as np
import faiss
import gdown

from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# APP SETTINGS
# ============================================================

st.set_page_config(
    page_title="Daraz Knowledge Assistant",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Daraz Knowledge Assistant")
st.caption("RAG-based assistant for Daraz policies and knowledge base")


# ============================================================
# CONFIGURATION
# ============================================================

# Google Drive folder URL
DEFAULT_DRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/"
    "1X--qt9GqHiOPn0nMKe6c0WIDQbED9DJa?usp=sharing"
)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

KNOWLEDGE_FOLDER = Path("knowledge_base")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

drive_url = st.sidebar.text_input(
    "Google Drive Folder URL",
    value=DEFAULT_DRIVE_FOLDER
)

top_k = st.sidebar.slider(
    "Number of retrieved chunks",
    min_value=2,
    max_value=10,
    value=5
)

if st.sidebar.button("🔄 Refresh Knowledge Base"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


# ============================================================
# GOOGLE DRIVE DOWNLOAD (FIXED GDOWN ERROR)
# ============================================================

@st.cache_data(show_spinner=False)
def download_google_drive_folder(folder_url):

    if KNOWLEDGE_FOLDER.exists():
        shutil.rmtree(KNOWLEDGE_FOLDER)

    KNOWLEDGE_FOLDER.mkdir(parents=True, exist_ok=True)

    try:
        # Error Fixed: 'remaining_ok' and 'use_cookies' parameters removed for gdown compatibility
        gdown.download_folder(
            url=folder_url,
            output=str(KNOWLEDGE_FOLDER),
            quiet=True
        )

        return True, None

    except Exception as e:
        return False, str(e)


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_pdf(file_path):
    text_pages = []
    try:
        reader = PdfReader(str(file_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                text_pages.append({
                    "text": text,
                    "page": page_number
                })
    except Exception as e:
        st.warning(f"Could not read PDF: {file_path.name} - {e}")
    return text_pages


def extract_docx(file_path):
    text = ""
    try:
        document = Document(str(file_path))
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        st.warning(f"Could not read DOCX: {file_path.name} - {e}")

    return [{
        "text": text,
        "page": None
    }] if text.strip() else []


def extract_text_file(file_path):
    try:
        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
        return [{
            "text": text,
            "page": None
        }] if text.strip() else []
    except Exception as e:
        st.warning(f"Could not read file: {file_path.name} - {e}")
    return []


def extract_document(file_path):
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        return extract_pdf(file_path)
    elif extension == ".docx":
        return extract_docx(file_path)
    elif extension in [".txt", ".md"]:
        return extract_text_file(file_path)
    return []


# ============================================================
# FIND CATEGORY
# ============================================================

def get_category(file_path):
    relative_parts = file_path.relative_to(KNOWLEDGE_FOLDER).parts
    if len(relative_parts) >= 2:
        return relative_parts[0]
    return "General"


# ============================================================
# TEXT CLEANING & CHUNKING
# ============================================================

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunks(text):
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# LOAD DOCUMENTS
# ============================================================

@st.cache_data(show_spinner=False)
def load_documents():
    documents = []
    supported_extensions = [".pdf", ".docx", ".txt", ".md"]

    for file_path in KNOWLEDGE_FOLDER.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in supported_extensions:
            continue

        pages = extract_document(file_path)
        category = get_category(file_path)

        for page_data in pages:
            text = page_data["text"]
            chunks = create_chunks(text)
            for chunk in chunks:
                documents.append({
                    "text": chunk,
                    "filename": file_path.name,
                    "category": category,
                    "page": page_data["page"]
                })

    return documents


# ============================================================
# EMBEDDINGS & FAISS INDEX
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def create_vector_database(documents):
    model = load_embedding_model()
    texts = [document["text"] for document in documents]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))

    return index


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def keyword_score(query, text):
    query_words = set(re.findall(r"\b[a-zA-Z0-9]+\b", query.lower()))
    text_words = set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))
    if not query_words:
        return 0
    matches = query_words.intersection(text_words)
    return len(matches) / len(query_words)


def retrieve_documents(query, documents, index, embedding_model, k=5):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    semantic_scores, indices = index.search(
        query_embedding,
        min(k * 3, len(documents))
    )

    results = []
    for score, idx in zip(semantic_scores[0], indices[0]):
        if idx < 0:
            continue

        document = documents[idx]
        semantic_score = float(score)
        keyword = keyword_score(query, document["text"])

        # Hybrid Score
        final_score = (0.75 * semantic_score) + (0.25 * keyword)

        results.append({
            "document": document,
            "score": final_score
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]


# ============================================================
# GROQ CLIENT & ANSWER GENERATION
# ============================================================

@st.cache_resource
def get_groq_client(api_key):
    return Groq(api_key=api_key)


def generate_answer(question, retrieved_documents, client):
    context_parts = []
    for item in retrieved_documents:
        document = item["document"]
        source = (
            f"Category: {document['category']}\n"
            f"File: {document['filename']}\n"
        )
        if document["page"]:
            source += f"Page: {document['page']}\n"
        source += f"Content:\n{document['text']}"
        context_parts.append(source)

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """
You are Daraz Knowledge Assistant.
Answer the user's question using ONLY the provided knowledge base context.

Rules:
1. Do not invent information.
2. If the answer is not available in the context, clearly say that the information was not found in the knowledge base.
3. Give a simple and helpful answer.
4. Keep the answer relevant to the question.
5. Mention the relevant source category/file when useful.
6. Do not claim that you checked Daraz's live website.
"""

    user_prompt = f"""
Knowledge Base Context:
{context}

User Question:
{question}

Answer the question using the knowledge base.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL = "llama-3.3-70b-versatile"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=800
    )

    return response.choices[0].message.content


# ============================================================
# APPLICATION FLOW
# ============================================================

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.warning("⚠️ GROQ_API_KEY is not configured.")
    st.info("Add GROQ_API_KEY in Streamlit Cloud → Settings → Secrets.")
    st.stop()


# Download knowledge base
with st.spinner("📥 Loading Google Drive knowledge base..."):
    success, error = download_google_drive_folder(drive_url)

if not success:
    st.error("Could not download the Google Drive folder.")
    st.code(error)
    st.info("Make sure the Google Drive folder is shared as 'Anyone with the link → Viewer'.")
    st.stop()


# Load documents
with st.spinner("📚 Reading documents..."):
    documents = load_documents()

if not documents:
    st.error("No supported documents were found.")
    st.info("Add PDF, DOCX, TXT or MD files inside your Google Drive knowledge-base folders.")
    st.stop()


# Create FAISS Index
with st.spinner("🧠 Creating embeddings and FAISS index..."):
    vector_index = create_vector_database(documents)

embedding_model = load_embedding_model()


# Sidebar Info
st.sidebar.success(f"📄 {len(documents)} chunks loaded")
categories = sorted(set(document["category"] for document in documents))

st.sidebar.write("### 📂 Categories")
for category in categories:
    st.sidebar.write(f"• {category}")


# Chat UI
st.subheader("💬 Ask about Daraz policies")

question = st.chat_input("Example: What is the return policy?")

if question:
    st.chat_message("user").write(question)

    with st.spinner("🔎 Searching the knowledge base..."):
        retrieved = retrieve_documents(
            question,
            documents,
            vector_index,
            embedding_model,
            top_k
        )

    client = get_groq_client(groq_api_key)

    with st.spinner("🤖 Generating answer..."):
        answer = generate_answer(question, retrieved, client)

    st.chat_message("assistant").write(answer)

    # Sources Expander
    with st.expander("📚 Sources used"):
        for number, item in enumerate(retrieved, start=1):
            document = item["document"]
            st.markdown(
                f"""
**{number}. {document['filename']}**
- Category: `{document['category']}`
- Page: `{document['page'] if document['page'] else 'N/A'}`
- Retrieval score: `{item['score']:.3f}`

{document['text'][:500]}...
"""
            )
