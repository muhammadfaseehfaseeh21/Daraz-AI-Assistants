# Daraz-AI-Assistants# 🛍️ Daraz Knowledge Assistant

A simple RAG-based AI Knowledge Assistant built with:

- Python
- Streamlit
- Google Drive
- Sentence Transformers
- FAISS
- Groq
- RAG

## 📚 Knowledge Base

The Google Drive folder should contain these subfolders:

Daraz-Knowledge-Base/

├── return/
├── delivery/
├── sellers/
├── payments/
├── refund/
└── customer-supports-policy/

Each folder can contain:

- PDF
- DOCX
- TXT
- MD

documents.

## 🔄 How the App Works

Google Drive
       ↓
Download Knowledge Base
       ↓
Extract Text
       ↓
Split Text into Chunks
       ↓
Create Embeddings
       ↓
FAISS Vector Database
       ↓
User Question
       ↓
Semantic Search + Keyword Search
       ↓
Relevant Context
       ↓
Groq LLM
       ↓
Final Answer

## 🔐 Groq API Key

The app uses the environment variable:

GROQ_API_KEY

For Streamlit Cloud:

1. Open your Streamlit app.
2. Go to Settings.
3. Open Secrets.
4. Add:

GROQ_API_KEY = "your_groq_api_key"

Optional model setting:

GROQ_MODEL = "openai/gpt-oss-120b"

## 📁 Google Drive

The Google Drive folder must be publicly accessible.

Set:

Anyone with the link → Viewer

Copy the folder URL and put it in the Streamlit sidebar.

## 🚀 Deploy on Streamlit Cloud

1. Create a GitHub repository.
2. Upload:

app.py
requirements.txt
README.md

3. Open Streamlit Cloud.
4. Create a new app.
5. Select your GitHub repository.
6. Select `app.py` as the main file.
7. Deploy.
8. Add `GROQ_API_KEY` in Streamlit Secrets.

## 🤖 Features

- Google Drive knowledge base
- Multiple knowledge-base categories
- PDF extraction
- DOCX extraction
- TXT extraction
- Markdown extraction
- Text chunking
- Sentence Transformer embeddings
- FAISS semantic search
- Keyword search
- Hybrid retrieval
- Groq LLM
- Source display
- Filename metadata
- Page metadata
- Category metadata
- Knowledge-base refresh
- Chat interface
