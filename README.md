# 📄 Smart Document Insights

A RAG-powered document Q&A system. Upload any PDF, DOCX, or TXT file and ask questions about it using natural language.

## Stack
- **Backend**: Python + Flask
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2) — runs locally, free
- **Vector DB**: ChromaDB — persistent, local
- **LLM**: Groq API (LLaMA 3) — free tier
- **Frontend**: Vanilla HTML/CSS/JS (black & white aesthetic)

---

## 🚀 Setup

### 1. Clone & Install
```bash
cd smart-doc-insights
pip install -r requirements.txt
```

### 2. Get a Free Groq API Key
1. Go to https://console.groq.com
2. Sign up (free)
3. Create an API key

### 3. Configure Environment
Edit `.env` and paste your key:
```
GROQ_API_KEY=gsk_your_key_here
```

### 4. Run
```bash
python app.py
```

Visit: http://localhost:5000

---

## 📁 Project Structure
```
smart-doc-insights/
├── app.py                  # Flask app + routes
├── rag/
│   ├── ingestion.py        # Parse + chunk documents
│   ├── embedder.py         # sentence-transformers embeddings
│   ├── vectorstore.py      # ChromaDB operations
│   └── pipeline.py         # RAG query pipeline (Groq)
├── templates/
│   └── index.html          # Single-page UI
├── uploads/                # Temp file storage (auto-cleaned)
├── chroma_db/              # Persistent vector store
├── requirements.txt
└── .env                    # Your API key (never commit this!)
```

---

## 🌐 Deploying (Free)

### Render.com (Recommended)
1. Push to GitHub
2. Create new Web Service on Render
3. Set environment variable: `GROQ_API_KEY`
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app`

> Add `gunicorn` to requirements.txt for deployment.

### Railway
Same process — Railway auto-detects Flask apps.

---

## How It Works

```
Upload → Extract Text → Chunk → Embed → Store in ChromaDB
Query  → Embed Query  → Search ChromaDB → Build Prompt → Groq LLM → Answer
```
