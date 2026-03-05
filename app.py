"""
app.py
------
Flask application entry point.

Routes:
  GET  /              → Serve the main UI
  POST /upload        → Upload + process a document
  POST /query         → Ask a question about the document
  GET  /documents     → List all uploaded documents
  DELETE /document    → Delete a document
"""

import os
import uuid
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from rag.ingestion import ingest_document
from rag.embedder import embed_texts
from rag.vectorstore import store_chunks, list_documents, delete_document
from rag.pipeline import query_document

load_dotenv(override=True)

app = Flask(__name__)

# Config
UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    "um the first page"
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    upload here so it can extract text then chucnk them , then embed them and store them in chromadb
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not supported. Use PDF, DOCX, or TXT."}), 400

    try:
        #save it for tempo use, and add a unique id so it wont clash
        filename = secure_filename(file.filename)
        # Add unique prefix to avoid collisions
        unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(file_path)

        # Use original filename as the document identifier (without extension)
        doc_name = os.path.splitext(filename)[0]

        # Run ingestion pipeline
        chunks = ingest_document(file_path)

        if not chunks:
            return jsonify({"error": "Could not extract text from document. Is it scanned/image-based?"}), 400

        # Embed all chunks
        embeddings = embed_texts(chunks)

        # Store in ChromaDB
        store_chunks(chunks, embeddings, doc_name)

        # Clean up temp file
        os.remove(file_path)

        return jsonify({
            "success": True,
            "doc_name": doc_name,
            "chunks_created": len(chunks),
            "message": f"Document '{filename}' processed successfully! {len(chunks)} chunks indexed."
        })

    except Exception as e:
        print(f"[Upload Error] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    """
    return a answer to the question asked in json format
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    question = data.get("question", "").strip()
    doc_name = data.get("doc_name", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    if not doc_name:
        return jsonify({"error": "Please select a document first"}), 400

    try:
        result = query_document(question, doc_name)
        return jsonify({
            "success": True,
            "answer": result["answer"],
            "sources_count": len(result["sources"])
        })
    except Exception as e:
        print(f"[Query Error] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/documents", methods=["GET"])
def documents():
    """Return list of all indexed documents."""
    docs = list_documents()
    return jsonify({"documents": docs})


@app.route("/document", methods=["DELETE"])
def remove_document():
    """Delete a document from the vector store."""
    data = request.get_json()
    doc_name = data.get("doc_name", "").strip()

    if not doc_name:
        return jsonify({"error": "doc_name required"}), 400

    delete_document(doc_name)
    return jsonify({"success": True, "message": f"'{doc_name}' deleted."})


#_-___--------

if __name__ == "__main__":
    print("running at http://localhost:5000")
    app.run(debug=False, port=5000)
