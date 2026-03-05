"""
pipeline.py
-----------
The core RAG pipeline. This is where everything connects:

1. Embed the user's question
2. Search ChromaDB for relevant chunks
3. Build a prompt with those chunks as context
4. Send to Groq (LLaMA 3) and get an answer
5. Return the answer
"""

import os
from groq import Groq
from dotenv import load_dotenv
from rag.embedder import embed_query
from rag.vectorstore import search_similar

load_dotenv(override=True)

# Initialize Groq client
_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Model to use — llama3-8b is free, fast, and capable
GROQ_MODEL = "llama-3.1-8b-instant"


def build_prompt(question: str, context_chunks: list[str]) -> str:
    """
    Build the RAG prompt.
    We inject retrieved chunks as context, then ask the question.
    This grounds the LLM's answer in YOUR document.
    """
    context = "\n\n---\n\n".join(context_chunks)

    return f"""You are a helpful document assistant. Answer the user's question based ONLY on the provided document context below.

If the answer is not found in the context, say: "I couldn't find information about that in the uploaded document."

Be thorough and detailed. Cover all sections of the document. Do not make up information.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:"""


def query_document(question: str, doc_name: str, top_k: int = 10) -> dict:
    """
    Full RAG query pipeline:
    question + doc_name → relevant chunks → LLM answer
    
    Returns a dict with 'answer' and 'sources' (the chunks used)
    """
    print(f"[Pipeline] Processing query: '{question[:60]}...'")

    # Step 1: Embed the question
    query_embedding = embed_query(question)

    # Step 2: Retrieve relevant chunks from vector DB
    relevant_chunks = search_similar(query_embedding, doc_name, top_k=top_k)

    if not relevant_chunks:
        return {
            "answer": "No relevant content found. Please make sure a document has been uploaded and processed.",
            "sources": []
        }

    # Step 3: Build the prompt
    prompt = build_prompt(question, relevant_chunks)

    # Step 4: Call Groq API
    print(f"[Pipeline] Calling Groq with {len(relevant_chunks)} context chunks...")
    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise document analysis assistant. Answer questions based strictly on provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,      # Low temp = more factual, less creative
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()
    print(f"[Pipeline] Got answer ({len(answer)} chars)")

    return {
        "answer": answer,
        "sources": relevant_chunks  # Return chunks so UI can show them
    }
