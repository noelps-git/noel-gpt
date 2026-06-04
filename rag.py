import torch
import fitz
import os
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load embedding model
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")

# Load your fine-tuned GPT-2
print("Loading your fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained("./gosense-gpt-final")
model = AutoModelForCausalLM.from_pretrained("./gosense-gpt-final")
model = model.to(device)
model.eval()
print("Language model loaded.")

# Setup ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection("gosense_knowledge")

# Function to chunk text
def chunk_text(text, chunk_size=200, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks

# Function to ingest a PDF
def ingest_pdf(pdf_path):
    print(f"Ingesting: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    chunks = chunk_text(text)
    embeddings = embed_model.encode(chunks, show_progress_bar=True)
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"{pdf_path}_chunk_{i}" for i in range(len(chunks))]
    )
    print(f"Ingested {len(chunks)} chunks from {pdf_path}")

# Function to retrieve relevant chunks
def retrieve(query, n=2):
    query_embedding = embed_model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n
    )
    return results["documents"][0]

# Function to generate with RAG
def rag_generate(query, max_new_tokens=200, temperature=0.8):
    # Retrieve context
    context_chunks = retrieve(query)
    context = "\n\n".join(context_chunks)[:800]

    # Build prompt
    prompt = f"""Context from your knowledge base:
{context}

Based on the above context, write in your voice:
{query}
---
"""
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output[len(prompt):]

# Ingest your training PDF as knowledge base
ingest_pdf("training data for posts.pdf")

# Test RAG
print("\n--- RAG GENERATION ---\n")
query = "Write about UPI fraud detection"
print(f"Query: {query}\n")
response = rag_generate(query)
print(response)