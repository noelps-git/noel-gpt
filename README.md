# Noel's GPT

> A domain-specific AI that writes about fraud, security, and payment rails — in your voice, grounded in your knowledge. Runs entirely on your machine.

---

## The problem

Fraud and security professionals spend hours every week writing the same things.

Threat briefings. LinkedIn posts. Case narratives. Analyst reports. Client explainers.

The knowledge exists. Getting it out is the bottleneck.

Generic AI tools don't help much. They hallucinate fraud concepts. They write in a bland corporate voice. They have no access to your documents, your research, your thinking.

This project fixes that.

---

## What it does

Type a topic. The system searches your document library for the most relevant knowledge, passes it to a language model trained on your writing, and generates a response that sounds like you and stays accurate.

```
You type:   "Write about SIM swapping attacks on UPI platforms"

It does:    → searches your document library by meaning, not keywords
            → retrieves the 2 most relevant research chunks
            → passes them to LLaMA 3 as context
            → generates a response in your voice

You get:    a draft that reads like you wrote it
```

Everything runs locally. No data leaves your machine. No API costs. No subscriptions.

---

## Who this is for

**Fraud and risk professionals**
The model understands behavioral biometrics, entity graphs, rule engines, mule accounts, SIM swapping, UPI fraud patterns — because it was trained on them. Write faster without losing depth.

**Security researchers**
Feed your threat intelligence PDFs into the knowledge base. Ask for a briefing on a new attack vector. The system retrieves your own research and writes from it.

**Fintech and payments people**
Generate content about fraud detection, payment rail security, and identity verification without spending hours writing from scratch.

**Developers building domain AI**
This is a complete, working blueprint. Swap in your own documents and training data and you have a domain-specific AI system for any vertical — legal, medical, compliance, engineering.

---

## Use cases

| What you want | What happens |
|---|---|
| A LinkedIn post about a fraud pattern | Retrieves your research → writes in your voice |
| A threat briefing on a new attack | Feed the PDF → query for a briefing → grounded output |
| A case narrative for an alert | Describe the case → retrieves similar history → drafts it |
| A client explainer on mobile security | Retrieves product docs → writes a clear explainer |
| An answer from your document library | Ask anything → finds the answer in your corpus |

---

## How it works

**Two systems working together**

**Fine-tuning** teaches the model your voice. A base language model (GPT-2) is trained further on your writing — your sentence rhythm, your analytical style, your domain vocabulary. It learns the delta between general language and how you specifically write.

**RAG (Retrieval Augmented Generation)** gives the model memory. At the moment you ask a question, your document library is searched by semantic similarity. The most relevant chunks are retrieved and passed to the model as context. It generates using that knowledge — not from memory, from your actual documents.

Voice from fine-tuning. Accuracy from retrieval. Each compensates for the other's weakness.

---

## Why not just use ChatGPT

| | ChatGPT | Noel's GPT |
|---|---|---|
| Knows your documents | ✗ | ✓ via RAG |
| Writes in your voice | ✗ | ✓ via fine-tuning |
| Domain depth | Generic | Trained on domain data |
| Runs locally | ✗ | ✓ fully offline |
| Data privacy | Sent to OpenAI | Never leaves your Mac |
| Cost per query | API pricing | Zero |
| Customisable | Prompt only | Fully — model + data |

---

## What was built

Every component was built from first principles — not assembled from tutorials.

**Transformer architecture from scratch**
Token and positional embeddings, causal self-attention with 12 heads, feed-forward layers with GELU, residual connections, layer normalisation, output projection. 163 million parameters. Every line written and understood.

**Character-level training loop**
A GPT trained from random weights on domain writing. 80-token vocabulary, 5,000 steps, Apple Silicon MPS. Built to understand the full training loop before touching pre-trained models.

**GPT-2 fine-tuning**
GPT-2 fine-tuned on 281,933 characters of fraud and security writing. Learning rate 2e-5 to preserve pre-trained knowledge while adapting to domain voice.

**RAG pipeline**
PDF ingestion → 200-word chunking with overlap → semantic embedding via `all-MiniLM-L6-v2` → ChromaDB storage → cosine similarity retrieval at query time.

**LLaMA 3 integration**
Generation model upgraded to LLaMA 3 (8B parameters) via Ollama. Same RAG pipeline. Dramatically better coherence and output quality.

---

## Stack

```
PyTorch              model architecture and training
Hugging Face         GPT-2 fine-tuning and tokenization
LLaMA 3 via Ollama   generation
ChromaDB             vector storage and retrieval
Sentence Transformers semantic embedding
Flask                local web UI
PyMuPDF              PDF ingestion
```

---

## Setup

**Requirements**
- Python 3.10+
- [Ollama](https://ollama.com) installed
- Apple Silicon Mac recommended (MPS acceleration)

**Install**
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch transformers datasets accelerate
pip install chromadb sentence-transformers pymupdf flask pypdf
ollama pull llama3
```

**Run**
```bash
python3 convert.py      # convert your PDF corpus to text
python3 finetune.py     # fine-tune on your writing
ollama serve            # start LLaMA 3 (new terminal tab)
python3 app.py          # launch the UI
```

Open `http://127.0.0.1:8080`

---

## Extend it

**Add documents to your knowledge base**
```python
ingest_pdf("your_research.pdf")
```
Drop any PDF in. The system chunks, embeds, and indexes it immediately.

**Train on your own writing**
Replace the training corpus with your own text. Run `finetune.py`. The model adapts to your voice.

**Swap the generation model**
```python
"model": "mistral"      # or llama3:70b, gemma, phi3
```
Any model available via Ollama works as a drop-in replacement.

---

## Project structure

```
gpt.py          GPT architecture from scratch
train.py        Character-level training loop
finetune.py     GPT-2 fine-tuning pipeline
rag.py          RAG pipeline with ChromaDB
app.py          Flask UI — LLaMA 3 + RAG
convert.py      PDF to text converter
```

---

## Author

**Noel Rajakumar PS**

Fraud PM. Builder. Chennai.

[LinkedIn](https://www.linkedin.com/in/noelrajakumarps)