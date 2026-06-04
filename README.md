# Noel's GPT

A GPT trained on fraud, security, and payment rails content.

Built from first principles — character-level GPT from scratch,
GPT-2 fine-tuning, and a RAG pipeline with ChromaDB.

## What's inside

- `gpt.py` — GPT architecture from scratch
- `train.py` — character-level training loop
- `finetune.py` — GPT-2 fine-tuning on your writing
- `rag.py` — RAG pipeline with ChromaDB
- `app.py` — Flask UI to showcase the model
- `convert.py` — PDF to text converter

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch transformers datasets accelerate
pip install chromadb sentence-transformers pymupdf flask pypdf
```

## Run

```bash
# Convert your PDF to text
python3 convert.py

# Fine-tune GPT-2 on your writing
python3 finetune.py

# Launch the UI
python3 app.py
```

## Built with

- PyTorch
- Hugging Face Transformers
- ChromaDB
- Sentence Transformers
- Flask

## Author

Noel Rajakumar PS · GoSense AI · 2026