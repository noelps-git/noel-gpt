from flask import Flask, request, jsonify, render_template_string
import torch
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("./gosense-gpt-final")
model = AutoModelForCausalLM.from_pretrained("./gosense-gpt-final")
model = model.to(device)
model.eval()

client = chromadb.Client()
collection = client.get_or_create_collection("gosense_knowledge")

def chunk_text(text, chunk_size=200, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks

def ingest_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    chunks = chunk_text(text)
    embeddings = embed_model.encode(chunks)
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"Ingested {len(chunks)} chunks")

def retrieve(query, n=2):
    query_embedding = embed_model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n
    )
    return results["documents"][0]

def rag_generate(query, max_new_tokens=150, temperature=0.8):
    context_chunks = retrieve(query)
    context = "\n\n".join(context_chunks)[:800]
    prompt = f"""Context:
{context}

Write about: {query}
---
"""
    inputs = tokenizer.encode(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=800
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=150,
            temperature=temperature,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output[len(prompt):]

ingest_pdf("training data for posts.pdf")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Noel's GPT</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0a0a0a;
    color: #f0f0f0;
    font-family: 'Georgia', serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 60px 20px;
  }
  .header {
    text-align: center;
    margin-bottom: 60px;
  }
  .header h1 {
    font-size: 28px;
    font-weight: normal;
    letter-spacing: 0.1em;
    color: #ffffff;
    margin-bottom: 8px;
  }
  .header p {
    font-size: 14px;
    color: #666;
    font-family: 'Helvetica Neue', sans-serif;
    letter-spacing: 0.05em;
  }
  .container {
    width: 100%;
    max-width: 700px;
  }
  .presets {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 30px;
    justify-content: center;
  }
  .preset-btn {
    background: transparent;
    border: 1px solid #333;
    color: #999;
    padding: 8px 16px;
    border-radius: 20px;
    cursor: pointer;
    font-size: 13px;
    font-family: 'Helvetica Neue', sans-serif;
    transition: all 0.2s;
  }
  .preset-btn:hover {
    border-color: #666;
    color: #fff;
  }
  .preset-btn.active {
    border-color: #fff;
    color: #fff;
  }
  .input-area {
    display: flex;
    gap: 12px;
    margin-bottom: 40px;
  }
  .input-area input {
    flex: 1;
    background: #111;
    border: 1px solid #222;
    color: #fff;
    padding: 14px 18px;
    border-radius: 8px;
    font-size: 15px;
    font-family: 'Georgia', serif;
    outline: none;
    transition: border-color 0.2s;
  }
  .input-area input:focus {
    border-color: #444;
  }
  .input-area button {
    background: #fff;
    color: #000;
    border: none;
    padding: 14px 24px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    font-family: 'Helvetica Neue', sans-serif;
    font-weight: 500;
    transition: opacity 0.2s;
  }
  .input-area button:hover { opacity: 0.85; }
  .input-area button:disabled { opacity: 0.4; cursor: not-allowed; }
  .output-area {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 30px;
    min-height: 200px;
    display: none;
  }
  .output-area.visible { display: block; }
  .output-label {
    font-size: 11px;
    letter-spacing: 0.15em;
    color: #444;
    font-family: 'Helvetica Neue', sans-serif;
    text-transform: uppercase;
    margin-bottom: 16px;
  }
  .output-text {
    font-size: 16px;
    line-height: 1.8;
    color: #d0d0d0;
  }
  .loading {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #444;
    font-size: 14px;
    font-family: 'Helvetica Neue', sans-serif;
  }
  .dot {
    width: 6px;
    height: 6px;
    background: #444;
    border-radius: 50%;
    animation: pulse 1.2s infinite;
  }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes pulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1; }
  }
  .footer {
    margin-top: 60px;
    font-size: 12px;
    color: #333;
    font-family: 'Helvetica Neue', sans-serif;
    letter-spacing: 0.05em;
  }
</style>
</head>
<body>
<div class="header">
  <h1>Noel's GPT</h1>
  <p>Trained on fraud, security, and payment rails</p>
</div>
<div class="container">
  <div class="presets">
    <button class="preset-btn" onclick="setPreset(this, 'Write about UPI fraud detection')">UPI Fraud</button>
    <button class="preset-btn" onclick="setPreset(this, 'Write about fraud rings and coordinated attacks')">Fraud Rings</button>
    <button class="preset-btn" onclick="setPreset(this, 'Write about behavioral biometrics in fraud detection')">Behavioral Biometrics</button>
    <button class="preset-btn" onclick="setPreset(this, 'Write about SIM swapping attacks')">SIM Swapping</button>
    <button class="preset-btn" onclick="setPreset(this, 'Write about AI agents in fraud operations')">AI Agents</button>
    <button class="preset-btn" onclick="setPreset(this, 'Write about mobile banking security')">Mobile Security</button>
  </div>
  <div class="input-area">
    <input type="text" id="query" placeholder="Ask anything about fraud, payments, or security..." />
    <button id="generateBtn" onclick="runGenerate()">Generate</button>
  </div>
  <div class="output-area" id="output">
    <div class="output-label">Generated</div>
    <div class="output-text" id="outputText"></div>
  </div>
</div>
<div class="footer">Noel · 2026</div>
<script>
  function setPreset(btn, text) {
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('query').value = text;
  }
  function runGenerate() {
    const query = document.getElementById('query').value.trim();
    if (!query) return;
    const btn = document.getElementById('generateBtn');
    const output = document.getElementById('output');
    const outputText = document.getElementById('outputText');
    btn.disabled = true;
    output.classList.add('visible');
    outputText.innerHTML = '<div class="loading"><div class="dot"></div><div class="dot"></div><div class="dot"></div><span>Generating...</span></div>';
    fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    })
    .then(res => res.json())
    .then(data => {
      outputText.innerText = data.response;
      btn.disabled = false;
    })
    .catch(() => {
      outputText.innerText = 'Something went wrong. Try again.';
      btn.disabled = false;
    });
  }
  document.getElementById('query').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') runGenerate();
  });
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({'response': 'Please enter a query.'})
    response = rag_generate(query)
    return jsonify({'response': response})

if __name__ == '__main__':
    print("Starting Noel's GPT...")
    print("Go to http://127.0.0.1:8080 in your browser")
    app.run(debug=False, host='0.0.0.0', port=8080)