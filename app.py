import os
import torch
import faiss
import pickle
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline

# --- Streamlit Page Config ---
st.set_page_config(page_title="Bible Q&A", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .st-emotion-cache-1avcm0n {padding: 2rem 3rem;}
    footer {visibility: hidden;}
    .stSidebar {display: none;}
    </style>
    """, unsafe_allow_html=True
)

# --- Session State for Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- File Upload (Admin Only) ---
st.title("📖 Bible Q&A Chat")

# --- Paths & Model Config ---
PDF_PATH =  r"C:\Users\Philip Meshach\Rag_model\The King James Holy Bible.pdf"
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "text_chunks.pkl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --- Load Embedding Model ---
@st.cache_resource
def load_embedder():
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
    return tokenizer, model

embed_tokenizer, embed_model = load_embedder()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

def get_embeddings(texts):
    inputs = embed_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return mean_pooling(outputs, inputs['attention_mask']).cpu().numpy()

# --- Load/Create FAISS Index ---
def process_pdf_and_index():
    reader = PdfReader(PDF_PATH)
    full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_text(full_text)

    embeddings = []
    batch_size = 16
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_embeds = get_embeddings(batch)
        embeddings.append(batch_embeds)

    embeddings = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks

@st.cache_resource
def load_index_and_chunks():
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
    elif os.path.exists(PDF_PATH):
        index, chunks = process_pdf_and_index()
    else:
        return None, None
    return index, chunks

index, chunks = load_index_and_chunks()

# --- Load LLM ---
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")
    return pipe, tokenizer

generator, tokenizer = load_llm()

# --- Ask Question ---
def ask_question(query, k=5, max_context_tokens=1000):
    if not index or not chunks:
        return "Bible data not ready. Please upload the document as admin."

    query_embed = get_embeddings([query])
    D, I = index.search(query_embed.astype("float32"), k)

    selected_chunks = []
    total_tokens = 0
    for i in I[0]:
        chunk = chunks[i]
        tokens = len(tokenizer.encode(chunk))
        if total_tokens + tokens > max_context_tokens:
            break
        selected_chunks.append(chunk)
        total_tokens += tokens

    context = "\n\n".join(selected_chunks)
    prompt = f"""<|system|>You are a Bible scholar. Use the context below to answer the question truthfully. If the answer is not in the context, say you don't know.
<|user|>
Context:
{context}

Question:
{query}
<|assistant|>
Answer:"""

    output = generator(prompt, max_new_tokens=200, do_sample=False, temperature=0.3,
                       pad_token_id=tokenizer.eos_token_id)[0]["generated_text"]

    if "<|assistant|>" in output:
        answer_part = output.split("<|assistant|>")[-1].strip()
        if answer_part.lower().startswith("answer:"):
            return answer_part[7:].strip()
        return answer_part.strip()
    return output.strip()

# --- Chat Interface ---
with st.container():
    for msg in st.session_state.messages:
        role, text = msg["role"], msg["text"]
        st.chat_message(role).write(text)

    query = st.chat_input("Ask a Bible question...")
    if query:
        st.session_state.messages.append({"role": "user", "text": query})
        st.chat_message("user").write(query)

        answer = ask_question(query)
        st.session_state.messages.append({"role": "assistant", "text": answer})
        st.chat_message("assistant").write(answer)


