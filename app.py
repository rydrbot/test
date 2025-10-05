import os
import json
import streamlit as st
import numpy as np
import faiss
import urllib.parse
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PyPDF2 import PdfReader
from tqdm import tqdm

# =========================================
# CONFIG
# =========================================
MODEL_NAME = "all-MiniLM-L6-v2"
SUMMARY_MODEL = "facebook/bart-large-cnn"
JSDELIVR_BASE = "https://cdn.jsdelivr.net/gh/rydrbot/go-search-app-final@main/pdfs"
INDEX_PATH = "go_index_v2.faiss"
META_PATH = "metadata_v2.json"

# =========================================
# LOAD COMPONENTS
# =========================================
@st.cache_resource
def load_components():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model = SentenceTransformer(MODEL_NAME)
    summarizer = pipeline("summarization", model=SUMMARY_MODEL)
    return index, metadata, model, summarizer

index, metadata, model, summarizer = load_components()

# =========================================
# SEARCH FUNCTION
# =========================================
def search(query, top_k=5):
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    similarities, indices = index.search(query_emb.astype("float32"), top_k)

    results = []
    for idx, sim in zip(indices[0], similarities[0]):
        if idx >= len(metadata):
            continue
        doc = metadata[idx]
        pdf_file = doc.get("file_name", "").replace("_raw.txt", ".pdf")
        pdf_file_encoded = urllib.parse.quote(pdf_file)
        pdf_link = f"{JSDELIVR_BASE}/{pdf_file_encoded}"

        results.append({
            "file_name": pdf_file,
            "page_number": doc.get("page_number", ""),
            "similarity": round(float(sim), 4),
            "pdf_link": pdf_link
        })
    return results


# =========================================
# SEMANTIC SUMMARIZER
# =========================================
def semantic_summary(text, summarizer, max_tokens=800):
    # Split long text into manageable chunks (~1000 tokens each)
    parts = []
    sentences = text.split(". ")
    chunk, count = "", 0
    for s in sentences:
        chunk += s + ". "
        count += len(s.split())
        if count > 600:
            parts.append(chunk)
            chunk, count = "", 0
    if chunk:
        parts.append(chunk)

    summaries = []
    for part in parts[:3]:  # Limit to 3 chunks for performance
        try:
            summary = summarizer(part, max_length=130, min_length=40, do_sample=False)[0]["summary_text"]
            summaries.append(summary)
        except Exception as e:
            continue
    return " ".join(summaries) if summaries else text[:600]


# =========================================
# STREAMLIT UI
# =========================================
st.set_page_config(page_title="GO Search (v5)", layout="wide")

st.title("📑 Government Order Semantic Search (v5)")
st.write("Search across Government Orders — with semantic summaries and dynamic sidebar context.")

query = st.text_input("Enter your search query (English):", "")
top_k = st.slider("Number of results:", 1, 10, 3)

# Sidebar placeholder
st.sidebar.header("📄 Document Summary")
st.sidebar.info("Click '🧠 Summarize' under a result to view its summary here.")

# Session state
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""

# =========================================
# SEARCH RESULTS + INTERACTION
# =========================================
if query:
    results = search(query, top_k=top_k)
    st.write(f"### 🔎 Results for: `{query}`")

    for i, r in enumerate(results, 1):
        with st.container():
            st.markdown(f"**📄 {i}. File:** [{r['file_name']}]({r['pdf_link']})")
            st.markdown(f"**Page:** {r['page_number']} | **Similarity:** {r['similarity']:.4f}")

            if st.button(f"🧠 Summarize {r['file_name']}", key=f"btn_{i}"):
                st.session_state.selected_doc = r['file_name']
                doc_chunks = [d for d in metadata if d["file_name"] == r["file_name"]]
                combined_text = " ".join(
                    [t.get("text", "") or t.get("translated_text", "") or "" for t in doc_chunks]
                )
                st.session_state.summary_text = semantic_summary(combined_text, summarizer)
                st.sidebar.markdown(f"### {r['file_name']}")
                st.sidebar.write(st.session_state.summary_text)

            st.markdown(f"[📎 Open PDF]({r['pdf_link']})")
            st.markdown("---")


# =========================================
# UPLOAD + INCREMENTAL UPDATE
# =========================================
st.subheader("📤 Add New Document to Index")

uploaded_pdf = st.file_uploader("Upload a new PDF document", type=["pdf"])
if uploaded_pdf is not None:
    with st.spinner("Processing uploaded PDF..."):
        reader = PdfReader(uploaded_pdf)
        texts, new_meta = [], []

        for i, page in enumerate(tqdm(reader.pages)):
            text = page.extract_text()
            if not text or len(text.strip()) < 50:
                continue
            texts.append(text.strip())
            new_meta.append({
                "file_name": uploaded_pdf.name,
                "category": "uploaded",
                "page_number": i + 1,
                "language": "en",
                "text": text.strip()
            })

        # Generate embeddings
        new_vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

        # Append to FAISS index
        index.add(new_vecs)
        faiss.write_index(index, INDEX_PATH)

        # Append to metadata
        metadata.extend(new_meta)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    st.success(f"✅ {uploaded_pdf.name} added to index successfully!")
