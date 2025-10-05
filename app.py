import os
import json
import streamlit as st
import numpy as np
import faiss
import urllib.parse
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from tqdm import tqdm
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

# =========================================
# CONFIG
# =========================================
MODEL_NAME = "all-MiniLM-L6-v2"
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
    return index, metadata, model

index, metadata, model = load_components()

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
# FACTUAL (LEXRANK) SUMMARIZER
# =========================================
def factual_summary(text, sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in summary_sentences)

# =========================================
# STREAMLIT UI
# =========================================
st.set_page_config(page_title="GO Search (v6)", layout="wide")

st.title("📑 Government Order Semantic Search (v6)")
st.write("Search across Government Orders — accurate results, factual summaries, and instant updates.")

query = st.text_input("Enter your search query (English):", "")
top_k = st.slider("Number of results:", 1, 10, 3)

# Sidebar placeholder
st.sidebar.header("📄 Document Summary")
st.sidebar.info("Click '🧠 Summarize' under a result to view its factual summary here.")

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

                st.session_state.summary_text = factual_summary(combined_text)
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

        new_vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        index.add(new_vecs)
        faiss.write_index(index, INDEX_PATH)
        metadata.extend(new_meta)

        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    st.success(f"✅ {uploaded_pdf.name} added to index successfully!")
