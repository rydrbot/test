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

# Ensure tokenizer data is present
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
LOCAL_PDF_FOLDER = "pdfs"  # if you also keep local copies
INDEX_PATH = "go_index_v2.faiss"
META_PATH = "metadata_v2.json"

# =========================================
# LOAD INDEX + MODEL
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
# FACTUAL SUMMARY (LEXRANK)
# =========================================
def summarize_pdf(pdf_name, sentence_count=7):
    """Read a PDF (local or via CDN) and produce a factual summary."""
    text = ""
    local_path = os.path.join(LOCAL_PDF_FOLDER, pdf_name)
    if os.path.exists(local_path):
        reader = PdfReader(local_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    else:
        # Optional: download from jsDelivr temporarily
        import requests, io
        try:
            url = f"{JSDELIVR_BASE}/{urllib.parse.quote(pdf_name)}"
            response = requests.get(url)
            if response.status_code == 200:
                reader = PdfReader(io.BytesIO(response.content))
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            st.warning(f"Could not fetch PDF: {e}")
            return "⚠️ Unable to access PDF content for summary."

    if not text.strip():
        return "⚠️ No text could be extracted from this PDF."

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in summary_sentences) if summary_sentences else text[:800]

# =========================================
# STREAMLIT UI
# =========================================
st.set_page_config(page_title="GO Search (v7)", layout="wide")

st.title("📑 Government Order Semantic Search (v7)")
st.write("Search across Government Orders and generate instant summaries from the actual PDF text.")

query = st.text_input("Enter your search query (English):", "")
top_k = st.slider("Number of results:", 1, 10, 3)

# Sidebar placeholder
st.sidebar.header("📄 Document Summary")
st.sidebar.info("Click '🧠 Summarize PDF' to read and summarize the actual file.")

if query:
    results = search(query, top_k=top_k)
    st.write(f"### 🔎 Results for: `{query}`")

    for i, r in enumerate(results, 1):
        with st.container():
            st.markdown(f"**📄 {i}. File:** [{r['file_name']}]({r['pdf_link']})")
            st.markdown(f"**Page:** {r['page_number']} | **Similarity:** {r['similarity']:.4f}")

            if st.button(f"🧠 Summarize PDF: {r['file_name']}", key=f"btn_{i}"):
                with st.spinner("Reading and summarizing PDF..."):
                    summary = summarize_pdf(r["file_name"])
                    st.sidebar.markdown(f"### {r['file_name']}")
                    st.sidebar.write(summary)

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
