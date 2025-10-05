import os
import json
import streamlit as st
import numpy as np
import faiss
import urllib.parse
from sentence_transformers import SentenceTransformer
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
JSON_FOLDER = "/content/drive/MyDrive/ICT_Project_IntelligentGOSearch/json_files"  # change if needed
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
# LOAD JSON FOR SELECTED DOCUMENT
# =========================================
def get_text_from_json(file_name):
    """Fetch text from JSON stored on GitHub (remote)."""
    base_url = "https://raw.githubusercontent.com/rydrbot/go-search-app-final/main/json_files"
    file_base = file_name.lower().replace(".pdf", "")

    import requests

    # Try to find a matching JSON file remotely
    possible_url = f"{base_url}/{urllib.parse.quote(file_name.replace('.pdf', '.json'))}"
    alt_url = f"{base_url}/{urllib.parse.quote(file_base)}.json"

    for url in [possible_url, alt_url]:
        resp = requests.get(url)
        if resp.status_code == 200:
            try:
                data = json.loads(resp.text)
                pages = data.get("pages", [])
                combined_text = " ".join([
                    p.get("translated_text", "") or p.get("original_text", "")
                    for p in pages if p.get("translated_text") or p.get("original_text")
                ])
                return combined_text.strip(), None
            except Exception as e:
                return None, f"⚠️ JSON parsing error: {e}"

    return None, f"⚠️ JSON file for '{file_name}' not found at {base_url}."


# =========================================
# LEXRANK FACTUAL SUMMARY
# =========================================
def factual_summary(text, sentence_count=7):
    if not text.strip():
        return "⚠️ No text available for summarization."
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in summary_sentences) or text[:800]

# =========================================
# STREAMLIT UI
# =========================================
st.set_page_config(page_title="GO Search (v11)", layout="wide")

st.title("📑 Government Order Semantic Search (v11)")
st.write("Now with instant JSON-based summaries (no OCR or translation delay).")

query = st.text_input("Enter your search query (English):", "")
top_k = st.slider("Number of results:", 1, 10, 3)

st.sidebar.header("📄 Document Summary")
st.sidebar.info("Click 🧠 to generate summary instantly from preprocessed JSON data.")

if query:
    results = search(query, top_k=top_k)
    st.write(f"### 🔎 Results for: `{query}`")

    for i, r in enumerate(results, 1):
        with st.container():
            st.markdown(f"**📄 {i}. File:** [{r['file_name']}]({r['pdf_link']})")
            st.markdown(f"**Page:** {r['page_number']} | **Similarity:** {r['similarity']:.4f}")

            if st.button(f"🧠 Summarize: {r['file_name']}", key=f"btn_{i}"):
                with st.spinner("Loading preprocessed text..."):
                    text, err = get_text_from_json(r["file_name"])
                    if err:
                        st.sidebar.warning(err)
                    else:
                        summary = factual_summary(text)
                        st.sidebar.markdown(f"### {r['file_name']}")
                        st.sidebar.write(summary)

            st.markdown(f"[📎 Open PDF]({r['pdf_link']})")
            st.markdown("---")
