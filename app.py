import os
import json
import urllib.parse
import requests
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk

# =========================================
# INITIAL SETUP
# =========================================
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
MANIFEST_URL = "https://raw.githubusercontent.com/rydrbot/test/main/file_manifest.json"
JSON_BASE_URL = "https://raw.githubusercontent.com/rydrbot/go-search-app-final/main/json_files"
INDEX_PATH = "go_index_v2.faiss"
META_PATH = "metadata_v2.json"

# =========================================
# LOAD CORE COMPONENTS
# =========================================
@st.cache_resource
def load_components():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model = SentenceTransformer(MODEL_NAME)
    # Load manifest
    resp = requests.get(MANIFEST_URL)
    if resp.status_code != 200:
        st.error("⚠️ Could not load file_manifest.json from GitHub.")
        return index, metadata, model, {}
    manifest = resp.json()
    return index, metadata, model, manifest

index, metadata, model, manifest = load_components()

# =========================================
# SEARCH FUNCTION
# =========================================
def search(query, top_k=5):
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    similarities, indices = index.search(query_emb.astype("float32"), top_k)
    results = []
    for idx, sim in zip(indices[0], similarities[0]):
        if idx >= len(metadata):
            continue
        doc = metadata[idx]
        pdf_file = doc.get("file_name", "")
        pdf_file_encoded = urllib.parse.quote(pdf_file)
        pdf_link = f"{JSDELIVR_BASE}/{pdf_file_encoded}"
        results.append({
            "file_name": pdf_file,
            "similarity": round(float(sim), 4),
            "pdf_link": pdf_link
        })
    return results

# =========================================
# LOAD TEXT FROM JSON USING MANIFEST
# =========================================
def get_text_from_manifest(file_name):
    # Find manifest entry matching this PDF
    entry = next((v for v in manifest.values() if v["pdf"].lower() == file_name.lower()), None)
    if not entry or not entry.get("json"):
        return None, f"⚠️ No JSON linked for '{file_name}' in manifest."

    json_url = f"{JSON_BASE_URL}/{urllib.parse.quote(entry['json'])}"
    resp = requests.get(json_url)
    if resp.status_code != 200:
        return None, f"⚠️ Could not fetch JSON from {json_url}"

    try:
        data = json.loads(resp.text)
        pages = data.get("pages", [])
        combined_text = " ".join([
            p.get("translated_text", "") or p.get("original_text", "")
            for p in pages if p.get("translated_text") or p.get("original_text")
        ])
        if not combined_text.strip():
            return None, "⚠️ JSON found, but contains no readable text."
        return combined_text.strip(), f"✅ Using JSON: {entry['json']}"
    except Exception as e:
        return None, f"⚠️ JSON parsing error: {e}"

# =========================================
# SUMMARIZER
# =========================================
def summarize_text(text, sentence_count=7):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    summary = " ".join(str(s) for s in summary_sentences)
    return summary if summary else text[:800]

# =========================================
# STREAMLIT UI
# =========================================
st.set_page_config(page_title="GO Search – Final", layout="wide")

st.title("📑 Government Order Semantic Search (Final Version)")
st.markdown(
    "Search across Government Orders with instant, factual summaries loaded from preprocessed JSON files. "
    "No OCR or translation delay."
)

query = st.text_input("Enter your search query (English):", "")
top_k = st.slider("Number of results:", 1, 10, 5)

st.sidebar.header("📄 Document Summary")
st.sidebar.info("Click 🧠 to generate instant summary from linked JSON text.")

# =========================================
# SEARCH + SUMMARIZE FLOW
# =========================================
if query:
    results = search(query, top_k=top_k)
    st.write(f"### 🔎 Results for: `{query}`")

    for i, r in enumerate(results, 1):
        with st.container():
            st.markdown(f"**📄 {i}. [{r['file_name']}]({r['pdf_link']})**")
            st.markdown(f"**Similarity:** {r['similarity']:.4f}")

            if st.button(f"🧠 Summarize {r['file_name']}", key=f"btn_{i}"):
                with st.spinner("Fetching JSON text..."):
                    text, info = get_text_from_manifest(r["file_name"])
                    if not text:
                        st.sidebar.warning(info)
                    else:
                        summary = summarize_text(text)
                        st.sidebar.markdown(f"### {r['file_name']}")
                        st.sidebar.success(info)
                        st.sidebar.write(summary)

            st.markdown("---")
