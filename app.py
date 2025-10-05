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

# --- Update these two if you move repos later ---
GITHUB_USER = "rydrbot"
GITHUB_REPO = "test"   # repo where file_manifest.json and json_files live
PDF_REPO = "go-search-app-final"  # repo hosting PDFs for jsDelivr links
# -------------------------------------------------

# URLs auto-adjust for main/master branch
BASE_RAW = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}"
JSDELIVR_BASE = f"https://cdn.jsdelivr.net/gh/{GITHUB_USER}/{PDF_REPO}@main/pdfs"

INDEX_PATH = "go_index_v2.faiss"
META_PATH = "metadata_v2.json"

# =========================================
# LOAD MANIFEST SAFELY
# =========================================
def load_manifest():
    """Try to load file_manifest.json from main or master branch."""
    for branch in ["main", "master"]:
        url = f"{BASE_RAW}/{branch}/file_manifest.json"
        r = requests.get(url)
        if r.status_code == 200:
            st.sidebar.success(f"✅ Loaded manifest from {branch} branch.")
            return r.json()
    st.sidebar.error("⚠️ Could not load file_manifest.json from GitHub.\n"
                     "Check that the repo is public and the file is in the root.")
    return {}

# =========================================
# LOAD CORE COMPONENTS
# =========================================
@st.cache_resource
def load_components():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model = SentenceTransformer(MODEL_NAME)
    manifest = load_manifest()
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
    if not manifest:
        return None, "⚠️ Manifest not loaded."

    # Match PDF name to manifest entry
    entry = next((v for v in manifest.values() if v["pdf"].lower() == file_name.lower()), None)
    if not entry or not entry.get("json"):
        return None, f"⚠️ No JSON linked for '{file_name}' in manifest."

    # Build JSON URL and try both branches
    for branch in ["main", "master"]:
        json_url = f"{BASE_RAW}/{branch}/json_files/{urllib.parse.quote(entry['json'])}"
        resp = requests.get(json_url)
        if resp.status_code == 200:
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

    return None, f"⚠️ JSON file for '{file_name}' not found on GitHub."

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
st.set_page_config(page_title="GO Search – Final (Auto Manifest)", layout="wide")

st.title("📑 Government Order Semantic Search – Auto Manifest")
st.markdown(
    "Search across Government Orders with instant, factual summaries using preprocessed JSON files. "
    "The app automatically detects the correct GitHub branch for your data repository."
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
