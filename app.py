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
from difflib import get_close_matches

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

# Your GitHub setup
GITHUB_USER = "rydrbot"
GITHUB_REPO = "test"               # repo where file_manifest.json + json_files live
PDF_REPO = "go-search-app-final"   # repo hosting PDFs for jsDelivr access

BASE_RAW = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}"
BASE_API = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/json_files"
JSDELIVR_BASE = f"https://cdn.jsdelivr.net/gh/{GITHUB_USER}/{PDF_REPO}@main/pdfs"

INDEX_PATH = "go_index_v2.faiss"
META_PATH = "metadata_v2.json"

# =========================================
# LOAD MANIFEST
# =========================================
def load_manifest():
    """Try to load file_manifest.json from main or master."""
    for branch in ["main", "master"]:
        url = f"{BASE_RAW}/{branch}/file_manifest.json"
        r = requests.get(url)
        if r.status_code == 200:
            st.sidebar.success(f"✅ Loaded manifest from {branch} branch.")
            return r.json()
    st.sidebar.error("⚠️ Could not load file_manifest.json from GitHub. "
                     "Check that the repo is public and the file is in the root directory.")
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
# LOAD TEXT FROM JSON USING MANIFEST + FALLBACK
# =========================================
def get_text_from_manifest(file_name):
    """Fetch JSON text via manifest; fall back to fuzzy GitHub match if needed."""
    if not manifest:
        return None, "⚠️ Manifest not loaded."

    entry = next((v for v in manifest.values() if v["pdf"].lower() == file_name.lower()), None)
    if not entry:
        return None, f"⚠️ No manifest entry for '{file_name}'"

    expected_json = entry.get("json")
    if not expected_json:
        return None, f"⚠️ No JSON name listed for '{file_name}' in manifest."

    # Try main and master branches
    for branch in ["main", "master"]:
        json_url = f"{BASE_RAW}/{branch}/json_files/{urllib.parse.quote(expected_json)}"
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
                    return None, "⚠️ JSON found but contains no readable text."
                return combined_text.strip(), f"✅ Using JSON: {expected_json}"
            except Exception as e:
                return None, f"⚠️ JSON parsing error: {e}"

    # Fallback: fuzzy match JSON filename using GitHub API
    alt_resp = requests.get(BASE_API)
    if alt_resp.status_code == 200:
        try:
            json_list = [item["name"] for item in alt_resp.json() if item["name"].endswith(".json")]
            matches = get_close_matches(expected_json.lower(), [j.lower() for j in json_list], n=1, cutoff=0.5)
            if matches:
                matched_name = next(j for j in json_list if j.lower() == matches[0])
                for branch in ["main", "master"]:
                    alt_url = f"{BASE_RAW}/{branch}/json_files/{urllib.parse.quote(matched_name)}"
                    r2 = requests.get(alt_url)
                    if r2.status_code == 200:
                        data = json.loads(r2.text)
                        pages = data.get("pages", [])
                        combined_text = " ".join([
                            p.get("translated_text", "") or p.get("original_text", "")
                            for p in pages if p.get("translated_text") or p.get("original_text")
                        ])
                        return combined_text.strip(), f"✅ Matched JSON: {matched_name}"
        except Exception as e:
            return None, f"⚠️ Fallback JSON fetch failed: {e}"

    return None, f"⚠️ JSON not found for '{file_name}' (tried {expected_json})"

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
st.set_page_config(page_title="GO Search – Final (Auto-Fix JSON)", layout="wide")

st.title("📑 Government Order Semantic Search – Smart JSON Matching")
st.markdown(
    "Search across Government Orders with instant summaries from preprocessed JSON files. "
    "Automatically finds correct JSON even if filenames differ slightly."
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
