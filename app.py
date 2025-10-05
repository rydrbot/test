# ===========================================
# app_v2_with_links.py
# ===========================================
# Streamlit app for intelligent GO Search
# Hugging Face Embeddings + FAISS + PDF Links
# ===========================================

import streamlit as st
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# ========== CONFIGURATION ==========
INDEX_PATH = "go_index_v2.faiss"
META_PATH = "metadata_v2.json"
MODEL_NAME = "all-MiniLM-L6-v2"
PDF_FOLDER = "pdfs"  # local folder containing the PDF files
GITHUB_BASE = "https://cdn.jsdelivr.net/gh/rydrbot/go-search-app-final/pdf/"  # optional GitHub link base

# ========== LOAD COMPONENTS ==========
@st.cache_resource
def load_components():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model = SentenceTransformer(MODEL_NAME)
    return index, metadata, model

index, metadata, model = load_components()

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="Intelligent GO Search", layout="wide")
st.title("📘 Intelligent GO Search (v2 - Hugging Face + PDF Links)")
st.markdown("Search across Government Orders with embedded PDF access 🔗")

query = st.text_input("🔍 Enter your search query:")
num_results = st.slider("Number of results", 1, 15, 5)

with st.expander("🎚️ Advanced Filters"):
    file_filter = st.text_input("Filter by File Name (optional)").strip()
    category_filter = st.text_input("Filter by Category (optional)").strip()
    lang_filter = st.text_input("Filter by Language (optional)").strip()

# ========== SEARCH ENGINE ==========
if query:
    st.markdown("### ⏳ Searching... Please wait.")
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    distances, indices = index.search(q_vec, num_results * 2)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx >= len(metadata):
            continue
        meta = metadata[idx]

        # Apply filters
        if file_filter and file_filter.lower() not in meta["file_name"].lower():
            continue
        if category_filter and category_filter.lower() not in meta["category"].lower():
            continue
        if lang_filter and lang_filter.lower() not in meta["language"].lower():
            continue

        # PDF link logic
        file_name = meta["file_name"]
        local_path = os.path.join(PDF_FOLDER, file_name)
        if os.path.exists(local_path):
            pdf_link = f"./{PDF_FOLDER}/{file_name}"
        else:
            # fallback to GitHub CDN link if hosted
            pdf_link = f"{GITHUB_BASE}{file_name.replace(' ', '%20')}"

        results.append({
            "score": float(dist),
            "file_name": file_name,
            "category": meta["category"],
            "page_number": meta["page_number"],
            "language": meta["language"],
            "pdf_link": pdf_link
        })

        if len(results) >= num_results:
            break

    # ========== DISPLAY RESULTS ==========
    if results:
        st.success(f"✅ Found {len(results)} relevant results for your query.")
        for i, res in enumerate(results, 1):
            st.markdown(f"### {i}. [{res['file_name']}]({res['pdf_link']})")
            st.markdown(f"- **Category:** {res['category']}")
            st.markdown(f"- **Page:** {res['page_number']}")
            st.markdown(f"- **Language:** {res['language']}")
            st.markdown(f"- **Similarity:** {round(res['score'], 4)}")
            st.divider()
    else:
        st.warning("No matching results found. Try another query or adjust filters.")
