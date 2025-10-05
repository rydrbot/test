import os
import json
import streamlit as st
import numpy as np
import faiss
import urllib.parse
from sentence_transformers import SentenceTransformer

# =========================================
# CONFIG
# =========================================
MODEL_NAME = "all-MiniLM-L6-v2"

# ✅ Update your jsDelivr base link (same as old setup)
JSDELIVR_BASE = "https://cdn.jsdelivr.net/gh/rydrbot/go-search-app-final@main/pdfs"

INDEX_PATH = "go_index_v2.faiss"
META_PATH = "metadata_v2.json"

# =========================================
# LOAD INDEX + MODEL
# =========================================
@st.cache_resource
def load_index():
    # Load FAISS index
    index = faiss.read_index(INDEX_PATH)

    # Load metadata
    with open(META_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    return documents, index, model

documents, index, model = load_index()

# =========================================
# SEARCH FUNCTION
# =========================================
def search(query, top_k=5):
    # Encode query
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

    # Search FAISS
    similarities, indices = index.search(query_emb.astype("float32"), top_k)

    results = []
    for idx, sim in zip(indices[0], similarities[0]):
        if idx >= len(documents):
            continue
        doc = documents[idx]

        # Derive file name and encode for URL safety
        pdf_file = doc.get("file_name", "").replace("_raw.txt", ".pdf")
        pdf_file_encoded = urllib.parse.quote(pdf_file)

        # Build jsDelivr link
        pdf_link = f"{JSDELIVR_BASE}/{pdf_file_encoded}"

        results.append({
            "file_name": pdf_file,
            "category": doc.get("category", ""),
            "page_number": doc.get("page_number", ""),
            "language": doc.get("language", ""),
            "similarity": round(float(sim), 4),
            "pdf_link": pdf_link
        })
    return results

# =========================================
# STREAMLIT UI
# =========================================
st.set_page_config(page_title="GO Search", layout="wide")

st.title("📑 Government Order Search ")
st.write("Kerala State IT MIssion")

query = st.text_input("Enter your search query (English):", "")
top_k = st.slider("Number of results:", 1, 10, 3)

if query:
    results = search(query, top_k=top_k)
    st.write(f"### 🔎 Results for: `{query}`")

    for r in results:
        with st.container():
            st.markdown(f"**📄 File:** {r['file_name']} | **Page:** {r['page_number']}")
            st.markdown(f"**Language:** {r['language']} | **Similarity:** {r['similarity']}")
            st.markdown(f"[📎 Open PDF]({r['pdf_link']})")
            st.markdown("---")
