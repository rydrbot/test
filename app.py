import os
import json
import streamlit as st
import numpy as np
import faiss
import urllib.parse
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from tqdm import tqdm

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
# SIDEBAR – DOCUMENT SUMMARIES
# =========================================
st.sidebar.header("📄 Document Summaries")
if metadata:
    doc_names = sorted(list(set([d["file_name"] for d in metadata])))
    selected_doc = st.sidebar.selectbox("Select a document to view summary:", doc_names)

    if selected_doc:
        doc_entries = [d for d in metadata if d["file_name"] == selected_doc]
        st.sidebar.markdown(f"**Total Segments:** {len(doc_entries)}")
        st.sidebar.markdown(f"**Languages:** {', '.join(set([d.get('language', 'unknown') for d in doc_entries]))}")

        # Simple summary by taking most frequent key sentences
        combined_text = " ".join([t.get("text", "") for t in doc_entries[:20]])
        summary = combined_text[:700] + "..." if len(combined_text) > 700 else combined_text
        st.sidebar.markdown("**Summary Preview:**")
        st.sidebar.write(summary if summary else "No summary available.")


# =========================================
# MAIN SEARCH AREA
# =========================================
st.set_page_config(page_title="GO Search (v3)", layout="wide")
st.title("📑 Government Order Semantic Search (v3)")
st.write("Enhanced version — Upload new documents, search instantly, and explore summaries.")

query = st.text_input("Enter your search query (English):", "")
top_k = st.slider("Number of results:", 1, 10, 3)

if query:
    results = search(query, top_k=top_k)
    st.write(f"### 🔎 Results for: `{query}`")
    for r in results:
        with st.container():
            st.markdown(f"**📄 File:** {r['file_name']} | **Page:** {r['page_number']}")
            st.markdown(f"**Similarity:** {r['similarity']}")
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
                "language": "en"
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
