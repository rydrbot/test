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
import pytesseract
from pdf2image import convert_from_path
from langdetect import detect
from googletrans import Translator

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
LOCAL_PDF_FOLDER = "pdfs"
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
# OCR + FACTUAL SUMMARY
# =========================================
def summarize_pdf(pdf_name, sentence_count=7):
    """Extract text (OCR if needed), translate if Malayalam, and summarize."""
    import io, requests
    text = ""

    # Try local copy first
    local_path = os.path.join(LOCAL_PDF_FOLDER, pdf_name)
    pdf_bytes = None
    if os.path.exists(local_path):
        pdf_bytes = open(local_path, "rb").read()
    else:
        url = f"{JSDELIVR_BASE}/{urllib.parse.quote(pdf_name)}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                pdf_bytes = response.content
        except Exception as e:
            st.warning(f"⚠️ Could not fetch PDF: {e}")
            return "⚠️ Unable to access PDF for summary."

    if not pdf_bytes:
        return "⚠️ PDF not found locally or online."

    # Extract text
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # If no text, do OCR
    if not text.strip():
        st.info("Performing OCR on scanned PDF pages... This may take a moment.")
        try:
            images = convert_from_path(local_path) if os.path.exists(local_path) else convert_from_path(io.BytesIO(pdf_bytes))
            for img in images:
                ocr_text = pytesseract.image_to_string(img, lang="eng+mal")
                text += ocr_text + "\n"
        except Exception as e:
            return f"⚠️ OCR failed: {e}"

    # Detect language and translate if Malayalam
    try:
        lang = detect(text[:500])
        if lang == "ml":
            st.info("Detected Malayalam text — translating to English for summarization.")
            translator = Translator()
            text = translator.translate(text, src="ml", dest="en").text
    except Exception:
        pass

    if not text.strip():
        return "⚠️ No readable content found in this PDF."

    # Summarize
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in summary_sentences) if summary_sentences else text[:800]

# =========================================
# STREAMLIT UI
# =========================================
st.set_page_config(page_title="GO Search (v8)", layout="wide")

st.title("📑 Government Order Semantic Search (v8)")
st.write("Search across Government Orders — with OCR & Malayalam translation support.")

query = st.text_input("Enter your search query (English):", "")
top_k = st.slider("Number of results:", 1, 10, 3)

st.sidebar.header("📄 Document Summary")
st.sidebar.info("Click '🧠 Summarize PDF' to read, OCR, translate, and summarize the actual file.")

if query:
    results = search(query, top_k=top_k)
    st.write(f"### 🔎 Results for: `{query}`")

    for i, r in enumerate(results, 1):
        with st.container():
            st.markdown(f"**📄 {i}. File:** [{r['file_name']}]({r['pdf_link']})")
            st.markdown(f"**Page:** {r['page_number']} | **Similarity:** {r['similarity']:.4f}")

            if st.button(f"🧠 Summarize PDF: {r['file_name']}", key=f"btn_{i}"):
                with st.spinner("Extracting text (OCR if needed)..."):
                    summary = summarize_pdf(r["file_name"])
                    st.sidebar.markdown(f"### {r['file_name']}")
                    st.sidebar.write(summary)

            st.markdown(f"[📎 Open PDF]({r['pdf_link']})")
            st.markdown("---")
