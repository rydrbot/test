import os
import io
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
from pdf2image import convert_from_bytes
from langdetect import detect
from deep_translator import GoogleTranslator
import requests

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
# OCR + TRANSLATION + SUMMARY (ERROR SAFE)
# =========================================
def summarize_pdf(pdf_name, sentence_count=7):
    """Extract text (OCR if needed), translate if Malayalam, and summarize."""
    text = ""
    pdf_bytes = None

    # Try local copy
    local_path = os.path.join(LOCAL_PDF_FOLDER, pdf_name)
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            pdf_bytes = f.read()
    else:
        # Try fetching from CDN
        url = f"{JSDELIVR_BASE}/{urllib.parse.quote(pdf_name)}"
        resp = requests.get(url)
        if resp.status_code == 200:
            pdf_bytes = resp.content
        else:
            return f"⚠️ Unable to fetch {pdf_name} from source."

    # Try extracting text normally
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    except Exception:
        pass

    # OCR fallback if no text
    if not text.strip():
        st.info("Performing OCR on scanned PDF pages... Please wait ⏳")
        try:
            images = convert_from_bytes(pdf_bytes)
            total_pages = len(images)
            for i, img in enumerate(images, start=1):
                ocr_text = pytesseract.image_to_string(img, lang="eng+mal")
                text += ocr_text + "\n"
                st.progress(i / total_pages)
        except Exception as e:
            # Poppler not installed or OCR failed
            if "poppler" in str(e).lower() or "page count" in str(e).lower():
                return ("⚠️ OCR unavailable: Poppler is not installed.\n\n"
                        "➡️ To enable OCR:\n"
                        "- **Colab/Ubuntu:** `sudo apt-get install poppler-utils`\n"
                        "- **Streamlit Cloud:** create a `packages.txt` file with `poppler-utils`\n"
                        "- **Windows:** install Poppler and add to PATH.\n")
            else:
                return f"⚠️ OCR failed: {e}"

    if not text.strip():
        return "⚠️ No readable text extracted from this PDF."

    # Detect language and translate if Malayalam
    try:
        lang = detect(text[:500])
        if lang == "ml":
            st.info("Detected Malayalam — translating to English for summarization.")
            translator = GoogleTranslator(source="auto", target="en")
            text = translator.translate(text)
    except Exception:
        pass

    # Summarize using LexRank
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary_sentences = summarizer(parser.document, sentence_count)
        summary = " ".join(str(s) for s in summary_sentences)
        return summary if summary else text[:800]
    except Exception as e:
        return f"⚠️ Summarization failed: {e}"

# =========================================
# STREAMLIT UI
# =========================================
st.set_page_config(page_title="GO Search", layout="wide")

st.title("📑 Government Order Search")
st.write("Kerala State IT Mission")

query = st.text_input("Enter your search query (English):", "")
top_k = st.slider("Number of results:", 1, 10, 3)

st.sidebar.header("📄 Document Summary")
st.sidebar.info("Click 🧠 to summarize selected PDF (works for scanned or Malayalam files).")

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
