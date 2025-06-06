import fitz
import pytesseract
from PIL import Image
import io
import openai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
from pdf2image import convert_from_path
from openai_client import client

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_all_pages(pdf_path):
    text_by_page = {}
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                text_by_page[i + 1] = text or ""
    except Exception:
        return perform_ocr(pdf_path)
    return text_by_page

def perform_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    ocr_text_by_page = {}
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        ocr_text_by_page[i + 1] = text
    return ocr_text_by_page

def get_context_for_keyword(text_by_page, keyword, page):
    page_text = text_by_page.get(page, "")
    if not page_text:
        return None

    chunks = [page_text[i:i+300] for i in range(0, len(page_text), 300)]
    if not chunks:
        return None

    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    keyword_embedding = model.encode([keyword])
    D, I = index.search(np.array(keyword_embedding), k=3)

    relevant_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
    return "\n".join(relevant_chunks)

def query_openai(prompt):
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for students studying from PDF textbooks."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()

def tag_important_points(text):
    prompt = f"List the 5 most important points in bullet format from this text:\n{text}"
    return query_openai(prompt)

def generate_quiz_questions(text):
    prompt = f"Generate 5 multiple-choice quiz questions from the following text:\n{text}"
    return query_openai(prompt)
