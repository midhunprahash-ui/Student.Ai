import PyPDF2
import pytesseract
from PIL import Image
import pdf2image
import openai
import os

def extract_text_all_pages(pdf_path):
    text_by_page = {}
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text = pdf_reader.pages[page_num].extract_text()
                text_by_page[page_num + 1] = text
    except Exception as e:
        print(f"Error extracting text: {e}")
        text_by_page[1] = ""
    return text_by_page

def perform_ocr(pdf_path):
    text_by_page = {}
    try:
        images = pdf2image.convert_from_path(pdf_path)
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            text_by_page[i + 1] = text
    except Exception as e:
        print(f"Error performing OCR: {e}")
        text_by_page[1] = ""
    return text_by_page

def get_context_for_keyword(full_text, keyword, page):
    if page not in full_text:
        return None
    text = full_text[page]
    if keyword.lower() not in text.lower():
        return None
    return text

def query_openai(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def tag_important_points(text):
    prompt = f"Extract important points from this text as a list:\n{text}"
    response = query_openai(prompt)
    return response.split('\n')

def generate_quiz_questions(text):
    prompt = f"Generate 5 multiple choice questions with answers based on this text:\n{text}"
    return query_openai(prompt)
