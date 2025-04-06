from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from txtgen import (
    extract_text_all_pages,
    get_context_for_keyword,
    query_openai,
    perform_ocr,
    tag_important_points,
    generate_quiz_questions,
    generate_explanations_from_pdf
)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
pdf_text_cache = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Perform OCR or extract normally
    if filename.lower().endswith('.pdf'):
        extracted_text = extract_text_all_pages(filepath)
        if all(not text.strip() for text in extracted_text.values()):
            extracted_text = perform_ocr(filepath)
    else:
        return jsonify({'error': 'Unsupported file format'}), 400

    pdf_text_cache[filename] = extracted_text
    return jsonify({'message': 'File uploaded successfully', 'filename': filename})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.form
    filename = data.get('filename')
    topic = data.get('topic')
    page = data.get('page')
    if not filename or not topic or not page:
        return jsonify({'error': 'Missing fields'}), 400

    try:
        page = int(page)
    except ValueError:
        return jsonify({'error': 'Invalid page number'}), 400

    full_text = pdf_text_cache.get(filename)
    if not full_text:
        return jsonify({'error': 'File not found'}), 404

    explanation = generate_explanations_from_pdf(full_text, topic, page)
    if not explanation or explanation['what'].startswith('No topics found'):
        return jsonify({
            'what': explanation['what'],
            'why': '',
            'how': '',
            'summary': '',
            'external': f'https://en.wikipedia.org/wiki/{topic.replace(" ", "_")}'
        })

    return jsonify(explanation)

@app.route('/important-points', methods=['POST'])
def important_points():
    data = request.form
    filename = data.get('filename')
    page = data.get('page')
    topic = data.get('topic')
    if not filename or not page or not topic:
        return jsonify({'error': 'Missing fields'}), 400

    try:
        page = int(page)
    except ValueError:
        return jsonify({'error': 'Invalid page number'}), 400

    full_text = pdf_text_cache.get(filename)
    if not full_text:
        return jsonify({'error': 'File not found'}), 404

    page_text = full_text.get(page, "")
    points = tag_important_points(page_text, topic)
    return jsonify({'points': points})

@app.route('/generate-quiz', methods=['POST'])
def generate_quiz():
    data = request.form
    filename = data.get('filename')
    page = data.get('page')
    topic = data.get('topic')
    if not filename or not page or not topic:
        return jsonify({'error': 'Missing fields'}), 400

    try:
        page = int(page)
    except ValueError:
        return jsonify({'error': 'Invalid page number'}), 400

    full_text = pdf_text_cache.get(filename)
    if not full_text:
        return jsonify({'error': 'File not found'}), 404

    page_text = full_text.get(page, "")
    quiz = generate_quiz_questions(page_text, topic)
    return jsonify({'quiz': quiz})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)