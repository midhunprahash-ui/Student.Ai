import os
import sys
import io
import threading
import queue
import json
from collections import Counter
import speech_recognition as sr
from gtts import gTTS
import pygame
import time
import glob
import cv2
import face_recognition
from operator import itemgetter

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from flask import Flask, render_template, Response

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GOOGLE_API_KEY"] = " YOUR_GOOGLE_API_KEY_HERE" 

app = Flask(__name__)
log_queue = queue.Queue()

def log_print(*args, **kwargs):
    msg = " ".join(str(arg) for arg in args)
    print(msg, **kwargs) 
    log_queue.put(msg)


builtins_print = print
def print(*args, **kwargs):
    log_print(*args, **kwargs)

class SmartTutor:
    def __init__(self):
        print("Initializing Smart Tutor...")
        self.mode = 'TEACHING'
        self.is_running = True
        self.textbook_file = "jesc105.pdf"
        
        # Vision Component Setup
        self.known_face_encodings = []; self.known_face_names = []
        self.recognized_student_id = None; self.camera_thread = None
        self._load_known_faces()
        
        pygame.mixer.init()
        self.recognizer = sr.Recognizer(); self.recognizer.energy_threshold = 3000
        
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = self._load_vector_db()
        self.retriever = self.db.as_retriever(search_kwargs={"k": 5})
        
        self.tutor_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        self.analyzer_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
        
        self._initialize_chains()
        
        self.topics_queue = []; self.current_topic = None
        self.class_profiles = []; self.cul_profile = {}; self.lesson_sentences = []
        print("Smart Tutor initialized and ready.")

    def speak(self, text):
        if not text or not text.strip():
            return
        log_print(f"Tutor: {text}")
        try:
            tts = gTTS(text=text, lang='en')
            temp_audio_file = "temp_speech.mp3"
            tts.save(temp_audio_file)
            pygame.mixer.music.load(temp_audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
            time.sleep(0.1)
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
        except Exception as e:
            log_print(f"Error in gTTS or pygame playback: {e}")

    def listen(self, source, timeout=5):
        try:
            log_print(f"...Listening...")
            audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=15)
            text = self.recognizer.recognize_google(audio).lower()
            log_print(f"Heard: '{text}'")
            return text
        except (sr.UnknownValueError, sr.WaitTimeoutError):
            return None

    def _load_known_faces(self, faces_directory='student_faces'):
        log_print("Loading and encoding known faces for recognition...")
        if not os.path.exists(faces_directory):
            log_print(f"Warning: '{faces_directory}' not found.")
            return
        for filename in os.listdir(faces_directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(faces_directory, filename)
                name = os.path.splitext(filename)[0].lower()
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(name)
                    log_print(f"- Encoded face for: {name}")

    def _run_camera_recognition(self):
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            log_print("Error: Could not open camera.")
            return
        log_print("[CV Thread] Camera recognition started.")
        while self.is_running:
            ret, frame = video_capture.read()
            if not ret:
                continue
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            if self.mode == 'TEACHING' and face_encodings:
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    if True in matches:
                        name = self.known_face_names[matches.index(True)]
                        if self.recognized_student_id is None:
                            log_print(f"[CV Thread] Recognized student: {name}")
                            self.recognized_student_id = name
                            break
            time.sleep(1)
        video_capture.release()

    def _load_vector_db(self):
        db_name = os.path.splitext(self.textbook_file)[0]
        db_path = f"./faiss_index_{db_name}"
        if os.path.exists(db_path):
            return FAISS.load_local(db_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            loader = PyPDFLoader(self.textbook_file)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            db = FAISS.from_documents(chunks, self.embeddings)
            db.save_local(db_path)
            return db
            
    def _initialize_chains(self):
        cul_template = "You are an AI teacher preparing a lesson on **'{topic}'**. The class's dominant learning style is {dominant_style}. Their common knowledge gaps are: {common_knowledge_gaps}. Start by briefly re-explaining these gap areas, then teach the main topic tailored to the learning style."
        self.cul_chain = PromptTemplate.from_template(cul_template) | self.tutor_llm | StrOutputParser()
        pul_template = "You are a tutor for {student_name}. Their profile: Style: {inferred_style}, Knowledge Gaps: {knowledge_gaps}. Based ONLY on the provided **Textbook Context**, answer their **Question**. Context: {context}. Question: {question}"
        pul_prompt = PromptTemplate.from_template(pul_template)
        self.pul_chain = (
            {
                "context": itemgetter("question") | self.retriever,
                "question": itemgetter("question"),
                "student_name": itemgetter("student_name"),
                "inferred_style": itemgetter("inferred_style"),
                "knowledge_gaps": itemgetter("knowledge_gaps")
            }
            | pul_prompt
            | self.tutor_llm
            | StrOutputParser()
        )
        objectives_template = "For a class, list 2 brief learning objectives for '{topic}'."
        self.objectives_chain = PromptTemplate.from_template(objectives_template) | self.tutor_llm | StrOutputParser()
    
    def create_student_profile(self, student_answer_text: str):
        profile_prompt_template = "Analyze student's answers: --- {student_answers} ---. Generate JSON with keys: `inferred_style`, `style_reasoning`, `knowledge_gaps` (list), `strengths` (list). Respond ONLY with raw JSON."
        profile_prompt = PromptTemplate.from_template(profile_prompt_template)
        profile_chain = profile_prompt | self.analyzer_llm | StrOutputParser()
        log_print("  - Creating detailed student profile...")
        llm_response = profile_chain.invoke({"student_answers": student_answer_text})
        try:
            start_index = llm_response.find('{')
            end_index = llm_response.rfind('}') + 1
            json_string = llm_response[start_index:end_index]
            return json.loads(json_string)
        except json.JSONDecodeError:
            return {"inferred_style": "unknown", "knowledge_gaps": [], "strengths": []}
    
    def analyze_and_prepare_class(self, papers_directory='answer_papers'):
        log_print("\n--- Analyzing Class Answer Papers for CUL/PUL ---")
        pdf_files = glob.glob(os.path.join(papers_directory, '*.pdf'))
        if not pdf_files:
            log_print(f"Warning: No PDF files found in '{papers_directory}'. Using default profiles.")
            self.class_profiles.append({
                "student_id": "aman",
                "inferred_style": "Gist Learner",
                "knowledge_gaps": ["Details of cellular respiration"]
            })
            return

        for student_file in pdf_files:
            loader = PyPDFLoader(student_file)
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
            if text.strip():
                profile = self.create_student_profile(text)
                filename = os.path.basename(student_file)
                clean_id = os.path.splitext(filename)[0].lower().replace('_answers', '')
                profile['student_id'] = clean_id
                self.class_profiles.append(profile)
        log_print(f"Analysis complete. {len(self.class_profiles)} student profiles created.")
        loaded_ids = [p['student_id'] for p in self.class_profiles]
        log_print(f"Successfully loaded profiles for: {loaded_ids}")

    def create_cul(self):
        if not self.class_profiles:
            return
        all_gaps = [gap for p in self.class_profiles for gap in p.get('knowledge_gaps', [])]
        all_styles = [p.get('inferred_style') for p in self.class_profiles if p.get('inferred_style')]
        gap_counts = Counter(all_gaps)
        style_counts = Counter(all_styles)
        self.cul_profile = {
            "style_distribution": dict(style_counts),
            "common_knowledge_gaps": [item for item, count in gap_counts.most_common(3)],
            "dominant_style": style_counts.most_common(1)[0][0] if style_counts else "general"
        }
        log_print(f"--- CUL GENERATED: {self.cul_profile} ---")

    def prepare_lesson(self, topic):
        self.current_topic = topic
        log_print(f"\n--- Preparing lesson for: {self.current_topic} ---")
        lesson_text = self.cul_chain.invoke({
            "dominant_style": self.cul_profile.get('dominant_style'),
            "common_knowledge_gaps": str(self.cul_profile.get('common_knowledge_gaps', 'None')),
            "topic": topic
        })
        self.lesson_sentences = [s.strip() for s in lesson_text.replace('\n', '. ').split('. ') if s.strip()]

    def handle_question(self, question_text: str, student_id: str):
        if not question_text:
            return
        student_profile = next((p for p in self.class_profiles if p['student_id'] == student_id), None)
        if student_profile:
            self.speak(f"A personalized answer for {student_id.capitalize()}. Let me check the textbook.")
            input_data = {
                "question": question_text,
                "student_name": student_id,
                "inferred_style": student_profile.get("inferred_style"),
                "knowledge_gaps": str(student_profile.get("knowledge_gaps", "None"))
            }
            final_answer = self.pul_chain.invoke(input_data)
            self.speak(final_answer)
        else:
            self.speak(f"My apologies, I couldn't find a student profile for '{student_id}'.")

    def run(self):
        self.analyze_and_prepare_class()
        self.create_cul()
        self.topics_queue = ["Photosynthesis", "Cellular Respiration"]

        self.camera_thread = threading.Thread(target=self._run_camera_recognition, daemon=True)
        self.camera_thread.start()

        self.speak("Good morning. Let's begin today's session.")
        with sr.Microphone() as source:
            student_in_session = None
            while self.is_running:
                if self.recognized_student_id:
                    student_in_session = self.recognized_student_id
                    self.recognized_student_id = None
                    self.mode = 'QA'
                    self.speak(f"Hello {student_in_session.capitalize()}, I see you have a question. Please ask.")
                
                if self.mode == 'TEACHING':
                    if not self.lesson_sentences:
                        if self.topics_queue:
                            next_topic = self.topics_queue.pop(0)
                            self.prepare_lesson(topic=next_topic)
                            self.speak(f"Our next topic is: {self.current_topic}.")
                            objectives = self.objectives_chain.invoke({"topic": self.current_topic})
                            self.speak(objectives)
                            self.speak("If you have a doubt, you can stand in front of the camera or say, 'I have a doubt.'")
                        else:
                            self.speak("We have covered all topics.")
                            self.is_running = False
                            continue
                    
                    self.speak(self.lesson_sentences.pop(0))
                    text = self.listen(source, timeout=1.5)
                    if text and "i have a doubt" in text: 
                        student_in_session = "aman"
                        self.mode = 'QA'
                        self.speak("Of course. What is your doubt?")
                
                elif self.mode == 'QA':
                    user_input = self.listen(source, timeout=7)
                    if user_input:
                        resume_phrases = ["no more", "continue"]
                        exit_phrases = ["goodbye", "end session"]
                        if any(phrase in user_input for phrase in resume_phrases): 
                            student_in_session = None
                            self.mode = 'TEACHING'
                            self.speak("Alright, let's continue.")
                            continue
                        elif any(phrase in user_input for phrase in exit_phrases): 
                            self.is_running = False
                            continue
                        else: 
                            self.handle_question(user_input, student_id=student_in_session)
                            self.speak("Do you have another question?")
                    else: 
                        student_in_session = None
                        self.mode = 'TEACHING'
                        self.speak("Hearing no more questions, we will continue.")

        self.is_running = False
        self.camera_thread.join()
        self.speak("That concludes our session. Goodbye!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logs')
def stream_logs():
    def generate():
        while True:
            try:
                msg = log_queue.get(timeout=1)
                yield f"data: {msg}\n\n"
            except queue.Empty:
                yield "data: \n\n"  # keep connection alive
    return Response(generate(), mimetype='text/event-stream')

def start_tutor():
    tutor = SmartTutor()
    tutor.run()

if __name__ == "__main__":
    tutor_thread = threading.Thread(target=start_tutor, daemon=True)
    tutor_thread.start()

    os.makedirs('templates', exist_ok=True)

    html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>Smart Tutor Logs</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f4f4f4; }
        #log { 
            background: white; 
            padding: 15px; 
            border-radius: 8px; 
            height: 70vh; 
            overflow-y: auto; 
            font-family: monospace; 
            white-space: pre-wrap;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { color: #2c3e50; }
    </style>
</head>
<body>
    <h1>Smart Tutor - Live Logs</h1>
    <div id="log">Initializing...</div>

    <script>
        const logDiv = document.getElementById('log');
        const eventSource = new EventSource('/logs');
        eventSource.onmessage = function(event) {
            if (event.data) {
                logDiv.innerHTML += event.data + '\\n';
                logDiv.scrollTop = logDiv.scrollHeight;
            }
        };
        eventSource.onerror = function() {
            logDiv.innerHTML += '[Connection lost]\\n';
        };
    </script>
</body>
</html>
    '''
    with open('templates/index.html', 'w') as f:
        f.write(html_content)

    print("Starting Flask server at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)