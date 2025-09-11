import os
import json
import time
from datetime import datetime, timedelta
import speech_recognition as sr
import pyttsx3
from collections import Counter
import threading
import random

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Set up API key
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY_HERE"

class SmartTutoringSystem:
    def __init__(self):
        print("Initializing Smart AI Tutoring System...")
        
        # Core components
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.recognizer = sr.Recognizer()
        
        # RAG setup
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = self._load_textbook_db()
        self.retriever = self.db.as_retriever(search_kwargs={"k": 5})
        
        # LLM setup
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
        
        # Load all system data
        self.student_profiles = self.load_student_profiles()
        self.class_cul = self.calculate_class_cul()
        self.timetable = self.load_timetable()
        self.syllabus = self.load_syllabus()
        
        # System state
        self.current_period = None
        self.is_teaching = False
        self.doubt_queue = []
        
        # Automatic doubt simulation
        self.simulate_doubts = True
        
        self._initialize_chains()
        print("System initialized successfully!")
        print(f"Class CUL Summary: {self.class_cul}")

    def _load_textbook_db(self):
        """Load textbook content into vector database"""
        db_directory = "./textbook_db"
        if os.path.exists(db_directory):
            return Chroma(persist_directory=db_directory, embedding_function=self.embeddings)
        else:
            # Create sample documents if no textbooks exist
            sample_docs = self.create_sample_textbook_content()
            if sample_docs:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_documents(sample_docs)
                db = Chroma.from_documents(chunks, self.embeddings, persist_directory=db_directory)
                return db
            else:
                return Chroma(persist_directory=db_directory, embedding_function=self.embeddings)

    def create_sample_textbook_content(self):
        """Create sample textbook content for demo"""
        from langchain.schema import Document
        
        sample_content = [
            {
                'subject': 'biology',
                'topic': 'Photosynthesis',
                'content': """Photosynthesis is the process by which green plants make their own food using sunlight, carbon dioxide, and water. The process occurs in the leaves of plants, specifically in structures called chloroplasts. Chloroplasts contain a green pigment called chlorophyll which captures sunlight. The equation for photosynthesis is: 6CO2 + 6H2O + sunlight → C6H12O6 + 6O2. This process is vital for all life on Earth as it produces oxygen and food."""
            },
            {
                'subject': 'biology',
                'topic': 'Cell Division',
                'content': """Cell division is the process by which a single cell divides to form two or more daughter cells. There are two main types of cell division: mitosis and meiosis. Mitosis produces identical diploid cells for growth and repair. Meiosis produces genetically different haploid gametes for reproduction. The cell cycle consists of interphase, prophase, metaphase, anaphase, and telophase."""
            },
            {
                'subject': 'math',
                'topic': 'Quadratic Equations',
                'content': """A quadratic equation is a polynomial equation of degree 2. The standard form is ax² + bx + c = 0, where a ≠ 0. Solutions can be found using factoring, completing the square, or the quadratic formula: x = (-b ± √(b²-4ac))/2a. The discriminant b²-4ac determines the nature of roots. If positive, there are two real roots; if zero, one repeated root; if negative, complex roots."""
            },
            {
                'subject': 'math',
                'topic': 'Trigonometry',
                'content': """Trigonometry is the study of triangles and the relationships between their sides and angles. The main trigonometric ratios are sine (sin), cosine (cos), and tangent (tan). For a right triangle: sin θ = opposite/hypotenuse, cos θ = adjacent/hypotenuse, tan θ = opposite/adjacent. These ratios are used to solve problems involving heights, distances, and angles."""
            },
            {
                'subject': 'physics',
                'topic': 'Light Reflection',
                'content': """Light reflection occurs when light rays bounce off a surface. The laws of reflection state: 1) The incident ray, reflected ray, and normal all lie in the same plane. 2) The angle of incidence equals the angle of reflection. Mirrors use reflection - plane mirrors produce virtual images, while curved mirrors (concave and convex) can produce both real and virtual images depending on object position."""
            },
            {
                'subject': 'physics',
                'topic': 'Electric Current',
                'content': """Electric current is the flow of electric charge through a conductor. It is measured in amperes (A). Current flows from positive to negative terminal in a circuit. Ohm's law states V = IR, where V is voltage, I is current, and R is resistance. There are two types of current: direct current (DC) flows in one direction, alternating current (AC) changes direction periodically."""
            }
        ]
        
        documents = []
        for item in sample_content:
            doc = Document(
                page_content=item['content'],
                metadata={'subject': item['subject'], 'topic': item['topic']}
            )
            documents.append(doc)
        
        return documents

    def load_student_profiles(self):
        """Load student understanding levels from answer paper analysis"""
        profiles = {
            'student_001': {
                'name': 'Rajesh',
                'biology': {'understanding_level': 'high', 'learning_style': 'visual'},
                'math': {'understanding_level': 'medium', 'learning_style': 'step_by_step'},
                'physics': {'understanding_level': 'low', 'learning_style': 'conceptual'}
            },
            'student_002': {
                'name': 'Priya',
                'biology': {'understanding_level': 'medium', 'learning_style': 'auditory'},
                'math': {'understanding_level': 'high', 'learning_style': 'analytical'},
                'physics': {'understanding_level': 'medium', 'learning_style': 'practical'}
            },
            'student_003': {
                'name': 'Harpreet',
                'biology': {'understanding_level': 'low', 'learning_style': 'step_by_step'},
                'math': {'understanding_level': 'low', 'learning_style': 'visual'},
                'physics': {'understanding_level': 'high', 'learning_style': 'conceptual'}
            },
            'student_004': {
                'name': 'Simran',
                'biology': {'understanding_level': 'medium', 'learning_style': 'practical'},
                'math': {'understanding_level': 'medium', 'learning_style': 'step_by_step'},
                'physics': {'understanding_level': 'medium', 'learning_style': 'visual'}
            },
            'student_005': {
                'name': 'Gurpreet',
                'biology': {'understanding_level': 'high', 'learning_style': 'conceptual'},
                'math': {'understanding_level': 'low', 'learning_style': 'visual'},
                'physics': {'understanding_level': 'low', 'learning_style': 'step_by_step'}
            }
        }
        return profiles

    def calculate_class_cul(self):
        """Calculate Common Understanding Level for the class"""
        subjects = ['biology', 'math', 'physics']
        cul = {}
        
        for subject in subjects:
            levels = []
            styles = []
            
            for student_id, profile in self.student_profiles.items():
                if subject in profile:
                    levels.append(profile[subject]['understanding_level'])
                    styles.append(profile[subject]['learning_style'])
            
            # Determine most common level and style
            if levels:
                most_common_level = Counter(levels).most_common(1)[0][0]
                most_common_style = Counter(styles).most_common(1)[0][0]
            else:
                most_common_level = 'medium'
                most_common_style = 'visual'
            
            cul[subject] = {
                'common_understanding_level': most_common_level,
                'common_learning_style': most_common_style
            }
        
        return cul

    def load_timetable(self):
        """Load weekly class timetable"""
        return {
            'monday': [
                {'time': '09:00', 'subject': 'biology', 'topic': 'Photosynthesis', 'duration': 45},
                {'time': '10:00', 'subject': 'math', 'topic': 'Quadratic Equations', 'duration': 45},
                {'time': '11:00', 'subject': 'physics', 'topic': 'Light Reflection', 'duration': 45}
            ],
            'tuesday': [
                {'time': '09:00', 'subject': 'biology', 'topic': 'Cell Division', 'duration': 45},
                {'time': '10:00', 'subject': 'math', 'topic': 'Trigonometry', 'duration': 45},
                {'time': '11:00', 'subject': 'physics', 'topic': 'Electric Current', 'duration': 45}
            ],
            'wednesday': [
                {'time': '09:00', 'subject': 'biology', 'topic': 'Respiration', 'duration': 45},
                {'time': '10:00', 'subject': 'math', 'topic': 'Geometry', 'duration': 45},
                {'time': '11:00', 'subject': 'physics', 'topic': 'Sound Waves', 'duration': 45}
            ],
            'thursday': [
                {'time': '09:00', 'subject': 'biology', 'topic': 'Genetics', 'duration': 45},
                {'time': '10:00', 'subject': 'math', 'topic': 'Statistics', 'duration': 45},
                {'time': '11:00', 'subject': 'physics', 'topic': 'Heat Transfer', 'duration': 45}
            ],
            'friday': [
                {'time': '09:00', 'subject': 'biology', 'topic': 'Ecology', 'duration': 45},
                {'time': '10:00', 'subject': 'math', 'topic': 'Probability', 'duration': 45},
                {'time': '11:00', 'subject': 'physics', 'topic': 'Magnetism', 'duration': 45}
            ]
        }

    def load_syllabus(self):
        """Load detailed syllabus for each subject"""
        return {
            'biology': {
                'Photosynthesis': ['Light reactions', 'Dark reactions', 'Factors affecting photosynthesis'],
                'Cell Division': ['Mitosis', 'Meiosis', 'Cell cycle regulation'],
                'Respiration': ['Aerobic respiration', 'Anaerobic respiration', 'Respiratory quotient'],
                'Genetics': ['Mendels laws', 'Genetic crosses', 'DNA structure'],
                'Ecology': ['Ecosystem', 'Food chains', 'Environmental factors']
            },
            'math': {
                'Quadratic Equations': ['Standard form', 'Quadratic formula', 'Nature of roots'],
                'Trigonometry': ['Trigonometric ratios', 'Identities', 'Applications'],
                'Geometry': ['Circles', 'Triangles', 'Area and perimeter'],
                'Statistics': ['Mean, median, mode', 'Standard deviation', 'Data representation'],
                'Probability': ['Basic probability', 'Conditional probability', 'Bayes theorem']
            },
            'physics': {
                'Light Reflection': ['Laws of reflection', 'Mirrors', 'Image formation'],
                'Electric Current': ['Ohms law', 'Resistance', 'Power and energy'],
                'Sound Waves': ['Wave properties', 'Sound propagation', 'Resonance'],
                'Heat Transfer': ['Conduction', 'Convection', 'Radiation'],
                'Magnetism': ['Magnetic fields', 'Electromagnetic induction', 'Motors and generators']
            }
        }

    def _initialize_chains(self):
        """Initialize LangChain prompts and chains"""
        
        # CUL Chain - for class teaching
        cul_template = """You are a teacher in a rural Punjab school teaching {subject}.

Class Common Understanding Level: {common_understanding_level}
Class Common Learning Style: {common_learning_style}
Topic: {topic}
Syllabus Points to Cover: {syllabus_points}
Textbook Context: {context}

Explain the concept of {topic} covering these syllabus points: {syllabus_points}.
Teach according to the class's common understanding level ({common_understanding_level}) and learning style ({common_learning_style}).

Guidelines:
- Speak as if teaching a real classroom of rural Punjab students
- Use simple language and local examples they can relate to
- Break complex concepts into smaller, digestible parts
- Make it engaging and interactive
- Cover all the syllabus points systematically
- Give real-world applications and examples"""

        cul_prompt = PromptTemplate.from_template(cul_template)
        self.cul_chain = (
            {"context": self.retriever, "subject": RunnablePassthrough(), "topic": RunnablePassthrough(),
             "syllabus_points": RunnablePassthrough(), "common_understanding_level": RunnablePassthrough(), 
             "common_learning_style": RunnablePassthrough()}
            | cul_prompt | self.llm | StrOutputParser()
        )

        # PUL Chain - for individual student doubts
        pul_template = """You are a teacher helping student {student_name} with a doubt about {topic}.

Student's Personal Understanding Level in {subject}: {personal_understanding_level}
Student's Learning Style: {personal_learning_style}
Student's Doubt/Question: {question}
Textbook Context: {context}

Answer the student's question considering their personal understanding level and learning style.
Make it specifically suitable for {student_name}'s learning needs.

Guidelines:
- Address the student personally
- Use their preferred learning style
- Match their understanding level
- Be encouraging and patient
- Provide clear, step-by-step explanations if needed"""

        pul_prompt = PromptTemplate.from_template(pul_template)
        self.pul_chain = (
            {"context": self.retriever, "student_name": RunnablePassthrough(), "subject": RunnablePassthrough(),
             "topic": RunnablePassthrough(), "question": RunnablePassthrough(), 
             "personal_understanding_level": RunnablePassthrough(), "personal_learning_style": RunnablePassthrough()}
            | pul_prompt | self.llm | StrOutputParser()
        )

        # Question generation chain
        question_template = """Generate a question about {topic} for {subject} suitable for {understanding_level} level students.

Syllabus Points: {syllabus_points}
Context: {context}

Create a question that tests understanding of the key concepts.
Make it appropriate for rural Punjab school students at {understanding_level} level."""
        
        question_prompt = PromptTemplate.from_template(question_template)
        self.question_chain = (
            {"context": self.retriever, "subject": RunnablePassthrough(), "topic": RunnablePassthrough(),
             "syllabus_points": RunnablePassthrough(), "understanding_level": RunnablePassthrough()}
            | question_prompt | self.llm | StrOutputParser()
        )

    def speak(self, text):
        """Convert text to speech with realistic pauses"""
        print(f"🎯 Teacher: {text}")
        sentences = text.split('.')
        for sentence in sentences:
            if sentence.strip():
                self.tts_engine.say(sentence.strip())
                self.tts_engine.runAndWait()
                time.sleep(0.5)  # Natural pause between sentences

    def get_current_period(self):
        """Get current period based on actual time"""
        now = datetime.now()
        current_day = now.strftime('%A').lower()
        current_time = now.strftime('%H:%M')
        
        if current_day in self.timetable:
            for period in self.timetable[current_day]:
                period_start = datetime.strptime(period['time'], '%H:%M').time()
                period_end = (datetime.combine(datetime.today(), period_start) + timedelta(minutes=period['duration'])).time()
                current_time_obj = datetime.strptime(current_time, '%H:%M').time()
                
                if period_start <= current_time_obj <= period_end:
                    return period
        
        # For demo purposes, return first period of Monday
        return self.timetable['monday'][0] if 'monday' in self.timetable else None

    def simulate_student_doubt(self):
        """Simulate student doubts during class"""
        if not self.is_teaching:
            return
            
        # Generate random doubt scenarios
        doubt_scenarios = [
            "I don't understand how this process works in detail",
            "Can you explain this with a simpler example?",
            "What is the practical application of this concept?",
            "I'm confused about the previous step, can you clarify?",
            "How does this relate to what we learned last week?"
        ]
        
        # Random student asks doubt
        students = list(self.student_profiles.keys())
        random_student = random.choice(students)
        random_doubt = random.choice(doubt_scenarios)
        
        self.doubt_queue.append({
            'student_id': random_student,
            'question': random_doubt,
            'timestamp': datetime.now()
        })
        
        print(f"\n💭 {self.student_profiles[random_student]['name']} raises hand with a doubt...")

    def start_class(self, period):
        """Start class with greetings and learning objectives"""
        subject = period['subject']
        topic = period['topic']
        
        print(f"\n{'='*60}")
        print(f"🏫 STARTING {subject.upper()} CLASS")
        print(f"📚 Topic: {topic}")
        print(f"⏰ Time: {period['time']}")
        print(f"{'='*60}")
        
        # Greetings
        self.speak("Sat Sri Akal everyone! Good morning class. Please take your seats.")
        time.sleep(2)
        
        # Attendance check
        self.speak("Let me check if everyone is present today.")
        time.sleep(1)
        
        # Announce subject and topic
        self.speak(f"Today we have {subject} class. Our topic for today is {topic}.")
        time.sleep(2)
        
        # Learning objectives based on syllabus
        syllabus_points = self.syllabus.get(subject, {}).get(topic, [])
        if syllabus_points:
            objectives = f"By the end of this class, you will understand: {', '.join(syllabus_points)}"
            self.speak(f"Learning objectives for today: {objectives}")
        
        time.sleep(2)

    def teach_class(self, subject, topic):
        """Automatically teach class using CUL and syllabus"""
        self.is_teaching = True
        
        # Get CUL data for the class
        cul_data = self.class_cul.get(subject, {
            'common_understanding_level': 'medium', 
            'common_learning_style': 'visual'
        })
        
        # Get syllabus points to cover
        syllabus_points = self.syllabus.get(subject, {}).get(topic, ['Basic concepts'])
        
        print(f"\n📊 Teaching using CUL:")
        print(f"   Level: {cul_data['common_understanding_level']}")
        print(f"   Style: {cul_data['common_learning_style']}")
        print(f"   Syllabus: {syllabus_points}")
        
        # Prepare lesson input
        lesson_input = {
            'subject': subject,
            'topic': topic,
            'syllabus_points': ', '.join(syllabus_points),
            'common_understanding_level': cul_data['common_understanding_level'],
            'common_learning_style': cul_data['common_learning_style']
        }
        
        # Generate lesson content using RAG
        self.speak("Now let me explain today's topic step by step.")
        lesson_content = self.cul_chain.invoke(lesson_input)
        
        # Break lesson into manageable parts
        lesson_parts = lesson_content.split('\n\n')
        
        for i, part in enumerate(lesson_parts):
            if part.strip():
                self.speak(part.strip())
                
                # Simulate natural teaching pace
                time.sleep(3)
                
                # Check for simulated doubts periodically
                if i > 0 and i % 2 == 0 and self.simulate_doubts:
                    if random.random() < 0.3:  # 30% chance of doubt
                        self.simulate_student_doubt()
                        if self.doubt_queue:
                            self.handle_student_doubt(subject, topic)
        
        self.is_teaching = False

    def handle_student_doubt(self, subject, topic):
        """Handle student doubts using PUL automatically"""
        if not self.doubt_queue:
            return
        
        doubt = self.doubt_queue.pop(0)
        student_id = doubt['student_id']
        question = doubt['question']
        student_profile = self.student_profiles[student_id]
        
        print(f"\n🙋 Handling doubt from {student_profile['name']}")
        
        self.speak(f"Yes {student_profile['name']}, I can see you have a question. Let me help you.")
        time.sleep(1)
        
        # Get student's PUL for this subject
        if subject in student_profile:
            personal_level = student_profile[subject]['understanding_level']
            personal_style = student_profile[subject]['learning_style']
        else:
            personal_level = 'medium'
            personal_style = 'visual'
        
        print(f"📈 Using PUL for {student_profile['name']}:")
        print(f"   Level: {personal_level}")
        print(f"   Style: {personal_style}")
        
        # Generate personalized response
        pul_input = {
            'student_name': student_profile['name'],
            'subject': subject,
            'topic': topic,
            'question': question,
            'personal_understanding_level': personal_level,
            'personal_learning_style': personal_style
        }
        
        # Announce the doubt to class
        self.speak(f"{student_profile['name']} wants to know: {question}")
        time.sleep(1)
        
        # Provide personalized answer
        answer = self.pul_chain.invoke(pul_input)
        self.speak(answer)
        
        # Ask if doubt is cleared
        self.speak(f"Is that clear {student_profile['name']}? Good! Let's continue with our lesson.")
        time.sleep(2)

    def ask_random_questions(self, subject, topic):
        """Automatically ask questions to random students"""
        self.speak("Now let me check your understanding with some questions.")
        time.sleep(2)
        
        # Get class understanding level for question difficulty
        cul_data = self.class_cul.get(subject, {'common_understanding_level': 'medium'})
        syllabus_points = self.syllabus.get(subject, {}).get(topic, ['Basic concepts'])
        
        # Select random students
        students = list(self.student_profiles.items())
        random.shuffle(students)
        num_questions = min(3, len(students))
        
        for i in range(num_questions):
            student_id, student_profile = students[i]
            student_name = student_profile['name']
            
            # Generate question
            question_input = {
                'subject': subject,
                'topic': topic,
                'syllabus_points': ', '.join(syllabus_points),
                'understanding_level': cul_data['common_understanding_level']
            }
            
            question = self.question_chain.invoke(question_input)
            
            self.speak(f"{student_name}, please answer this question: {question}")
            time.sleep(4)  # Give time for student to think
            
            # Simulate student response
            responses = [
                "gives a good answer with some minor mistakes",
                "provides correct answer with good explanation", 
                "struggles with the answer but shows understanding",
                "gives partially correct answer",
                "asks for clarification before answering"
            ]
            
            response = random.choice(responses)
            print(f"📝 {student_name} {response}")
            
            feedback = [
                "Very good! That's exactly right.",
                "Good attempt! Let me add one more point.",
                "You're on the right track. Let me clarify that part.",
                "Nice try! The correct answer is...",
                "Good question! Let me explain that again."
            ]
            
            self.speak(random.choice(feedback))
            time.sleep(2)

    def assign_homework(self, subject, topic):
        """Automatically assign homework based on topic and syllabus"""
        syllabus_points = self.syllabus.get(subject, {}).get(topic, [])
        
        homework_tasks = [
            f"Read chapter on {topic} from your {subject} textbook",
            f"Practice problems related to {', '.join(syllabus_points[:2]) if syllabus_points else topic}",
            f"Make notes on key points of {topic}",
            f"Draw diagrams if applicable for {topic}"
        ]
        
        homework = "; ".join(homework_tasks[:2])  # Assign 2 tasks
        
        self.speak(f"For homework, please complete these tasks: {homework}")
        time.sleep(2)
        self.speak("Please submit your homework next class. Thank you everyone!")
        self.speak("Class dismissed! Have a good day!")
        
        print(f"\n✅ Homework assigned for {topic}")
        print(f"📝 Tasks: {homework}")

    def run_automated_session(self):
        """Run fully automated class session based on timetable"""
        print("🚀 Starting Automated Smart Tutoring System...")
        
        # Get current period from timetable
        current_period = self.get_current_period()
        if not current_period:
            print("⏰ No class scheduled at this time. Running demo with Biology - Photosynthesis")
            current_period = {
                'subject': 'biology', 
                'topic': 'Photosynthesis', 
                'time': '09:00', 
                'duration': 45
            }
        
        subject = current_period['subject']
        topic = current_period['topic']
        
        print(f"\n🎯 Auto-starting {subject.title()} class - Topic: {topic}")
        
        # Step 1: Automatically start class
        self.start_class(current_period)
        
        # Step 2: Teach using CUL and syllabus
        self.teach_class(subject, topic)
        
        # Step 3: Handle any remaining doubts
        while self.doubt_queue:
            self.handle_student_doubt(subject, topic)
        
        # Step 4: Automatically ask questions
        self.ask_random_questions(subject, topic)
        
        # Step 5: Assign homework automatically
        self.assign_homework(subject, topic)
        
        print(f"\n✨ {subject.title()} class completed successfully!")
        print("📊 Session Summary:")
        print(f"   - Subject: {subject.title()}")
        print(f"   - Topic: {topic}")
        print(f"   - CUL Used: {self.class_cul[subject]['common_understanding_level']} level")
        print(f"   - Students Taught: {len(self.student_profiles)}")
        print(f"   - Doubts Handled: Individual PUL-based responses")

    def run_continuous_schedule(self):
        """Run continuous automated classes based on timetable"""
        print("🔄 Starting Continuous Automated Schedule...")
        
        while True:
            current_period = self.get_current_period()
            if current_period and not self.is_teaching:
                print(f"\n⏰ Time for {current_period['subject']} class!")
                self.run_automated_session()
                
                # Wait for next period
                time.sleep(current_period['duration'] * 60)  # Convert to seconds
            else:
                print("⏸️ Waiting for next scheduled class...")
                time.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    # Initialize system
    tutor = SmartTutoringSystem()
    
    # Run automated session
    tutor.run_automated_session()
    
    # Uncomment below for continuous schedule
    # tutor.run_continuous_schedule()