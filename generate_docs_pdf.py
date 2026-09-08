from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Face Attendance System - Technical Deep Dive', border=False, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, border=False, align='L', fill=True)
        self.ln(12)
        
    def sub_title(self, title):
        self.set_font('helvetica', 'B', 11)
        self.cell(0, 10, title, border=False, align='L')
        self.ln(8)

    def chapter_body(self, text):
        self.set_font('helvetica', '', 10)
        self.multi_cell(0, 7, text)
        self.ln(5)

pdf = PDF()
pdf.add_page()

pdf.chapter_title('1. System Architecture & Workflow')
pdf.sub_title('High-level Data Pipeline')
text_pipeline = (
    "The application operates as a standalone desktop kiosk rather than a distributed web architecture.\n"
    "- Webcam Frame Ingestion: Frames are captured locally via OpenCV (cv2.VideoCapture).\n"
    "- Processing: Frames are converted to grayscale and passed to a Haar Cascade classifier for face detection.\n"
    "- Recognition: Detected face regions are cropped, resized to 200x200, and passed to a local LBPH (Local Binary Patterns Histograms) model.\n"
    "- Database Persistence: Identified users are logged into a local SQLite database (attendance.db). The system checks for duplicate daily punches before committing."
)
pdf.chapter_body(text_pipeline)

pdf.sub_title('Flow Diagram / Sequence')
text_flow = (
    "Registration Flow (Admin Mode):\n"
    "1. Admin enters User ID, Name, and Date.\n"
    "2. System captures 40 grayscale face samples using the webcam, saving them to dataset/<user_id>/.\n"
    "3. System reads all images in dataset/ and trains the LBPHFaceRecognizer.\n"
    "4. The trained model is saved as models/trained_model.yml.\n\n"
    "Attendance Verification Flow (Kiosk Mode):\n"
    "1. User stands in front of the camera; 'Start Attendance' is triggered.\n"
    "2. OpenCV detects the face bounding box.\n"
    "3. LBPH predicts the face ID and provides a 'confidence' score (lower is better, acting as a distance metric).\n"
    "4. If confidence < 70, the user ID is queried in the database.\n"
    "5. mark_attendance() attempts an insert. If a unique constraint fails (User + Date), it flags as 'Already Marked'."
)
pdf.chapter_body(text_flow)


pdf.chapter_title('2. Core ML & Computer Vision Components')
text_ml_note = "(Note: The prompt mentions advanced components like ArcFace, FAISS, and Liveness detection. The current codebase implements a simpler approach. Comparisons are provided below.)"
pdf.chapter_body(text_ml_note)

pdf.sub_title('Feature Extraction & Embedding')
text_feature = (
    "- Implemented Model: Haar Cascades for detection + LBPH (Local Binary Patterns Histograms) for recognition.\n"
    "- Why it was chosen: LBPH is computationally inexpensive, runs seamlessly on CPUs without GPU acceleration, and works well for small, localized datasets.\n"
    "- Comparison to ArcFace/dlib: ArcFace extracts 128D/512D float embeddings using deep CNNs (like ResNet). LBPH relies on micro-patterns (texture) rather than deep semantic embeddings.\n"
    "- Raw Images: In a Deep Learning pipeline, raw images are discarded once embeddings are generated. However, in this LBPH implementation, raw grayscale images *are* stored locally in the dataset/ folder because LBPH requires full retraining on the dataset whenever a new user is added."
)
pdf.chapter_body(text_feature)

pdf.sub_title('Distance Metrics & Search')
text_distance = (
    "- Implemented Logic: The system uses LBPH's built-in prediction, which calculates the Chi-Square or Euclidean distance between the histogram of the input face and the trained histograms.\n"
    "- Comparison to FAISS: FAISS is NOT used here. FAISS provides O(1) / sub-millisecond vector indexing for millions of 512D embeddings. Since this system uses LBPH, the model handles the matching internally. SQLite is only used to map the predicted integer ID to a user's name."
)
pdf.chapter_body(text_distance)

pdf.sub_title('Anti-Spoofing & Liveness')
text_liveness = (
    "- Implemented Logic: The current codebase does not implement active liveness detection.\n"
    "- Future Recommendation: To prevent 2D presentation attacks (holding up a photo), mechanisms like EAR (Eye Aspect Ratio via landmarks) or texture analysis (LBP-SVM) should be integrated. Currently, a printed photo could trick the Haar + LBPH pipeline."
)
pdf.chapter_body(text_liveness)

pdf.add_page()
pdf.chapter_title('3. Backend & Database Architecture')
pdf.sub_title('Framework Choice')
text_framework = (
    "- Implemented Framework: CustomTkinter for the desktop GUI.\n"
    "- Comparison to FastAPI/Django: The system is a Desktop Application, not a Web API. CustomTkinter provides a bundled, zero-latency local interface (Kiosk) without needing a local web server or browsers."
)
pdf.chapter_body(text_framework)

pdf.sub_title('Database Design')
text_db = (
    "- Database Used: Raw SQLite3 via the built-in module (No SQLAlchemy ORM).\n"
    "- Schema: \n"
    "  * admins: username (PK), password_hash\n"
    "  * users: id (PK), user_code (Unique), name, registered_date\n"
    "  * attendance: entry_id (PK), user_code, name, date, time (UNIQUE constraint on user_code + date)\n"
    "- Duplicate Prevention: Handled at the DB level. INSERT INTO attendance fails gracefully via sqlite3.IntegrityError if the user has already punched in for the current date.\n"
    "- SQL Injection: Parameterized queries are used (e.g., execute('... WHERE id = ?', (id,))), preventing SQL injection."
)
pdf.chapter_body(text_db)

pdf.chapter_title('4. File-by-File Technical Breakdown')
text_files = (
    "- app.py: Main application entry point. Handles UI (CustomTkinter), camera integration, and state logic (Kiosk vs Admin).\n"
    "- db_manager.py: Handles all SQLite database interactions, schema initialization, and data querying.\n"
    "- model_trainer.py: Handles webcam data collection (dataset generation) and trains the LBPH model.\n"
    "- generate_pdf.py: Standalone script to generate PDF reports of the workflow using FPDF.\n"
    "- requirements.txt: Lists Python dependencies required to run the project.\n"
    "- attendance.db: Local SQLite database file containing persistent application state.\n"
    "- haarcascade_*.xml: Pre-trained model weights for OpenCV's face detector."
)
pdf.chapter_body(text_files)

pdf.chapter_title('5. Production Readiness & Edge Cases')
pdf.sub_title('Handled Edge Cases')
text_edge = (
    "- Duplicate Attendance: Database-level UNIQUE constraint prevents a user from marking attendance multiple times a day.\n"
    "- Missing Models: app.py gracefully degrades, showing a 'No Model Loaded' banner if trained_model.yml is missing.\n"
    "- Date Standardization: parse_and_standardize_date handles multiple date formats to ensure DB consistency."
)
pdf.chapter_body(text_edge)

pdf.sub_title('Limitations & Scalability (Areas for Improvement)')
text_limit = (
    "- Lighting/Pose Variance: Haar Cascades are limited to frontal faces and require good ambient lighting. LBPH struggles with extreme lighting changes.\n"
    "- Scalability: Retraining the entire LBPH model from scratch via model_trainer.py becomes computationally expensive as the dataset grows beyond a few hundred users.\n"
    "- Threshold Calibration: The confidence threshold is hardcoded to < 70. This might need dynamic calibration to balance False Positives and False Negatives."
)
pdf.chapter_body(text_limit)

pdf.output('Technical_DeepDive_Documentation.pdf')
