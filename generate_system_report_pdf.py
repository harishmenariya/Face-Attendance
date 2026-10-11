import os
import sys

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
except ImportError:
    print("ReportLab not found. Please install using: pip install reportlab")
    sys.exit(1)

def add_page_number(canvas_obj, doc):
    page_num = canvas_obj.getPageNumber()
    text = f"Page {page_num}"
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 9)
    canvas_obj.drawRightString(doc.pagesize[0] - 0.5*inch, 0.5*inch, text)
    canvas_obj.restoreState()

def generate_report():
    pdf_filename = "Face_Attendance_System_Complete_Report.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.8*inch
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=24,
        spaceAfter=20,
        textColor=colors.HexColor('#1E3A8A')
    )
    
    h1_style = ParagraphStyle(
        'H1Style',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#0F172A')
    )
    
    h2_style = ParagraphStyle(
        'H2Style',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#334155')
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        spaceBefore=6,
        spaceAfter=6,
        leading=14
    )

    story = []

    # 1. Cover Page & Executive Summary
    story.append(Paragraph("AURA-FACE: Enterprise Biometric Face Attendance System", title_style))
    story.append(Spacer(1, 0.5*inch))
    
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "The AURA-FACE system provides a robust, high-speed, and secure biometric attendance tracking solution. "
        "The primary problem solved is the prevention of buddy-punching and fraudulent attendance entries using sophisticated "
        "L2-normalized deep embeddings. "
        "The system integrates a sleek CustomTkinter UI for kiosk operation, enabling sub-millisecond retrieval via Meta's FAISS, "
        "ensuring a frictionless and secure experience for both administrators and employees.", 
        body_style
    ))
    
    # 2. Tech Stack & Architectural Justifications
    story.append(Paragraph("2. Comprehensive Tech Stack & Architectural Justifications", h1_style))
    
    tech_data = [
        ["Layer", "Technology", "Justification / Role"],
        ["Presentation Layer", "CustomTkinter", "Modern dark-mode Kiosk & Admin Dashboard."],
        ["Face Detection", "OpenCV YuNet", "Lightweight ONNX, high-speed bounding box & landmark extraction."],
        ["Feature Extraction", "MobileNetV2 (PyTorch)", "Produces 128-dimensional unit-normalized embedding vectors."],
        ["Vector Search", "Meta FAISS", "IndexIDMap2 wrapping IndexFlatL2 for sub-millisecond similarity matching."],
        ["Relational Storage", "SQLite3", "attendance.db tracks users, relationships, and timestamped attendance logs."]
    ]
    
    t = Table(tech_data, colWidths=[1.5*inch, 1.5*inch, 4.0*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FAFC')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))
    
    # 3. Module-by-Module Code Architecture
    story.append(Paragraph("3. Module-by-Module Code Architecture (File-by-File Breakdown)", h1_style))
    
    story.append(Paragraph("face_engine.py", h2_style))
    story.append(Paragraph(
        "Initializes the FaceEngine using YuNet for fast face detection. A PyTorch-based MobileNetV2 feature extractor processes "
        "the aligned face crops to output a 128-d vector. L2 normalization is strictly applied at the model output to map faces to a unit hypersphere.", body_style))

    story.append(Paragraph("vector_index.py", h2_style))
    story.append(Paragraph(
        "Manages the FAISS indexing. It creates an IndexIDMap2 wrapper around IndexFlatL2 to map user IDs (int64) to embeddings. "
        "Methods include add_vector(uid, vec), search(vec, threshold) for top-1 mapping, and remove_vector(uid) leveraging faiss.IDSelectorArray.", body_style))
        
    story.append(Paragraph("db_manager.py", h2_style))
    story.append(Paragraph(
        "Handles SQLite3 schema setups for the users and attendance tables. Supports full cascading deletes, sequential "
        "SQLite autoincrements, and enforces foreign-key logic mapping numeric IDs to string-based User Codes.", body_style))
        
    story.append(Paragraph("app.py", h2_style))
    story.append(Paragraph(
        "The core presentation tier. Provides a dual-mode workflow: A Kiosk scanner loop for live OpenCV inference, "
        "and a secure Admin Management Portal. Manages threading, UI updates via CustomTkinter, real-time bounding boxes, auto-close timers, and table renderings.", body_style))
        
    story.append(Paragraph("evaluate_accuracy.py & Maintenance Tools", h2_style))
    story.append(Paragraph(
        "Tools to benchmark the system via FAR, FRR, Precision, and Recall metrics. Maintenance scripts such as "
        "resequence_uids.py ensure database and FAISS index harmony over time.", body_style))

    # 4. End-to-End Operational Workflows
    story.append(Paragraph("4. End-to-End Operational Workflows", h1_style))
    
    story.append(Paragraph("Enrollment Workflow", h2_style))
    story.append(Paragraph(
        "1. Photo capture triggered in UI.<br/>"
        "2. YuNet crop extracts the face.<br/>"
        "3. Embedder generates a 128-d L2-normalized vector.<br/>"
        "4. Strict duplicate check (Threshold: 0.20) runs against FAISS.<br/>"
        "5. SQLite records the User ID.<br/>"
        "6. FAISS indexes the vector associated with the SQLite ID.", body_style))
        
    story.append(Paragraph("Attendance Kiosk Workflow", h2_style))
    story.append(Paragraph(
        "1. Live camera feed processed continuously.<br/>"
        "2. YuNet detects faces.<br/>"
        "3. MobileNetV2 embedding is generated.<br/>"
        "5. FAISS searches the index for a match (Threshold: 0.55).<br/>"
        "6. SQLite logs the attendance timestamp.<br/>"
        "7. 1.5s visual confirmation displayed before camera auto-shutoff.", body_style))

    # Build the document
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Report successfully generated at: {os.path.abspath(pdf_filename)}")

if __name__ == '__main__':
    generate_report()
