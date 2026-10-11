import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import io
import csv
import time
import sqlite3
from datetime import datetime

import db_manager as db
from face_engine import FaceEngine
from vector_index import VectorIndex

app = FastAPI(title="Face Attendance AI Engine")

# Next.js requests allow karne ke liye CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI Models & Index initialization
db.init_db()
engine = FaceEngine()
v_index = VectorIndex()

class FramePayload(BaseModel):
    image_base64: str

class EnrollPayload(BaseModel):
    user_code: str
    name: str
    images_base64: List[str]

class LoginPayload(BaseModel):
    username: str
    password: str

class UserUpdateSchema(BaseModel):
    user_code: str
    name: str

def parse_base64_image(b64_string: str):
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    raw_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# 1. Kiosk Attendance Endpoint
@app.post("/api/scan-attendance")
def scan_attendance(payload: FramePayload):
    try:
        frame = parse_base64_image(payload.image_base64)
        if frame is None:
            return {"status": "error", "message": "Corrupted frame"}

        faces = engine.detect_faces(frame)
        if faces is None or len(faces) == 0:
            return {"status": "no_face", "message": "No face detected in camera viewport"}

        # Extract embedding
        x, y, w, h = [int(v) for v in faces[0][:4]]
        face_crop = frame[max(0, y):max(0, y)+h, max(0, x):max(0, x)+w]
        emb = engine.extract_embedding(face_crop)

        # L2 Normalize embedding
        norm = np.linalg.norm(emb)
        emb = emb / (norm + 1e-10)

        # FAISS Vector Search (Attendance threshold <= 0.42)
        matched_id, dist = v_index.search(emb, threshold=0.42)

        if matched_id is not None:
            user = db.get_user_by_id(matched_id)
            if user:
                import sqlite3
                conn = sqlite3.connect(db.DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM attendance WHERE user_code = ? AND date = DATE('now', 'localtime')", (user[0],))
                already_marked = cursor.fetchone()
                conn.close()

                if already_marked:
                    return {
                        "status": "already_marked",
                        "name": user[1],
                        "user_code": user[0],
                        "uid": matched_id
                    }

                success, info = db.mark_attendance(user[0], user[1])
                if success:
                    return {
                        "status": "success",
                        "uid": matched_id,
                        "user_code": user[0],
                        "name": user[1],
                        "distance": round(float(dist), 4)
                    }
                else:
                    return {
                        "status": "already_marked",
                        "name": user[1],
                        "user_code": user[0],
                        "uid": matched_id
                    }

        return {"status": "unknown"}
    except Exception as e:
        print(f"Error processing frame: {e}")
        return {"status": "error", "message": "Invalid frame"}

# 2. Get Registered Users Endpoint
@app.get("/api/users")
def get_users():
    raw_users = db.get_all_users()
    return [{"uid": u[0], "code": u[1], "name": u[2], "date": u[3], "status": "Active"} for u in raw_users]

# 3. Enrollment Endpoint (Strict 0.15 Duplicate Collision Check)
@app.post("/api/enroll")
def enroll_user(payload: EnrollPayload):
    if len(payload.images_base64) == 0:
        raise HTTPException(status_code=400, detail="No images provided.")

    embeddings = []
    for b64 in payload.images_base64:
        frame = parse_base64_image(b64)
        if frame is None:
            continue
        faces = engine.detect_faces(frame)
        if faces is not None and len(faces) > 0:
            face = max(faces, key=lambda f: f[2]*f[3])
            x, y, w, h = face[:4].astype(int)
            pad_x = int(w * 0.1)
            pad_y = int(h * 0.1)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                emb = engine.extract_embedding(face_crop)
                embeddings.append(emb)

    if len(embeddings) == 0:
        raise HTTPException(status_code=400, detail="No valid faces detected in the provided samples.")

    # Calculate mean embedding
    mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm

    # Strict Duplicate Prevention (Threshold 0.15 to match app.py)
    matched_id, dist = v_index.search(mean_emb, threshold=0.15)
    if matched_id is not None:
        user = db.get_user_by_id(matched_id)
        if user:
            raise HTTPException(
                status_code=409, 
                detail=f"Duplicate biometric detected! Already enrolled as {user[1]} (UID #{matched_id})"
            )

    reg_date = time.strftime("%Y-%m-%d")
    user_id = db.add_user(payload.user_code, payload.name, reg_date)
    if user_id == -1:
         raise HTTPException(status_code=409, detail="User Code already exists!")

    v_index.add_vector(user_id, mean_emb)
    return {"status": "success", "uid": user_id, "name": payload.name}

# 4. Export Endpoints
@app.get("/api/export/today")
def export_today():
    records = db.get_today_attendance()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["User Code", "Name", "Date", "Time"])
    writer.writerows(records)
    
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=Attendance_{time.strftime('%Y-%m-%d')}.csv"}
    )

@app.get("/api/export/summary")
def export_summary():
    summary = db.get_user_attendance_summary()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["User Code", "Name", "Total Days", "Present", "Absent", "Attendance %"])
    for s in summary:
        writer.writerow([s['user_code'], s['name'], s['total_days'], s['present'], s['absent'], s['percentage']])
        
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=Attendance_Summary_{time.strftime('%Y-%m-%d')}.csv"}
    )

# 5. Admin Auth Endpoint
@app.post("/api/admin/login")
def admin_login(payload: LoginPayload):
    if db.verify_admin(payload.username, payload.password):
        return {"status": "success"}
    else:
        raise HTTPException(status_code=401, detail="Invalid Credentials")

@app.get("/api/reports/daily")
def get_daily_report(date_str: str = None):
    return db.get_daily_report(date_str)

@app.get("/api/attendance-logs")
def get_attendance_logs():
    records = db.get_today_attendance()
    return [{"id": idx, "code": r[0], "name": r[1], "time": r[3]} for idx, r in enumerate(records)]

@app.get("/api/reports/monthly")
def get_monthly_report(month_str: str = None):
    return db.get_monthly_report(month_str) 

@app.put("/api/users/{user_id}")
def update_user(user_id: int, payload: UserUpdateSchema):
    success, msg = db.update_user(user_id, payload.user_code, payload.name)
    if success:
        return {"status": "success", "message": msg}
    else:
        raise HTTPException(status_code=400, detail=msg)

@app.delete("/api/users/{user_id}")
def delete_user(user_id: int):
    # Remove from FAISS index first
    v_index.remove_vector(user_id)
    # Remove from SQLite
    success, msg = db.delete_user(user_id)
    if success:
        return {"status": "success", "message": "User deleted and FAISS index rebuilt"}
    else:
        raise HTTPException(status_code=400, detail=msg)

@app.get("/api/reports/user-monthly")
def get_user_monthly_report(user_id: int, month_str: str = None):
    res = db.get_user_monthly_report(user_id, month_str)
    if res.get("status") == "error":
        raise HTTPException(status_code=404, detail=res["message"])
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host="0.0.0.0", port=8000, reload=True)