import os
import cv2
import db_manager as db
from vector_index import VectorIndex
from face_engine import FaceEngine

def run_diagnostic():
    db.init_db()
    v_index = VectorIndex()
    engine = FaceEngine()

    reg_path = os.path.join("evaluation_data", "registered")
    print(f"Total vectors stored in FAISS: {v_index.index.ntotal}")

    if not os.path.exists(reg_path):
        print(f"Directory {reg_path} does not exist.")
        return

    for user_code in os.listdir(reg_path):
        folder = os.path.join(reg_path, user_code)
        if not os.path.isdir(folder):
            continue
        for img_name in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, img_name))
            if img is None:
                continue
                
            faces = engine.detect_faces(img)
            if faces is not None and len(faces) > 0:
                x, y, w, h = int(faces[0][0]), int(faces[0][1]), int(faces[0][2]), int(faces[0][3])
                crop = img[max(0, y):max(0, y)+h, max(0, x):max(0, x)+w]
                if crop.size == 0:
                    continue
                    
                emb = engine.extract_embedding(crop)
                
                # Query FAISS without distance gating
                D, I = v_index.index.search(emb.reshape(1, -1).astype("float32"), k=1)
                
                if len(D) > 0 and len(D[0]) > 0:
                    raw_dist = float(D[0][0])
                    matched_id = int(I[0][0])
                    print(f"[{user_code} / {img_name}] -> Matched ID: {matched_id}, Raw FAISS Distance: {raw_dist:.4f}")
                else:
                    print(f"[{user_code} / {img_name}] -> No match found in FAISS")

if __name__ == "__main__":
    run_diagnostic()
