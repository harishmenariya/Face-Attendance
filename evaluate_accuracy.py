import os
import cv2
import numpy as np
import db_manager as db
from vector_index import VectorIndex
from face_engine import FaceEngine

def evaluate_pipeline(eval_dir="evaluation_data", threshold=0.55):
    db.init_db()
    v_index = VectorIndex()
    engine = FaceEngine()

    tp = 0  # True Positive: Registered user sahi identify hua
    fp = 0  # False Positive: Kisi aur ka face galat match ho gaya
    fn = 0  # False Negative: Registered user ko Unknown bol diya
    tn = 0  # True Negative: Unknown person ko correctly Unknown reject kiya

    # 1. Test Registered Faces
    reg_path = os.path.join(eval_dir, "registered")
    if os.path.exists(reg_path):
        for user_code in os.listdir(reg_path):
            person_folder = os.path.join(reg_path, user_code)
            if not os.path.isdir(person_folder):
                continue

            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                faces = engine.detect_faces(img)
                if faces is None or len(faces) == 0:
                    fn += 1
                    continue

                # Crop & extract embedding
                x, y, w, h = int(faces[0][0]), int(faces[0][1]), int(faces[0][2]), int(faces[0][3])
                crop = img[max(0, y):max(0, y)+h, max(0, x):max(0, x)+w]
                if crop.size == 0:
                    fn += 1
                    continue

                emb = engine.extract_embedding(crop)
                matched_id, dist = v_index.search(emb, threshold=threshold)

                if matched_id is not None:
                    pred_code, _ = db.get_user_by_id(matched_id)
                    if pred_code == user_code:
                        tp += 1
                    else:
                        fp += 1  # Galat user match ho gaya
                else:
                    fn += 1  # Genuine user reject ho gaya

    # 2. Test Unknown Faces
    unk_path = os.path.join(eval_dir, "unknown")
    if os.path.exists(unk_path):
        for img_name in os.listdir(unk_path):
            img_path = os.path.join(unk_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = engine.detect_faces(img)
            if faces is None or len(faces) == 0:
                tn += 1
                continue

            x, y, w, h = int(faces[0][0]), int(faces[0][1]), int(faces[0][2]), int(faces[0][3])
            crop = img[max(0, y):max(0, y)+h, max(0, x):max(0, x)+w]
            if crop.size == 0:
                tn += 1
                continue

            emb = engine.extract_embedding(crop)
            matched_id, dist = v_index.search(emb, threshold=threshold)

            if matched_id is not None:
                fp += 1  # Unknown person ko registered maan liya
            else:
                tn += 1  # Correctly rejected

    total_samples = tp + fp + fn + tn
    if total_samples == 0:
        print("Koi test images nahi mili. evaluation_data folder me images add karein.")
        return

    accuracy = ((tp + tn) / total_samples) * 100
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    far = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
    frr = (fn / (tp + fn) * 100) if (tp + fn) > 0 else 0.0

    print("=" * 45)
    print("      FACE RECOGNITION EVALUATION REPORT      ")
    print("=" * 45)
    print(f"Total Test Images: {total_samples}")
    print(f"True Positives (Correct Matches): {tp}")
    print(f"True Negatives (Correct Unknowns): {tn}")
    print(f"False Positives (Wrong Identity): {fp}")
    print(f"False Negatives (Missed Genuine): {fn}")
    print("-" * 45)
    print(f"Overall Accuracy : {accuracy:.2f}%")
    print(f"Precision        : {precision:.2f}%")
    print(f"Recall (TPR)     : {recall:.2f}%")
    print(f"FAR (False Accept): {far:.2f}%")
    print(f"FRR (False Reject): {frr:.2f}%")
    print("=" * 45)

if __name__ == "__main__":
    evaluate_pipeline()
