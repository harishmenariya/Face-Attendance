import sqlite3
import numpy as np
import faiss
from vector_index import VectorIndex

def resequence_all_uids():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # 1. Fetch existing sorted users
    cursor.execute("SELECT id, user_code, name, registered_date FROM users ORDER BY id ASC")
    users = cursor.fetchall()

    if not users:
        print("No users found.")
        conn.close()
        return

    # Map old ID to new ID
    id_map = {}
    new_users_data = []
    for new_id, user in enumerate(users, start=1):
        old_id, code, name, reg_date = user
        id_map[old_id] = new_id
        new_users_data.append((new_id, code, name, reg_date))

    # 2. Re-index vectors in FAISS
    v_index = VectorIndex()
    total_vectors = v_index.index.ntotal
    
    new_index = faiss.IndexIDMap2(faiss.IndexFlatL2(128))

    for old_id, new_id in id_map.items():
        try:
            # Reconstruct old vector using its internal FAISS ID
            vec = v_index.index.reconstruct(int(old_id)).reshape(1, -1)
            new_index.add_with_ids(vec, np.array([new_id], dtype=np.int64))
        except Exception as e:
            print(f"Warning: Failed to recover vector for Old UID {old_id}: {e}")

    # Save new FAISS index
    faiss.write_index(new_index, "models/face_index.faiss")

    # 3. Clean and re-insert into SQLite
    cursor.execute("DROP TABLE IF EXISTS users_backup")
    cursor.execute("CREATE TABLE users_temp (id INTEGER PRIMARY KEY, user_code TEXT UNIQUE, name TEXT, registered_date TEXT)")
    cursor.executemany("INSERT INTO users_temp (id, user_code, name, registered_date) VALUES (?, ?, ?, ?)", new_users_data)
    
    cursor.execute("DROP TABLE users")
    cursor.execute("ALTER TABLE users_temp RENAME TO users")

    # Reset SQLite autoincrement sequence
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='users'")
    cursor.execute("INSERT INTO sqlite_sequence (name, seq) VALUES ('users', ?)", (len(new_users_data),))

    conn.commit()
    conn.close()

    print(f"Success! {len(new_users_data)} users re-sequenced to UIDs 1 to {len(new_users_data)} in both SQLite and FAISS.")

if __name__ == "__main__":
    resequence_all_uids()
