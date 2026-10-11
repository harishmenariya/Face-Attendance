import os
import numpy as np

class VectorIndex:
    def __init__(self, index_path='models/face_index.npz', dim=128):
        # Update extension if old .faiss path is passed
        if index_path.endswith('.faiss'):
            index_path = index_path.replace('.faiss', '.npz')
        self.index_path = index_path
        self.dim = dim
        self.embeddings = None
        self.user_ids = None
        self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(self.index_path):
            try:
                data = np.load(self.index_path)
                self.embeddings = data['embeddings']
                self.user_ids = data['user_ids']
                print(f"[VectorIndex] Loaded existing index from {self.index_path} with {len(self.user_ids)} vectors.")
            except Exception as e:
                print(f"[VectorIndex] Failed to load index: {e}. Creating a new one.")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        self.embeddings = np.empty((0, self.dim), dtype=np.float32)
        self.user_ids = np.empty((0,), dtype=np.int64)
        print(f"[VectorIndex] Initialized fresh NumPy Index ({self.dim}d).")
        self.save()

    @property
    def ntotal(self):
        return len(self.user_ids)

    def _normalize_L2(self, x):
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        # Avoid division by zero
        norm[norm == 0] = 1e-10
        return x / norm

    def add_vector(self, user_id: int, embedding: np.ndarray):
        """
        Normalizes and adds a 128-dimensional embedding to the index.
        """
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)
        
        embedding = np.ascontiguousarray(embedding, dtype=np.float32)
        
        if embedding.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dim}, got {embedding.shape[1]}")
            
        embedding = self._normalize_L2(embedding)
        
        id_array = np.array([user_id], dtype=np.int64)
        
        self.embeddings = np.vstack((self.embeddings, embedding))
        self.user_ids = np.concatenate((self.user_ids, id_array))
        
        self.save()
        return True

    def remove_vector(self, user_id: int):
        """Removes the vector associated with the given user_id from the index."""
        if self.ntotal == 0:
            return False
            
        mask = self.user_ids != user_id
        if np.all(mask):
            return False
            
        self.embeddings = self.embeddings[mask]
        self.user_ids = self.user_ids[mask]
        
        self.save()
        return True

    def search(self, query_emb: np.ndarray, threshold=0.55):
        """
        Searches the index for the closest vector to query_emb.
        Returns (matched_user_id, distance) if successful, else (None, distance).
        """
        if self.ntotal == 0:
            return None, float('inf')
            
        if query_emb.ndim == 1:
            query_emb = np.expand_dims(query_emb, axis=0)
            
        query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)
        query_emb = self._normalize_L2(query_emb)
        
        # Calculate L2 squared distances (matches faiss.IndexFlatL2 behavior)
        diff = self.embeddings - query_emb
        distances = np.sum(diff ** 2, axis=1)
        
        best_idx = np.argmin(distances)
        best_dist = float(distances[best_idx])
        best_id = int(self.user_ids[best_idx])
        
        if best_dist <= threshold:
            return best_id, best_dist
            
        return None, best_dist

    def save(self):
        """Saves the current index to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        np.savez(self.index_path, embeddings=self.embeddings, user_ids=self.user_ids)

if __name__ == '__main__':
    print("=== STEP 3: vector_index.py SELF-TEST ===")
    
    # Use a dummy path for the self-test
    test_index_path = 'models/test_face_index.npz'
    vi = VectorIndex(index_path=test_index_path, dim=128)
    
    # 1. Create a mock embedding and add to index
    mock_id = 999
    np.random.seed(42)
    mock_emb = np.random.randn(128).astype(np.float32)
    
    print(f"1. Adding mock vector for User ID {mock_id}...")
    vi.add_vector(mock_id, mock_emb)
    print(f"   [OK] Vector added. Total vectors in index: {vi.ntotal}")
    
    # 2. Perform a search query with the exact same embedding
    print(f"2. Querying the exact same vector...")
    found_id, dist = vi.search(mock_emb.copy(), threshold=0.55)
    
    if found_id == mock_id:
        print(f"   [OK] Match successful! Found ID {found_id} with distance {dist:.5f}")
    else:
        print(f"   [FAIL] Search returned ID {found_id} with distance {dist:.5f}")
        
    # 3. Query with random noise (should fail the threshold)
    print(f"3. Querying with random noise...")
    noise_emb = np.random.randn(128).astype(np.float32)
    found_id_noise, dist_noise = vi.search(noise_emb, threshold=0.55)
    
    if found_id_noise is None:
        print(f"   [OK] Noise successfully rejected. Distance {dist_noise:.5f} > threshold 0.55")
    else:
        print(f"   [FAIL] Noise was incorrectly matched to ID {found_id_noise} with distance {dist_noise:.5f}")

    # Clean up test index
    if os.path.exists(test_index_path):
        os.remove(test_index_path)
    
    print("=== SELF-TEST COMPLETE ===")
