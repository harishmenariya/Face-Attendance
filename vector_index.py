import os
import faiss
import numpy as np

class VectorIndex:
    def __init__(self, index_path='models/face_index.faiss', dim=128):
        self.index_path = index_path
        self.dim = dim
        self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                print(f"[VectorIndex] Loaded existing index from {self.index_path} with {self.index.ntotal} vectors.")
            except Exception as e:
                print(f"[VectorIndex] Failed to load index: {e}. Creating a new one.")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        # Create a new IndexIDMap2 wrapping an IndexFlatL2
        # IndexIDMap2 is required to store integer IDs alongside vectors natively in FAISS
        flat_index = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIDMap2(flat_index)
        print(f"[VectorIndex] Initialized fresh FAISS IndexIDMap2(IndexFlatL2({self.dim})).")
        self.save()

    def add_vector(self, user_id: int, embedding: np.ndarray):
        """
        Normalizes and adds a 128-dimensional embedding to the FAISS index.
        """
        # Ensure embedding is 2D array of shape (1, 128) and float32
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)
        
        embedding = np.ascontiguousarray(embedding, dtype=np.float32)
        
        if embedding.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dim}, got {embedding.shape[1]}")
            
        # Normalize the embedding using L2 norm
        faiss.normalize_L2(embedding)
        
        # FAISS expects IDs as int64
        id_array = np.array([user_id], dtype=np.int64)
        
        # Add to index
        self.index.add_with_ids(embedding, id_array)
        self.save()
        return True

    def remove_vector(self, user_id: int):
        """Removes the vector associated with the given user_id from the index."""
        if self.index.ntotal == 0:
            return False
        selector = faiss.IDSelectorArray(np.array([user_id], dtype=np.int64))
        self.index.remove_ids(selector)
        self.save()
        return True

    def search(self, query_emb: np.ndarray, threshold=0.55):
        """
        Searches the index for the closest vector to query_emb.
        Returns (matched_user_id, distance) if successful, else (None, distance).
        """
        if self.index.ntotal == 0:
            return None, float('inf')
            
        # Ensure query is 2D array and float32
        if query_emb.ndim == 1:
            query_emb = np.expand_dims(query_emb, axis=0)
            
        query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)
        
        # Normalize the query embedding for cosine similarity equivalent using L2
        faiss.normalize_L2(query_emb)
        
        # Search for top 1 closest vector
        distances, indices = self.index.search(query_emb, k=1)
        
        matched_user_id = int(indices[0][0])
        distance = float(distances[0][0])
        
        if matched_user_id != -1 and distance <= threshold:
            return matched_user_id, distance
            
        return None, distance

    def save(self):
        """Saves the current index to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

if __name__ == '__main__':
    print("=== STEP 3: vector_index.py SELF-TEST ===")
    
    # Use a dummy path for the self-test
    test_index_path = 'models/test_face_index.faiss'
    vi = VectorIndex(index_path=test_index_path, dim=128)
    
    # 1. Create a mock embedding and add to index
    mock_id = 999
    np.random.seed(42)
    mock_emb = np.random.randn(128).astype(np.float32)
    
    print(f"1. Adding mock vector for User ID {mock_id}...")
    vi.add_vector(mock_id, mock_emb)
    print(f"   [OK] Vector added. Total vectors in index: {vi.index.ntotal}")
    
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
