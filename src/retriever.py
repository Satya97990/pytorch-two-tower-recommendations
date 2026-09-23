import faiss
import numpy as np

class FAISSRetriever:
    def __init__(self, embedding_path='artifacts/item_embeddings.npy', index_path='artifacts/faiss_index.bin'):
        self.embedding_path = embedding_path
        self.index_path = index_path
        self.dimension = 128
        self.index = None

    def build_and_save_index(self):
        print("Loading frozen movie coordinates...")
        # Load the embeddings exported from Phase 3
        embeddings = np.load(self.embedding_path).astype('float32')
        num_items, d = embeddings.shape

        print(f"Building FAISS IndexIVFPQ for {num_items} items...")
        
        # --- ENTERPRISE SCALE CLUSTERING PARAMETERS ---
        # nlist: Number of clusters to group the movies into. 
        nlist = 50  
        # m: Number of sub-vectors for Product Quantization (compresses the memory footprint)
        m = 8       
        nbits = 8   

        # 1. Define how we measure distance (Inner Product / Dot Product)
        quantizer = faiss.IndexFlatIP(d)
        
        # 2. Initialize the Inverted File with Product Quantization Index
        self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        
        # 3. Train the index so it can learn where the cluster centers are
        print("Training FAISS index clusters...")
        self.index.train(embeddings)
        
        # 4. Drop the movie coordinates into their respective clusters
        print("Adding movies to index...")
        self.index.add(embeddings)
        
        # 5. Save the index for Phase 5 (Streamlit)
        faiss.write_index(self.index, self.index_path)
        print(f"Success! FAISS index saved to {self.index_path}")

    def load_index(self):
        self.index = faiss.read_index(self.index_path)
        # nprobe dictates how many nearby clusters to search. Higher = more accurate but slightly slower.
        self.index.nprobe = 5  

    def search(self, user_vector, top_k=10):
        # Format the incoming user vector for FAISS
        if len(user_vector.shape) == 1:
            user_vector = np.expand_dims(user_vector, axis=0)
        user_vector = user_vector.astype('float32')

        # Run the sub-millisecond search
        scores, indices = self.index.search(user_vector, top_k)
        return scores[0], indices[0]

if __name__ == "__main__":
    # Execute Phase 4
    retriever = FAISSRetriever()
    retriever.build_and_save_index()
    
    # Test it by dropping a random mock user into the map
    retriever.load_index()
    mock_user_coordinate = np.random.rand(128).astype('float32')
    
    scores, movie_ids = retriever.search(mock_user_coordinate)
    print("\n--- FAISS Search Test ---")
    print(f"Top 10 Nearest Movie IDs: {movie_ids}")
    print(f"Similarity Scores: {scores}")