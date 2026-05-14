import numpy as np
import faiss
import pandas as pd

class VectorStore:
    def __init__(self, vectors_path="data/embeddings.npy", metadata_path="data/chunks_with_metadata.csv"):
        print("加载向量库...")
        self.vectors = np.load(vectors_path).astype('float32')
        self.metadata = pd.read_csv(metadata_path, encoding='utf-8-sig')
        
        dim = self.vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        
        faiss.normalize_L2(self.vectors)
        self.index.add(self.vectors)
        
        print(f"FAISS索引构建完成！共有 {self.index.ntotal} 条向量")
    
    def search(self, query_embedding, k=5):
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    'score': float(distances[0][i]),
                    'chunk_id': self.metadata.iloc[idx]['chunk_id'],
                    'question': self.metadata.iloc[idx]['question'],
                    'answer_chunk': self.metadata.iloc[idx]['answer_chunk'],
                    'category': self.metadata.iloc[idx]['category']
                })
        return results

if __name__ == "__main__":
    vs = VectorStore()
    print("向量库加载测试通过！")