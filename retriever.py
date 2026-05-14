import os
# 强制离线模式（不联网）
#os.environ['TRANSFORMERS_OFFLINE'] = '1'
#os.environ['HF_HUB_OFFLINE'] = '1'

from sentence_transformers import SentenceTransformer
from vector_store import VectorStore

class Retriever:
    def __init__(self):
        """启动时自动加载模型和向量库"""
        print("正在加载检索器...")
        # 加载 embedding 模型（仅使用本地缓存）
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.vector_store = VectorStore()
        print("检索器加载完成！")
    
    def retrieve(self, query, k=5):
        query_vec = self.model.encode([query])[0]
        results = self.vector_store.search(query_vec, k=k)
        return results