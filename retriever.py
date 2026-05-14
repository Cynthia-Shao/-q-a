from sentence_transformers import SentenceTransformer
from vector_store import VectorStore

class Retriever:
    def __init__(self):
        """启动时自动加载模型和向量库"""
        print("正在加载检索器...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.vector_store = VectorStore()
        print("检索器加载完成！")
    
    def retrieve(self, query, k=5):
        """
        根据用户问题，检索最相关的答案片段
        
        参数:
            query: 用户的问题字符串，比如 "儿童怎么买票？"
            k: 返回多少个结果，默认5个
        
        返回:
            list of dict: 每个dict包含 answer_chunk, question, score 等
        """
        # 1. 把问题向量化
        query_vec = self.model.encode([query])[0]
        
        # 2. 去向量库里搜索
        results = self.vector_store.search(query_vec, k=k)
        
        # 3. 直接返回结果
        return results

 