import os
# 使用国内镜像站（hf-mirror.com）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 如果本地缓存完整，也可以同时开启离线模式，但镜像已足够
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

os.makedirs("data", exist_ok=True)

def embed_chunks():
    # 读取碎片
    df = pd.read_csv("data/chunks.csv", encoding='utf-8-sig')
    
    texts_to_embed = []
    for _, row in df.iterrows():
        combined = f"问题：{row['question']}\n答案片段：{row['answer_chunk']}"
        texts_to_embed.append(combined)
    
    print(f"加载模型 {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"开始向量化 {len(texts_to_embed)} 个文本片段...")
    embeddings = []
    batch_size = 32
    for i in tqdm(range(0, len(texts_to_embed), batch_size)):
        batch = texts_to_embed[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    
    embeddings_np = np.array(embeddings).astype('float32')
    
    np.save("data/embeddings.npy", embeddings_np)
    df.to_csv("data/chunks_with_metadata.csv", index=False, encoding='utf-8-sig')
    
    print(f"向量化完成！向量维度: {embeddings_np.shape}")
    print(f"向量保存到: data/embeddings.npy")

if __name__ == "__main__":
    embed_chunks()