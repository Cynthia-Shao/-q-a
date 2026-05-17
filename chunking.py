import pandas as pd
import re
import os

# 创建 data 文件夹（如果不存在）
os.makedirs("data", exist_ok=True)

def split_text(text, max_length=300, overlap=50):
    """
    把一段长文本切成小片段
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        
        if end < len(text):
            for sep in ['。', '？', '！', '；', '.', '?', '!', ';']:
                last_period = text.rfind(sep, start, end)
                if last_period != -1 and last_period > start + max_length//2:
                    end = last_period + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks

def build_chunks_from_csv():
    """
    读取12306_cleaned.csv，切成碎片
    """
    # 读取清洗好的数据（注意路径：上一级目录）
    df = pd.read_csv("12306_cleaned_new.csv", encoding='utf-8-sig')
    
    all_chunks = []
    
    for idx, row in df.iterrows():
        question = row['原始问题']
        answer = row['原始答案']
        category = row.get('问题分类', '其他')
        
        chunks = split_text(str(answer))
        
        for chunk_id, chunk_text in enumerate(chunks):
            all_chunks.append({
                'chunk_id': f"doc_{idx}_{chunk_id}",
                'question': question,
                'answer_chunk': chunk_text,
                'category': category,
                'source_row': idx
            })
    
    chunks_df = pd.DataFrame(all_chunks)
    chunks_df.to_csv("data/chunks.csv", index=False, encoding='utf-8-sig')
    
    print(f"切分完成！原始问答 {len(df)} 条 → 切出 {len(all_chunks)} 个片段")
    print(f"保存到: data/chunks.csv")
    return chunks_df

if __name__ == "__main__":
    build_chunks_from_csv()