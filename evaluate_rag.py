import os
import time
import numpy as np
import pandas as pd
import ollama

# 导入 retriever（它会加载模型和向量库）
from retriever import Retriever

# 配置
TEST_SET_PATH = "12306_final_testset.csv"
OLLAMA_MODEL = 'qwen2.5:latest'

def load_test_set():
    df = pd.read_csv(TEST_SET_PATH, encoding='utf-8')
    questions = df['question'].tolist()
    answers = df['answer'].tolist()
    print(f"加载测试集成功，共 {len(questions)} 个问题")
    return questions, answers

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def evaluate(questions, ground_truths, retriever, use_rag):
    scores = []
    for i, (q, gt) in enumerate(zip(questions, ground_truths)):
        print(f"处理第 {i+1}/{len(questions)} 个问题 (RAG={use_rag}): {q[:50]}...")
        
        if use_rag:
            retrieved = retriever.retrieve(q, k=3)
            if retrieved:
                context_parts = []
                for idx, item in enumerate(retrieved, 1):
                    context_parts.append(f"【参考{idx}】问题：{item['question']}\n答案片段：{item['answer_chunk']}")
                context = "\n\n".join(context_parts)
            else:
                context = "未找到直接相关的参考资料。"
            prompt = f"""你是铁路出行旅客常见问题问答助手。请严格依据下面提供的参考信息回答问题。
如果参考信息不足以回答用户问题，请如实告知“根据现有资料无法回答该问题”，不要编造信息。

### 参考信息：
{context}

### 用户问题：
{q}

### 回答："""
        else:
            prompt = f"""你是铁路出行旅客常见问题问答助手。请根据你自己的知识回答问题。
如果用户的问题与铁路出行无关，可以如实告知。

用户问题：{q}

回答："""

        try:
            response = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': prompt}])
            pred = response['message']['content']
        except Exception as e:
            pred = f"[ERROR] {str(e)}"

        # 使用 retriever 中已经加载的 embedding 模型计算相似度
        gt_emb = retriever.model.encode([gt])[0]
        pred_emb = retriever.model.encode([pred])[0]
        sim = cosine_similarity(gt_emb, pred_emb)
        scores.append(sim)
        print(f"  相似度: {sim:.4f}")
        time.sleep(0.5)
    return scores

def main():
    print("初始化检索器...")
    retriever = Retriever()
    
    questions, ground_truths = load_test_set()
    
    print("\n========== 评估未接入 RAG ==========")
    scores_no_rag = evaluate(questions, ground_truths, retriever, use_rag=False)
    avg_no_rag = np.mean(scores_no_rag)
    print(f"\n未接入 RAG 平均相似度: {avg_no_rag:.4f}")
    
    print("\n========== 评估接入 RAG 后 ==========")
    scores_rag = evaluate(questions, ground_truths, retriever, use_rag=True)
    avg_rag = np.mean(scores_rag)
    print(f"\n接入 RAG 后平均相似度: {avg_rag:.4f}")
    
    print("\n========== 对比结果 ==========")
    print(f"未接入 RAG: {avg_no_rag:.4f}")
    print(f"接入 RAG:   {avg_rag:.4f}")
    print(f"提升:       {avg_rag - avg_no_rag:.4f} ({((avg_rag - avg_no_rag) / avg_no_rag) * 100:.2f}%)")
    
    result_df = pd.DataFrame({
        'question': questions,
        'ground_truth': ground_truths,
        'score_no_rag': scores_no_rag,
        'score_rag': scores_rag
    })
    result_df.to_csv('evaluation_results.csv', index=False, encoding='utf-8-sig')
    print("\n详细结果已保存到 evaluation_results.csv")

if __name__ == '__main__':
    main()