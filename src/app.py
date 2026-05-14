import os
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import ollama
from embedding import get_embedding  # 假设你的 embedding.py 有这个函数
from vector_store import search_similar  # 假设 vector_store.py 有检索函数

app = Flask(__name__)
CORS(app)

# 假设你已有的铁路 FAQ 知识库向量已存储好
# 这里简化了，实际需要加载你的向量数据库

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    
    # 1. 从知识库检索相关内容（RAG 核心）
    # 这里调用你的 vector_store 检索函数
    # retrieved_docs = search_similar(user_input, top_k=3)
    # context = "\n".join(retrieved_docs)
    
    # 2. 构建 prompt
    prompt = f"""你是铁路出行旅客常见问题问答助手。
请根据以下参考信息回答问题，如果信息不足则如实告知。
    
问题: {user_input}"""
    
    # 3. 调用本地的 Ollama 模型
    response = ollama.chat(model='qwen2.5:latest', messages=[
        {'role': 'user', 'content': prompt}
    ])
    
    return jsonify({'reply': response['message']['content']})

if __name__ == '__main__':
    app.run(debug=True, port=5000)