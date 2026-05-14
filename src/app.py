import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import os
import sys
sys.path.insert(0, r'C:\Users\Lenovo\Desktop\-q-a-main')
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import ollama

# 导入你自己写的检索器
from retriever import Retriever

app = Flask(__name__)
CORS(app)

# 初始化检索器（会自动加载向量库和 embedding 模型）
print("正在初始化检索器...")
retriever = Retriever()
print("检索器就绪")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    if not user_input:
        return jsonify({'reply': '请输入问题。'})

    # 1. 从知识库中检索相关片段（RAG 核心）
    retrieved = retriever.retrieve(user_input, k=3)  # 检索最相关的3个片段
    
    # 2. 构建上下文
    if retrieved:
        context_parts = []
        for idx, item in enumerate(retrieved, 1):
            context_parts.append(f"【参考{idx}】问题：{item['question']}\n答案片段：{item['answer_chunk']}")
        context = "\n\n".join(context_parts)
    else:
        context = "未找到直接相关的参考资料。"

    # 3. 构建 prompt（引导模型基于检索结果回答）
    prompt = f"""你是铁路出行旅客常见问题问答助手。请严格依据下面提供的参考信息回答问题。
如果参考信息不足以回答用户问题，请如实告知“根据现有资料无法回答该问题”，不要编造信息。

### 参考信息：
{context}

### 用户问题：
{user_input}

### 回答："""

    # 4. 调用本地 Ollama 模型
    try:
        response = ollama.chat(model='qwen2.5:latest', messages=[
            {'role': 'user', 'content': prompt}
        ])
        answer = response['message']['content']
    except Exception as e:
        answer = f"调用模型出错：{str(e)}。请确保 Ollama 正在运行（系统托盘有羊驼图标）。"

    return jsonify({'reply': answer})

if __name__ == '__main__':
    app.run(debug=True, port=5000)