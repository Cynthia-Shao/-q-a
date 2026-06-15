import os

USE_RAG = os.environ.get('USE_RAG', 'true').lower() == 'true'

# 国内网络可设置 HF 镜像加速模型下载
if os.environ.get('HF_ENDPOINT') is None:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import ollama

from retriever import HybridRetriever

app = Flask(__name__)
CORS(app)

print("正在初始化检索器...")
retriever = HybridRetriever()
print("检索器就绪")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    if not user_input:
        return jsonify({'reply': '请输入问题。'})

    if USE_RAG:
        retrieved = retriever.retrieve_hybrid(user_input, k=3)
        if retrieved:
            context_parts = []
            for idx, item in enumerate(retrieved, 1):
                context_parts.append(
                    f"【参考{idx}】问题：{item['question']}\n"
                    f"答案片段：{item['answer_chunk']}"
                )
            context = "\n\n".join(context_parts)
        else:
            context = "未找到直接相关的参考资料。"

        prompt = f"""你是铁路出行旅客常见问题问答助手。请严格依据下面提供的参考信息回答问题。
如果参考信息不足以回答用户问题，请如实告知"根据现有资料无法回答该问题"，不要编造信息。

### 参考信息：
{context}

### 用户问题：
{user_input}

### 回答："""
    else:
        prompt = f"""你是铁路出行旅客常见问题问答助手。请根据你自己的知识回答问题。
如果用户的问题与铁路出行无关，可以如实告知。

用户问题：{user_input}

回答："""

    try:
        response = ollama.chat(model='qwen2.5:latest', messages=[
            {'role': 'user', 'content': prompt}
        ])
        answer = response['message']['content']
    except Exception as e:
        answer = f"调用模型出错：{str(e)}。请确保 Ollama 正在运行。"

    return jsonify({'reply': answer})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
