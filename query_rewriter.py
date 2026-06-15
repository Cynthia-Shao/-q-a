import ollama

REWRITE_MODEL = "qwen2.5:latest"
REWRITE_PROMPT = """你是一个铁路出行领域的问题规范化助手。请将用户的口语化问题改写为一个更规范、更具体的提问，便于在12306官方知识库中检索答案。

改写要求：
1. 保持原问题的核心意图不变
2. 使用更正式的书面表达
3. 将模糊指代替换为明确的名词
4. 如果原问题涉及具体数字、时间、比例，必须保留

只输出改写后的问题，不要任何解释、标签或格式。

用户问题：{question}
改写后的问题："""


def rewrite_query(question: str, model: str = REWRITE_MODEL) -> str:
    prompt = REWRITE_PROMPT.format(question=question)
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "seed": 42},
        )
        rewritten = response["message"]["content"].strip()
        if len(rewritten) < 3:
            return question
        return rewritten
    except Exception:
        return question


class QueryRewriter:
    def __init__(self, model: str = REWRITE_MODEL):
        self.model = model
        self._cache = {}

    def rewrite(self, question: str) -> str:
        if question in self._cache:
            return self._cache[question]
        rewritten = rewrite_query(question, self.model)
        self._cache[question] = rewritten
        return rewritten
