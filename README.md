# 铁路旅客智能问答系统（RAG）

基于 RAG（检索增强生成）的铁路出行旅客常见问题问答助手。使用 12306 官方知识库、混合检索策略和本地大模型，提供准确的铁路出行规则咨询。

## 技术架构

| 组件 | 技术选型 |
|------|---------|
| 知识库 | 12306 官方 Q&A 数据（272 条） |
| 向量化 | Sentence-Transformers（paraphrase-multilingual-MiniLM-L12-v2） |
| 向量索引 | FAISS（内积相似度） |
| 关键词检索 | BM25（中文字符 Bigram 分词） |
| 融合策略 | RRF（Reciprocal Rank Fusion） |
| 生成模型 | Qwen2.5（通过 Ollama 本地部署） |
| Web 界面 | Flask + 前端模板 |

## 检索流程

用户问题 → 向量检索（FAISS 语义匹配）+ 关键词检索（BM25 精确匹配）→ RRF 分数融合 → Top-3 注入 LLM Prompt → 生成回答

## 快速开始

### 1. 启动 Ollama

```bash
ollama serve
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：Python 3.13 无法直接安装 faiss-cpu，建议使用 Python 3.10~3.12。

### 3. 启动 Web 问答

```bash
python src/app.py
```

浏览器打开 http://127.0.0.1:5000

### 4. 运行评估

```bash
python evaluate_rag.py                    # 完整 75 题评估
python evaluate_rag.py --limit 5          # 快速测试前 5 题
```

评估产出：
- `evaluation_results.csv` — 逐题详细结果
- `evaluation_summary.csv` — 按题型分组汇总
- `evaluation_report.html` — 可视化报告

## 项目结构

```
├── src/                    # Web 应用
│   ├── app.py              # Flask 服务入口
│   ├── templates/          # 前端模板
│   └── data/               # 向量库数据
├── retriever.py            # 混合检索器（向量 + BM25 + RRF）
├── vector_store.py         # FAISS 向量库封装
├── query_rewriter.py       # 查询改写模块
├── evaluate_rag.py         # 多模式评估脚本（纯LLM / 向量RAG / 混合RAG）
├── data/                   # 知识库数据与评估集
├── scripts/                # 数据构建脚本
├── eval_results/           # 历史评估结果
└── reports/                # 文档报告
```

## 评估数据

75 题评估集覆盖三类问题：

| 类型 | 数量 | 说明 |
|------|------|------|
| 规则细节 | 65 | 具体条款、时间、费用、比例 |
| 简单常识 | 5 | 换说法/常识性覆盖 |
| 资料外问题 | 5 | 知识库未覆盖的实时/非规则问题 |

混合检索整体语义相似度 0.68（纯向量 0.37），规则细节类提升最显著（0.35 → 0.71）。
