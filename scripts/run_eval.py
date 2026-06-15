"""一键评估：加载模型 → 跑15题 → 生成可视化 HTML"""
import sys, time, json, re, warnings, os
warnings.filterwarnings("ignore")
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import pandas as pd
import ollama
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

OLLAMA_MODEL = "qwen2.5:latest"
RETRIEVE_TOP_K = 3
ANSWER_TRUNCATE = 200
SEMANTIC_PASS_THRESHOLD = 0.75
OLLAMA_OPTIONS = {"temperature": 0, "top_p": 1, "seed": 42}

# ========== Tokenizer (Char Bigram for Chinese) ==========
def _tokenize(text):
    """Chinese-aware character bigram tokenizer for BM25."""
    clean = re.sub(r'[^\u4e00-\u9fff\w]', ' ', text)
    tokens = []
    chars = []
    for ch in clean:
        if ch == ' ':
            if chars:
                tokens.extend(_char_bigrams(chars))
                chars = []
        elif '\u4e00' <= ch <= '\u9fff':
            chars.append(ch)
        else:
            if chars:
                tokens.extend(_char_bigrams(chars))
                chars = []
            tokens.append(ch.lower())
    if chars:
        tokens.extend(_char_bigrams(chars))
    return tokens

def _char_bigrams(chars):
    if len(chars) == 1:
        return chars
    return chars + [chars[i] + chars[i+1] for i in range(len(chars) - 1)]

# ========== VectorStore ==========
class VS:
    def __init__(self, vp="data/embeddings.npy", mp="data/chunks_with_metadata.csv"):
        self.vectors = np.load(vp).astype('float32')
        self.metadata = pd.read_csv(mp, encoding='utf-8-sig')
        dim = self.vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(self.vectors)
        self.index.add(self.vectors)
    def search(self, qv, k=5):
        qv = qv.reshape(1, -1)
        faiss.normalize_L2(qv)
        dists, idxs = self.index.search(qv, k)
        results = []
        for i, idx in enumerate(idxs[0]):
            if idx != -1:
                results.append({
                    'score': float(dists[0][i]),
                    'chunk_id': self.metadata.iloc[idx]['chunk_id'],
                    'question': self.metadata.iloc[idx]['question'],
                    'answer_chunk': self.metadata.iloc[idx]['answer_chunk'],
                    'category': self.metadata.iloc[idx]['category'],
                })
        return results

# ========== BM25 ==========
def build_bm25(vs):
    docs = []
    for _, row in vs.metadata.iterrows():
        docs.append(f"{row['question']} {row['answer_chunk']}")
    tokenized = [_tokenize(d) for d in docs]
    return BM25Okapi(tokenized), docs, tokenized

def bm25_search(query, bm25_obj, vs, k=10):
    tokenized = _tokenize(query)
    scores = bm25_obj.get_scores(tokenized)
    top = np.argsort(scores)[::-1][:k]
    mx = max(scores) if len(scores) > 0 else 1.0
    results = []
    for idx in top:
        meta = vs.metadata.iloc[idx]
        results.append({
            'score': float(scores[idx] / mx) if mx > 0 else 0.0,
            'chunk_id': meta['chunk_id'],
            'question': meta['question'],
            'answer_chunk': meta['answer_chunk'],
            'category': meta['category'],
        })
    return results

# ========== Hybrid Retrieve ==========
def retrieve_hybrid(query, model, vs, bm25_obj, k=5, vec_k=10, bm25_k=10):
    qv = model.encode([query])[0]
    vec_results = vs.search(qv, k=vec_k)
    bm25_results = bm25_search(query, bm25_obj, vs, k=bm25_k)
    rrf = {}
    for rank, item in enumerate(vec_results):
        rrf[item['chunk_id']] = rrf.get(item['chunk_id'], 0) + 1.0 / (rank + 60)
    for rank, item in enumerate(bm25_results):
        rrf[item['chunk_id']] = rrf.get(item['chunk_id'], 0) + 1.0 / (rank + 60)
    sorted_ids = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    id_to_item = {}
    for item in vec_results + bm25_results:
        if item['chunk_id'] not in id_to_item:
            id_to_item[item['chunk_id']] = item
    merged = []
    for cid, score in sorted_ids[:k]:
        item = dict(id_to_item[cid])
        item['score'] = round(score, 6)
        merged.append(item)
    return merged

def retrieve_vector(query, model, vs, k=5):
    qv = model.encode([query])[0]
    return vs.search(qv, k=k)

# ========== Call LLM ==========
def call_llm(prompt):
    try:
        resp = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}],
                           options=OLLAMA_OPTIONS)
        return resp["message"]["content"]
    except Exception as e:
        return f"[LLM_ERROR] {e}"

# ========== Prompts ==========
def build_prompt(question, retrieved, use_rag):
    if not use_rag:
        return f"""你是铁路出行旅客常见问题问答助手。请根据你自己的知识回答问题。
如果问题涉及实时信息、资料外信息或你不确定，请明确说明无法确认，不要编造。

用户问题：
{question}

回答："""
    if retrieved:
        parts = []
        for idx, item in enumerate(retrieved, 1):
            chunk = item['answer_chunk']
            if len(chunk) > ANSWER_TRUNCATE:
                chunk = chunk[:ANSWER_TRUNCATE] + "..."
            parts.append(f"【参考{idx}】相似度：{item['score']:.4f}\n原问题：{item['question']}\n答案片段：{chunk}")
        context = "\n\n".join(parts)
    else:
        context = "未找到直接相关的参考资料。"
    return f"""你是铁路出行旅客常见问题问答助手。请严格依据参考信息回答问题。
要求：
1. 优先回答具体规则、时间、费用、比例、证件、材料等关键信息。
2. 如果参考信息不足以回答，请说"根据现有资料无法回答该问题"，不要编造。
3. 回答末尾用一句话说明依据了哪些参考编号，例如"依据：参考1、参考2"。

### 参考信息
{context}

### 用户问题
{question}

### 回答"""

# ========== Evaluation ==========
def cosine_sim(a, b):
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a/an, b/bn)) if an > 0 and bn > 0 else 0.0

def extract_facts(text):
    patterns = [
        r"\d+(?:\.\d+)?\s*(?:年|日|天|小时|分钟|次|元|%|折|公斤|千克|厘米|公里)",
        r"[一二三四五六七八九十百千万]+(?:年|日|天|小时|分钟|次|元|折|公斤|千克|厘米|公里)",
        r"(?:免费|收费|不收费|不得|不能|可以|应当|必须|有效|无效|退还|补收|删除)",
    ]
    facts = []
    for p in patterns:
        facts.extend(re.findall(p, text))
    seen = set()
    unique = []
    for f in facts:
        n = re.sub(r"\s+", "", f)
        if n and n not in seen:
            seen.add(n); unique.append(n)
    return unique

def fact_recall(gt, pred):
    facts = extract_facts(gt)
    if not facts:
        return None
    hits = sum(1 for f in facts if f in pred.replace(" ", ""))
    return hits / len(facts)

def eval_q(question, gt, model, vs, bm25_obj, use_rag, method):
    if not use_rag:
        retrieved = []
    elif method == "hybrid":
        retrieved = retrieve_hybrid(question, model, vs, bm25_obj, k=RETRIEVE_TOP_K)
    else:
        retrieved = retrieve_vector(question, model, vs, k=RETRIEVE_TOP_K)
    prompt = build_prompt(question, retrieved, use_rag)
    try:
        pred = call_llm(prompt)
        err = ""
    except Exception as e:
        pred = f"[ERROR] {e}"; err = str(e)
    gt_emb = model.encode([gt])[0]
    pred_emb = model.encode([pred])[0]
    sim = cosine_sim(gt_emb, pred_emb)
    fr = fact_recall(gt, pred)
    return {
        "prediction": pred, "similarity": sim,
        "semantic_pass": int(sim >= SEMANTIC_PASS_THRESHOLD),
        "key_fact_recall": fr,
        "top_score": retrieved[0]["score"] if retrieved else None,
        "top_question": retrieved[0]["question"] if retrieved else "",
        "error": err,
    }

# ========== Main ==========
print("1/4 加载嵌入模型..."); sys.stdout.flush()
model_enc = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("2/4 加载向量库+BM25..."); sys.stdout.flush()
vs = VS()
bm25, _, _ = build_bm25(vs)
print("3/4 加载测试集..."); sys.stdout.flush()

df_all = pd.read_csv("data/12306_rag_eval_dataset.csv", encoding="utf-8")
df_all = df_all.dropna(subset=["question", "answer"]).reset_index(drop=True)
if "test_type" in df_all.columns:
    mapping = {"detail": "rule_detail", "boundary": "rule_detail",
               "composition": "rule_detail", "paraphrase": "simple",
               "out_of_scope": "out_of_scope"}
    df_all["eval_group"] = df_all["test_type"].map(lambda x: mapping.get(str(x), "simple"))
else:
    df_all["eval_group"] = "all"

methods = ["no_rag", "vector", "hybrid"]
labels = {"no_rag": "纯LLM", "vector": "向量RAG", "hybrid": "混合RAG"}

N = min(10, len(df_all))
df = df_all.head(N)
print(f"4/4 开始评估 {N} 题 x 3 模式...\n"); sys.stdout.flush()

rows = []
csv_path = "eval_v2.csv"
sum_path = "eval_v2_summary.csv"
html_path = "eval_v2.html"
for i, row in df.iterrows():
    q = row["question"]; gt = row["answer"]; grp = row["eval_group"]
    print(f"[{i+1}/{N}] [{grp}] {q[:45]}...")
    res = {}
    for m in methods:
        if m == "no_rag":
            r = eval_q(q, gt, model_enc, vs, bm25, use_rag=False, method="vector")
        elif m == "hybrid":
            r = eval_q(q, gt, model_enc, vs, bm25, use_rag=True, method="hybrid")
        else:
            r = eval_q(q, gt, model_enc, vs, bm25, use_rag=True, method="vector")
        res[m] = r
        print(f"  {labels[m]:8s} sim={r['similarity']:.4f}  fact={r.get('key_fact_recall') or 'N/A'}")

    row_data = {
        "question": q, "ground_truth": gt, "eval_group": grp,
        "source_category": row.get("category", ""),
        "difficulty": row.get("difficulty", ""),
    }
    for m in methods:
        row_data[f"answer_{m}"] = res[m]["prediction"]
        row_data[f"sim_{m}"] = res[m]["similarity"]
        row_data[f"pass_{m}"] = res[m]["semantic_pass"]
        row_data[f"fact_{m}"] = res[m]["key_fact_recall"]
        row_data[f"top_score_{m}"] = res[m]["top_score"]
        row_data[f"top_q_{m}"] = res[m]["top_question"]
        row_data[f"error_{m}"] = res[m]["error"]
    for m in ["vector", "hybrid"]:
        row_data[f"improvement_{m}"] = res[m]["similarity"] - res["no_rag"]["similarity"]
    rows.append(row_data)
    # Incremental save every question
    tmp_details = pd.DataFrame(rows)
    tmp_details.to_csv(csv_path, index=False, encoding="utf-8-sig")
    time.sleep(0.3)

details = pd.DataFrame(rows)

# Summary
sum_rows = []
for grp_name, grp_df in details.groupby("eval_group", dropna=False):
    sr = {"eval_group": grp_name, "count": len(grp_df)}
    for m in methods:
        sr[f"avg_sim_{m}"] = grp_df[f"sim_{m}"].mean()
        sr[f"pass_rate_{m}"] = grp_df[f"pass_{m}"].mean()
        fv = grp_df[f"fact_{m}"].dropna()
        sr[f"avg_fact_{m}"] = fv.mean() if len(fv) > 0 else None
    for m in ["vector", "hybrid"]:
        sr[f"avg_imp_{m}"] = grp_df[f"improvement_{m}"].mean()
    sum_rows.append(sr)
total = {"eval_group": "all", "count": len(details)}
for m in methods:
    total[f"avg_sim_{m}"] = details[f"sim_{m}"].mean()
    total[f"pass_rate_{m}"] = details[f"pass_{m}"].mean()
    fv = details[f"fact_{m}"].dropna()
    total[f"avg_fact_{m}"] = fv.mean() if len(fv) > 0 else None
for m in ["vector", "hybrid"]:
    total[f"avg_imp_{m}"] = details[f"improvement_{m}"].mean()
sum_rows.append(total)
summary = pd.DataFrame(sum_rows)

details.to_csv("eval_v2.csv", index=False, encoding="utf-8-sig")
summary.to_csv("eval_v2_summary.csv", index=False, encoding="utf-8-sig")

# ========== HTML Report ==========
summary_json = json.dumps(summary.to_dict(orient="records"), ensure_ascii=False)
methods_json = json.dumps(methods, ensure_ascii=False)
labels_json = json.dumps(labels, ensure_ascii=False)

html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG 多模式对比评估报告</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, Segoe UI, system-ui, sans-serif; background: #f5f5f5; color: #2c2c2a; padding: 24px; }}
h1 {{ font-size: 22px; font-weight: 500; margin-bottom: 6px; }}
h2 {{ font-size: 16px; font-weight: 500; margin: 24px 0 10px; }}
.subtitle {{ font-size: 13px; color: #888; margin-bottom: 20px; }}
.card {{ background: #fff; border-radius: 12px; padding: 18px 22px; margin-bottom: 14px; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; margin-bottom: 14px; }}
.metric {{ background: #f7f8fa; border-radius: 8px; padding: 12px 14px; }}
.metric-label {{ font-size: 12px; color: #888; }}
.metric-value {{ font-size: 22px; font-weight: 500; margin-top: 4px; }}
.metric-value.good {{ color: #1d9e75; }}
.metric-value.better {{ color: #378add; }}
.metric-detail {{ font-size: 11px; color: #aaa; margin-top: 3px; }}
.chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }}
@media (max-width: 700px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
.chart-wrap {{ position: relative; height: 290px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th, td {{ padding: 6px 10px; text-align: left; border-bottom: 0.5px solid #e5e5e5; }}
th {{ font-weight: 500; color: #888; font-size: 11px; }}
td.pos {{ color: #1d9e75; }}
td.neg {{ color: #e24b4a; }}
</style>
</head>
<body>
<h1>RAG 多模式对比评估报告</h1>
<p class="subtitle">测试集：{N} 题（规则细节 / 简单常识 / 资料外）| 模型：{OLLAMA_MODEL} | 检索 Top-{RETRIEVE_TOP_K}</p>

<div class="summary-grid" id="metricsGrid"></div>

<div class="card">
  <h2>按题型分组 — 平均语义相似度</h2>
  <div class="chart-wrap"><canvas id="barSim" role="img" aria-label="分组相似度对比"></canvas></div>
</div>

<div class="card">
  <h2>相对于纯LLM的语义相似度提升</h2>
  <div class="chart-wrap"><canvas id="barImp" role="img" aria-label="相似度提升对比"></canvas></div>
</div>

<div class="card">
  <h2>逐题详细对比</h2>
  <div style="overflow-x:auto;"><table id="detailTable"></table></div>
</div>

<script>
const methods = {methods_json};
const labels = {labels_json};
const summary = {summary_json};
const allRow = summary.find(r => r.eval_group === 'all');
const groups = summary.filter(r => r.eval_group !== 'all');
const palette = ['#888780', '#378add', '#1d9e75'];
const bgPalette = ['#d3d1c7', '#b5d4f4', '#9fe1cb'];

function fmt(v) {{ return v != null ? v.toFixed(4) : 'N/A'; }}
function pct(v) {{ return v != null ? (v * 100).toFixed(1) + '%' : 'N/A'; }}

let gh = '';
methods.forEach((m, i) => {{
    const sim = allRow ? allRow['avg_sim_' + m] : null;
    const pr = allRow ? allRow['pass_rate_' + m] : null;
    const fv = allRow ? allRow['avg_fact_' + m] : null;
    gh += `<div class="metric">
      <div class="metric-label">${{labels[m]}}</div>
      <div class="metric-value ${{i > 0 ? 'better' : ''}}">${{fmt(sim)}}</div>
      <div class="metric-detail">通过率 ${{pct(pr)}} | 事实命中 ${{fmt(fv)}}</div>
    </div>`;
}});
document.getElementById('metricsGrid').innerHTML = gh;

new Chart(document.getElementById('barSim').getContext('2d'), {{
    type: 'bar',
    data: {{
        labels: groups.map(g => g.eval_group),
        datasets: methods.map((m, i) => ({{
            label: labels[m], data: groups.map(g => g['avg_sim_' + m]),
            backgroundColor: bgPalette[i], borderColor: palette[i], borderWidth: 1.5,
        }}))
    }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ position: 'bottom' }} }},
        scales: {{ y: {{ beginAtZero: true, max: 1, ticks: {{ callback: v => v.toFixed(2) }} }} }}
    }}
}});

new Chart(document.getElementById('barImp').getContext('2d'), {{
    type: 'bar',
    data: {{
        labels: groups.map(g => g.eval_group),
        datasets: ['vector','hybrid'].map((m, i) => ({{
            label: labels[m], data: groups.map(g => g['avg_imp_' + m]),
            backgroundColor: bgPalette[i+1], borderColor: palette[i+1], borderWidth: 1.5,
        }}))
    }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ position: 'bottom' }} }},
        scales: {{ y: {{ ticks: {{ callback: v => v.toFixed(3) }} }} }}
    }}
}});

let th = '<tr><th>#</th><th>问题</th><th>分组</th><th>纯LLM</th><th>向量RAG</th><th>混合RAG</th><th>向量提升</th><th>混合提升</th></tr>';
const details = {json.dumps(details.to_dict(orient='records'), ensure_ascii=False)};
details.forEach((row, idx) => {{
    th += '<tr>';
    th += `<td>${{idx+1}}</td>`;
    th += `<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${{(row.question||'').replace(/"/g,'&quot;')}}">${{(row.question||'').substring(0,30)}}</td>`;
    th += `<td>${{row.eval_group||''}}</td>`;
    methods.forEach(m => th += `<td>${{fmt(row['sim_' + m])}}</td>`);
    ['vector','hybrid'].forEach(m => {{
        const v = row['improvement_' + m];
        const cls = v != null && v > 0 ? 'pos' : (v != null && v < 0 ? 'neg' : '');
        th += `<td class="${{cls}}">${{v != null ? (v>=0?'+':'')+v.toFixed(4) : 'N/A'}}</td>`;
    }});
    th += '</tr>';
}});
document.getElementById('detailTable').innerHTML = th + '</tbody>';
</script>
</body>
</html>"""

Path("eval_v2.html").write_text(html, encoding="utf-8")

# Print summary
print("\n" + "=" * 55)
print("评估完成！")
print("=" * 55)
for _, row in summary.iterrows():
    print(f"\n[{row['eval_group']}] ({int(row['count'])}题)")
    for m in methods:
        print(f"  {labels[m]:8s} sim={row[f'avg_sim_{m}']:.4f}  pass={row[f'pass_rate_{m}']:.1%}  fact={row[f'avg_fact_{m}'] or 'N/A'}")

print(f"\n详细结果: eval_v2.csv")
print(f"汇总结果: eval_v2_summary.csv")
print(f"可视化报告: eval_v2.html")
