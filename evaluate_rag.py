from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import ollama
import pandas as pd

TEST_SET_PATH = "data/12306_rag_eval_dataset.csv"
OLLAMA_MODEL = "qwen2.5:0.5b"
RETRIEVE_TOP_K = 3
SEMANTIC_PASS_THRESHOLD = 0.75
ANSWER_TRUNCATE = 200
OLLAMA_OPTIONS = {"temperature": 0, "top_p": 1, "seed": 42}

METHODS = ["no_rag", "vector", "hybrid"]
METHOD_LABELS = {
    "no_rag": "纯 LLM（无 RAG）",
    "vector": "向量检索 RAG",
    "hybrid": "混合检索 RAG",
}


def load_test_set(path: str = TEST_SET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    required_columns = {"question", "answer"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"测试集缺少必要列: {missing_columns}")

    keep_columns = ["question", "answer"]
    for col in ["id", "category", "difficulty", "test_type", "should_use_rag", "source_question"]:
        if col in df.columns:
            keep_columns.append(col)

    df = df[keep_columns].dropna(subset=["question", "answer"]).reset_index(drop=True)

    if "test_type" in df.columns:
        df["eval_group"] = df["test_type"].map(lambda x: _normalize_test_type(str(x)))
    else:
        df["eval_group"] = "all"
    print(f"测试集加载成功，共 {len(df)} 个问题")
    print("题型分布:")
    print(df["eval_group"].value_counts().to_string())
    return df


def _normalize_test_type(test_type: str) -> str:
    mapping = {
        "detail": "rule_detail", "boundary": "rule_detail",
        "composition": "rule_detail", "paraphrase": "simple",
        "out_of_scope": "out_of_scope",
    }
    return mapping.get(test_type, "simple")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a / a_norm, b / b_norm))


def extract_key_facts(text: str) -> list[str]:
    fact_patterns = [
        r"\d+(?:\.\d+)?\s*(?:年|日|天|小时|分钟|次|元|%|折|公斤|千克|厘米|公里)",
        r"[一二三四五六七八九十百千万]+(?:年|日|天|小时|分钟|次|元|折|公斤|千克|厘米|公里)",
        r"(?:免费|收费|不收费|不得|不能|可以|应当|必须|有效|无效|退还|补收|删除)",
    ]
    facts: list[str] = []
    for pattern in fact_patterns:
        facts.extend(re.findall(pattern, text))
    seen = set()
    unique_facts = []
    for fact in facts:
        normalized = re.sub(r"\s+", "", fact)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_facts.append(normalized)
    return unique_facts


def key_fact_recall(ground_truth: str, prediction: str) -> float | None:
    facts = extract_key_facts(ground_truth)
    if not facts:
        return None
    hits = sum(1 for fact in facts if fact in prediction.replace(" ", ""))
    return hits / len(facts)


def build_prompt(question: str, retrieved: list[dict], use_rag: bool) -> str:
    if not use_rag:
        return f"""你是铁路出行旅客常见问题问答助手。请根据你自己的知识回答问题。
如果问题涉及实时信息、资料外信息或你不确定，请明确说明无法确认，不要编造。

用户问题：
{question}

回答："""

    if retrieved:
        context_parts = []
        for idx, item in enumerate(retrieved, 1):
            chunk = item['answer_chunk']
            if len(chunk) > ANSWER_TRUNCATE:
                chunk = chunk[:ANSWER_TRUNCATE] + "..."
            context_parts.append(
                f"【参考{idx}】相似度：{item['score']:.4f}\n"
                f"原问题：{item['question']}\n"
                f"答案片段：{chunk}"
            )
        context = "\n\n".join(context_parts)
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


def call_model(prompt: str) -> str:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options=OLLAMA_OPTIONS,
    )
    return response["message"]["content"]


def evaluate_one(question: str, ground_truth: str, retriever, model_enc,
                 use_rag: bool, method: str = "vector") -> dict:
    if not use_rag:
        retrieved = []
    elif method == "hybrid":
        retrieved = retriever.retrieve_hybrid(question, k=RETRIEVE_TOP_K)
    else:
        retrieved = retriever.retrieve(question, k=RETRIEVE_TOP_K)

    prompt = build_prompt(question, retrieved, use_rag)

    try:
        prediction = call_model(prompt)
        error = ""
    except Exception as exc:
        prediction = f"[ERROR] {exc}"
        error = str(exc)

    gt_emb = model_enc.encode([ground_truth])[0]
    pred_emb = model_enc.encode([prediction])[0]
    similarity = cosine_similarity(gt_emb, pred_emb)
    fact_recall = key_fact_recall(ground_truth, prediction)

    return {
        "prediction": prediction,
        "similarity": similarity,
        "semantic_pass": int(similarity >= SEMANTIC_PASS_THRESHOLD),
        "key_fact_recall": fact_recall,
        "top_retrieval_score": retrieved[0]["score"] if retrieved else None,
        "top_retrieval_question": retrieved[0]["question"] if retrieved else "",
        "error": error,
    }


def evaluate_all(df: pd.DataFrame, retriever, model_enc,
                 limit: int | None = None,
                 details_out: str = "evaluation_results.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_df = df.head(limit) if limit else df

    # 断点续跑：加载已有结果，跳过已完成的问题
    completed_questions = set()
    existing_rows = []
    if Path(details_out).exists():
        print(f"检测到已有结果文件 {details_out}，加载断点...")
        old_df = pd.read_csv(details_out)
        completed_questions = set(old_df["question"].tolist())
        existing_rows = old_df.to_dict(orient="records")
        print(f"已加载 {len(completed_questions)} 条已完成记录，将跳过")

    rows = list(existing_rows)

    total = len(eval_df)
    pending = 0
    for i, row in eval_df.iterrows():
        question = row["question"]
        ground_truth = row["answer"]
        eval_group = row["eval_group"]

        if question in completed_questions:
            continue

        pending += 1

    if pending == 0:
        print("所有问题已完成评估！")
        details_df = pd.DataFrame(rows)
        summary = build_summary(details_df)
        return details_df, summary

    done = len(completed_questions)
    current = 0
    for i, row in eval_df.iterrows():
        question = row["question"]
        ground_truth = row["answer"]
        eval_group = row["eval_group"]

        if question in completed_questions:
            continue

        current += 1
        print(f"\n[{done + current}/{total}] [{eval_group}] {question[:55]}...")

        results = {}
        for method in METHODS:
            if method == "no_rag":
                r = evaluate_one(question, ground_truth, retriever, model_enc,
                                 use_rag=False, method="vector")
            elif method == "hybrid":
                r = evaluate_one(question, ground_truth, retriever, model_enc,
                                 use_rag=True, method="hybrid")
            else:  # vector
                r = evaluate_one(question, ground_truth, retriever, model_enc,
                                 use_rag=True, method="vector")

            results[method] = r
            print(f"  {METHOD_LABELS[method]:16s} sim={r['similarity']:.4f}  "
                  f"fact={r.get('key_fact_recall') or 'N/A'}")

        base_row = {
            "question": question, "ground_truth": ground_truth,
            "eval_group": eval_group,
            "source_category": row.get("category", ""),
            "difficulty": row.get("difficulty", ""),
            "test_type": row.get("test_type", ""),
            "should_use_rag": row.get("should_use_rag", ""),
        }
        for method in METHODS:
            r = results[method]
            base_row[f"answer_{method}"] = r["prediction"]
            base_row[f"sim_{method}"] = r["similarity"]
            base_row[f"pass_{method}"] = r["semantic_pass"]
            base_row[f"fact_{method}"] = r["key_fact_recall"]
            base_row[f"top_score_{method}"] = r["top_retrieval_score"]
            base_row[f"top_q_{method}"] = r["top_retrieval_question"]
            base_row[f"error_{method}"] = r["error"]

        for method in ["vector", "hybrid"]:
            base_row[f"improvement_{method}"] = (
                results[method]["similarity"] - results["no_rag"]["similarity"]
            )

        rows.append(base_row)

        # 增量保存：每完成一题立即写入 CSV
        pd.DataFrame(rows).to_csv(details_out, index=False, encoding="utf-8-sig")
        time.sleep(0.4)

    details_df = pd.DataFrame(rows)
    summary = build_summary(details_df)
    pd.DataFrame(rows).to_csv(details_out, index=False, encoding="utf-8-sig")
    return details_df, summary


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, group_df in results.groupby("eval_group", dropna=False):
        row = {"eval_group": group_name, "count": len(group_df)}
        for method in METHODS:
            row[f"avg_sim_{method}"] = group_df[f"sim_{method}"].mean()
            row[f"pass_rate_{method}"] = group_df[f"pass_{method}"].mean()
            fact_vals = group_df[f"fact_{method}"].dropna()
            row[f"avg_fact_{method}"] = fact_vals.mean() if len(fact_vals) > 0 else None
        for method in ["vector", "hybrid"]:
            row[f"avg_imp_{method}"] = group_df[f"improvement_{method}"].mean()
        rows.append(row)

    total = {"eval_group": "all", "count": len(results)}
    for method in METHODS:
        total[f"avg_sim_{method}"] = results[f"sim_{method}"].mean()
        total[f"pass_rate_{method}"] = results[f"pass_{method}"].mean()
        fact_vals = results[f"fact_{method}"].dropna()
        total[f"avg_fact_{method}"] = fact_vals.mean() if len(fact_vals) > 0 else None
    for method in ["vector", "hybrid"]:
        total[f"avg_imp_{method}"] = results[f"improvement_{method}"].mean()
    rows.append(total)

    return pd.DataFrame(rows)


def generate_report_html(details: pd.DataFrame, summary: pd.DataFrame, output_path: str):
    summary_json = json.dumps(summary.to_dict(orient="records"), ensure_ascii=False)
    methods_json = json.dumps(METHODS, ensure_ascii=False)
    labels_json = json.dumps(METHOD_LABELS, ensure_ascii=False)

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
h1 {{ font-size: 22px; font-weight: 500; margin-bottom: 8px; }}
h2 {{ font-size: 16px; font-weight: 500; margin: 28px 0 12px; }}
.subtitle {{ font-size: 13px; color: #888; margin-bottom: 24px; }}
.card {{ background: #fff; border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-bottom: 16px; }}
.metric {{ background: #f7f8fa; border-radius: 8px; padding: 14px 16px; }}
.metric-label {{ font-size: 12px; color: #888; }}
.metric-value {{ font-size: 24px; font-weight: 500; margin-top: 4px; }}
.metric-value.good {{ color: #1d9e75; }}
.metric-value.better {{ color: #378add; }}
.chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
@media (max-width: 700px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
.chart-wrap {{ position: relative; height: 310px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 0.5px solid #e5e5e5; }}
th {{ font-weight: 500; color: #888; font-size: 12px; }}
tr.best {{ background: #e1f5ee; }}
</style>
</head>
<body>
<h1>RAG 多模式对比评估报告</h1>
<p class="subtitle">测试集：75 题（规则细节 / 简单常识 / 资料外） &nbsp;|&nbsp; 模型：{OLLAMA_MODEL}</p>

<div class="summary-grid" id="metricsGrid"></div>

<div class="card">
  <h2>按题型分组 — 平均语义相似度</h2>
  <div class="chart-wrap"><canvas id="barSim" role="img" aria-label="分组相似度对比"></canvas></div>
</div>

<div class="card">
  <h2>按题型分组 — 关键事实命中率</h2>
  <div class="chart-wrap"><canvas id="barFact" role="img" aria-label="分组事实命中率对比"></canvas></div>
</div>

<div class="card">
  <h2>各方法相对于纯LLM的语义相似度提升</h2>
  <div class="chart-wrap"><canvas id="barImp" role="img" aria-label="相似度提升对比"></canvas></div>
</div>

<div class="card">
  <h2>评估结果明细</h2>
  <div style="overflow-x:auto;" id="detailTable"></div>
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

let gridHtml = '';
methods.forEach((m, i) => {{
    const sim = allRow ? allRow['avg_sim_' + m] : null;
    const pr = allRow ? allRow['pass_rate_' + m] : null;
    const fv = allRow ? allRow['avg_fact_' + m] : null;
    gridHtml += `<div class="metric">
      <div class="metric-label">${{labels[m]}}</div>
      <div class="metric-value ${{i > 0 ? 'better' : ''}}">${{fmt(sim)}}</div>
      <div style="font-size:11px;color:#888;margin-top:4px;">通过率 ${{pct(pr)}} &nbsp;|&nbsp; 事实命中 ${{fmt(fv)}}</div>
    </div>`;
}});
document.getElementById('metricsGrid').innerHTML = gridHtml;

function makeChart(id, labelKey, yLabel, isPct) {{
    const ctx = document.getElementById(id).getContext('2d');
    const datasets = methods.map((m, i) => ({{
        label: labels[m],
        data: groups.map(g => g[labelKey + m]),
        backgroundColor: bgPalette[i],
        borderColor: palette[i],
        borderWidth: 1.5,
    }}));
    new Chart(ctx, {{
        type: 'bar',
        data: {{ labels: groups.map(g => g.eval_group), datasets: datasets }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{ legend: {{ position: 'bottom' }} }},
            scales: {{
                y: {{
                    beginAtZero: true, max: isPct ? 1 : undefined,
                    ticks: {{ callback: v => isPct ? (v*100).toFixed(0) + '%' : v.toFixed(3) }}
                }}
            }}
        }}
    }});
}}

makeChart('barSim', 'avg_sim_', '语义相似度', false);
makeChart('barFact', 'avg_fact_', '关键事实命中率', true);

const impCtx = document.getElementById('barImp').getContext('2d');
const impMethods = ['vector', 'hybrid'];
const impDatasets = impMethods.map((m, i) => ({{
    label: labels[m],
    data: groups.map(g => g['avg_imp_' + m]),
    backgroundColor: bgPalette[i+1],
    borderColor: palette[i+1],
    borderWidth: 1.5,
}}));
new Chart(impCtx, {{
    type: 'bar',
    data: {{ labels: groups.map(g => g.eval_group), datasets: impDatasets }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ position: 'bottom' }} }},
        scales: {{ y: {{ ticks: {{ callback: v => v.toFixed(3) }} }} }}
    }}
}});

let tableHtml = '<table><thead><tr>';
const headers = ['问题', '分组', '纯LLM', '向量RAG', '混合RAG', '向量提升', '混合提升'];
headers.forEach(h => tableHtml += `<th>${{h}}</th>`);
tableHtml += '</tr></thead><tbody>';

const details = {json.dumps(details.to_dict(orient='records'), ensure_ascii=False)};
details.forEach(row => {{
    tableHtml += '<tr>';
    tableHtml += `<td style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${{(row.question||'').replace(/"/g,'&quot;')}}">${{(row.question||'').substring(0,35)}}</td>`;
    tableHtml += `<td>${{row.eval_group||''}}</td>`;
    methods.forEach(m => {{
        const v = row['sim_' + m];
        tableHtml += `<td>${{v != null ? v.toFixed(4) : 'N/A'}}</td>`;
    }});
    ['vector','hybrid'].forEach(m => {{
        const v = row['improvement_' + m];
        const cls = v != null && v > 0 ? ' style="color:#1d9e75"' : (v != null && v < 0 ? ' style="color:#e24b4a"' : '');
        tableHtml += `<td${{cls}}>${{v != null ? v.toFixed(4) : 'N/A'}}</td>`;
    }});
    tableHtml += '</tr>';
}});
tableHtml += '</tbody></table>';
document.getElementById('detailTable').innerHTML = tableHtml;
</script>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"可视化报告已保存到: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 多模式对比评估")
    parser.add_argument("--test-set", default=TEST_SET_PATH, help="测试集 CSV 路径")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条")
    parser.add_argument("--details-out", default="evaluation_results.csv")
    parser.add_argument("--summary-out", default="evaluation_summary.csv")
    parser.add_argument("--report-out", default="evaluation_report.html")
    return parser.parse_args()


def main() -> None:
    from retriever import HybridRetriever

    args = parse_args()

    print("初始化 HybridRetriever（向量 + BM25）...")
    retriever = HybridRetriever(enable_bm25=True)
    print("加载测试集...")
    test_df = load_test_set(args.test_set)

    model_enc = retriever.model

    print(f"\n开始多模式评估，共 {len(test_df)} 题（{len(METHODS)} 种模式）...")
    print(f"模式: {', '.join(METHOD_LABELS[m] for m in METHODS)}")

    details, summary = evaluate_all(test_df, retriever, model_enc,
                                    limit=args.limit, details_out=args.details_out)

    details.to_csv(args.details_out, index=False, encoding="utf-8-sig")
    summary.to_csv(args.summary_out, index=False, encoding="utf-8-sig")

    generate_report_html(details, summary, args.report_out)

    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    for _, row in summary.iterrows():
        print(f"\n[{row['eval_group']}] ({row['count']} 题)")
        for m in METHODS:
            sim = row[f"avg_sim_{m}"]
            pr = row[f"pass_rate_{m}"]
            fact = row[f"avg_fact_{m}"]
            print(f"  {METHOD_LABELS[m]:20s} sim={sim:.4f}  pass={pr:.1%}  fact={fact or 'N/A'}")

    print(f"\n详细结果: {args.details_out}")
    print(f"汇总结果: {args.summary_out}")
    print(f"可视化报告: {args.report_out}")


if __name__ == "__main__":
    main()
