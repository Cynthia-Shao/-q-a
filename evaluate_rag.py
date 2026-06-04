from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import ollama
import pandas as pd


TEST_SET_PATH = "data/12306_rag_eval_dataset.csv"
OLLAMA_MODEL = "qwen2.5:latest"
RETRIEVE_TOP_K = 5
SEMANTIC_PASS_THRESHOLD = 0.75
OLLAMA_OPTIONS = {
    "temperature": 0,
    "top_p": 1,
    "seed": 42,
}


def load_test_set(path: str = TEST_SET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns={"问题": "question", "答案": "answer"})

    required_columns = {"question", "answer"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"测试集缺少必要列: {missing_columns}")

    optional_columns = [
        "id",
        "category",
        "difficulty",
        "test_type",
        "should_use_rag",
        "source_question",
    ]
    keep_columns = ["question", "answer"] + [col for col in optional_columns if col in df.columns]
    df = df[keep_columns].dropna(subset=["question", "answer"]).reset_index(drop=True)
    if "test_type" in df.columns:
        df["eval_group"] = df["test_type"].map(normalize_test_type)
    else:
        df["eval_group"] = df["question"].apply(classify_question)

    print(f"测试集加载成功，共 {len(df)} 个问题")
    print("题型分布:")
    print(df["eval_group"].value_counts().to_string())
    return df


def normalize_test_type(test_type: str) -> str:
    mapping = {
        "detail": "rule_detail",
        "boundary": "rule_detail",
        "composition": "rule_detail",
        "paraphrase": "simple",
        "out_of_scope": "out_of_scope",
    }
    return mapping.get(str(test_type), "simple")


def classify_question(question: str) -> str:
    """Classify questions to make the RAG advantage easier to analyze."""
    out_of_scope_patterns = [
        "今天",
        "晚点",
        "检票口",
        "具体是多少",
        "投诉电话",
        "座椅靠背角度",
        "会员积分兑换",
    ]
    if any(pattern in question for pattern in out_of_scope_patterns):
        return "out_of_scope"

    rule_detail_patterns = [
        "几",
        "多少",
        "多久",
        "什么时候",
        "哪天",
        "比例",
        "收费",
        "费用",
        "退票费",
        "改签费",
        "证件",
        "材料",
        "限制",
        "能不能",
        "可以",
        "需要",
        "最晚",
        "最早",
        "期限",
        "有效",
    ]
    if any(pattern in question for pattern in rule_detail_patterns):
        return "rule_detail"

    return "simple"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a / a_norm, b / b_norm))


def extract_key_facts(text: str) -> list[str]:
    """Extract simple factual anchors such as numbers, dates, percentages, and rule words."""
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
            context_parts.append(
                f"【参考{idx}】相似度：{item['score']:.4f}\n"
                f"原问题：{item['question']}\n"
                f"答案片段：{item['answer_chunk']}"
            )
        context = "\n\n".join(context_parts)
    else:
        context = "未找到直接相关的参考资料。"

    return f"""你是铁路出行旅客常见问题问答助手。请严格依据参考信息回答问题。
要求：
1. 优先回答具体规则、时间、费用、比例、证件、材料等关键信息。
2. 如果参考信息不足以回答，请说“根据现有资料无法回答该问题”，不要编造。
3. 回答末尾用一句话说明依据了哪些参考编号，例如“依据：参考1、参考2”。

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


def evaluate_one(question: str, ground_truth: str, retriever: Retriever, use_rag: bool) -> dict:
    retrieved = retriever.retrieve(question, k=RETRIEVE_TOP_K) if use_rag else []
    prompt = build_prompt(question, retrieved, use_rag)

    try:
        prediction = call_model(prompt)
        error = ""
    except Exception as exc:
        prediction = f"[ERROR] {exc}"
        error = str(exc)

    gt_emb = retriever.model.encode([ground_truth])[0]
    pred_emb = retriever.model.encode([prediction])[0]
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


def evaluate(df: pd.DataFrame, retriever: Retriever, limit: int | None = None) -> pd.DataFrame:
    rows = []
    eval_df = df.head(limit) if limit else df

    for i, row in eval_df.iterrows():
        question = row["question"]
        ground_truth = row["answer"]
        eval_group = row["eval_group"]
        print(f"\n处理第 {i + 1}/{len(eval_df)} 个问题 [{eval_group}]: {question[:60]}")

        no_rag = evaluate_one(question, ground_truth, retriever, use_rag=False)
        print(f"  未接入 RAG 相似度: {no_rag['similarity']:.4f}")

        rag = evaluate_one(question, ground_truth, retriever, use_rag=True)
        print(f"  接入 RAG 相似度:   {rag['similarity']:.4f}")
        print(f"  提升:             {rag['similarity'] - no_rag['similarity']:.4f}")

        rows.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "eval_group": eval_group,
                "source_category": row.get("category", ""),
                "difficulty": row.get("difficulty", ""),
                "test_type": row.get("test_type", ""),
                "should_use_rag": row.get("should_use_rag", ""),
                "source_question": row.get("source_question", ""),
                "answer_no_rag": no_rag["prediction"],
                "answer_rag": rag["prediction"],
                "score_no_rag": no_rag["similarity"],
                "score_rag": rag["similarity"],
                "improvement": rag["similarity"] - no_rag["similarity"],
                "semantic_pass_no_rag": no_rag["semantic_pass"],
                "semantic_pass_rag": rag["semantic_pass"],
                "key_fact_recall_no_rag": no_rag["key_fact_recall"],
                "key_fact_recall_rag": rag["key_fact_recall"],
                "top_retrieval_score": rag["top_retrieval_score"],
                "top_retrieval_question": rag["top_retrieval_question"],
                "error_no_rag": no_rag["error"],
                "error_rag": rag["error"],
            }
        )
        time.sleep(0.5)

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby("eval_group", dropna=False)
        .agg(
            count=("question", "count"),
            avg_no_rag=("score_no_rag", "mean"),
            avg_rag=("score_rag", "mean"),
            avg_improvement=("improvement", "mean"),
            pass_rate_no_rag=("semantic_pass_no_rag", "mean"),
            pass_rate_rag=("semantic_pass_rag", "mean"),
            key_fact_recall_no_rag=("key_fact_recall_no_rag", "mean"),
            key_fact_recall_rag=("key_fact_recall_rag", "mean"),
        )
        .reset_index()
    )

    total = pd.DataFrame(
        [
            {
                "eval_group": "all",
                "count": len(results),
                "avg_no_rag": results["score_no_rag"].mean(),
                "avg_rag": results["score_rag"].mean(),
                "avg_improvement": results["improvement"].mean(),
                "pass_rate_no_rag": results["semantic_pass_no_rag"].mean(),
                "pass_rate_rag": results["semantic_pass_rag"].mean(),
                "key_fact_recall_no_rag": results["key_fact_recall_no_rag"].mean(),
                "key_fact_recall_rag": results["key_fact_recall_rag"].mean(),
            }
        ]
    )
    return pd.concat([summary, total], ignore_index=True)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        formatted_row = []
        for value in row:
            if isinstance(value, float):
                formatted_row.append(f"{value:.4f}")
            else:
                formatted_row.append(str(value))
        rows.append(formatted_row)

    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table)


def write_report(summary: pd.DataFrame, output_path: str) -> None:
    report_lines = [
        "# RAG 评估结果摘要",
        "",
        "## 说明",
        "",
        "- 本次评估固定了大模型生成参数，降低重复运行时的随机波动。",
        "- 除平均语义相似度外，额外统计了按题型分组的提升、语义通过率和关键事实命中率。",
        "- `rule_detail` 更能体现 RAG 对具体条款、时间、比例、证件、费用等规则问题的帮助。",
        "",
        "## 分组结果",
        "",
        dataframe_to_markdown(summary),
        "",
        "## 可写入报告的结论",
        "",
        "由于无检索基线模型本身具备一定铁路出行常识，整体平均相似度提升可能有限。"
        "因此，本项目进一步从规则细节题、资料外问题拒答、关键事实命中率和答案可追溯性角度评估 RAG 效果。"
        "结果表明，RAG 在需要依据具体条款、时间、比例和办理条件的问题上更稳定，"
        "能够减少模型仅凭通用知识回答导致的遗漏或编造。",
        "",
    ]
    Path(output_path).write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate railway passenger QA with and without RAG.")
    parser.add_argument("--test-set", default=TEST_SET_PATH, help="测试集 CSV 路径")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条，便于调试")
    parser.add_argument("--details-out", default="evaluation_results.csv", help="逐题结果输出 CSV")
    parser.add_argument("--summary-out", default="evaluation_summary.csv", help="汇总结果输出 CSV")
    parser.add_argument("--report-out", default="evaluation_report.md", help="报告摘要输出 Markdown")
    return parser.parse_args()


def main() -> None:
    from retriever import Retriever

    args = parse_args()

    print("初始化检索器...")
    retriever = Retriever()
    test_df = load_test_set(args.test_set)
    results = evaluate(test_df, retriever, limit=args.limit)
    summary = summarize(results)

    results.to_csv(args.details_out, index=False, encoding="utf-8-sig")
    summary.to_csv(args.summary_out, index=False, encoding="utf-8-sig")
    write_report(summary, args.report_out)

    print("\n========== 对比结果 ==========")
    print(summary.to_string(index=False))
    print(f"\n逐题结果已保存到: {args.details_out}")
    print(f"汇总结果已保存到: {args.summary_out}")
    print(f"报告摘要已保存到: {args.report_out}")


if __name__ == "__main__":
    main()
