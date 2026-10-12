import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import time
import random
from collections import Counter
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from dotenv import load_dotenv

from retriever import bfs_retrieve, driver, get_embedding, find_similar_any_label
from generator import (
    GENERATOR_MODEL, SYSTEM_PROMPT, call_llm_for_verdict,
    generate_reports_from_file, check_mandatory_coverage,
)
from mot import DEFAULT_THETA_SIM

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_MODEL = os.getenv("JUDGE_MODEL", GENERATOR_MODEL)

VIOLATION_VERDICTS = {"違反", "缺漏", "不符合"}

def classification_metrics(reports: list, ground_truth: dict) -> dict:

    gt_violation_ids = {v["contract_node_id"] for v in ground_truth["violations"]}
    by_id = {r["clause_id"]: r for r in reports}

    tp, fn, fp, tn = [], [], [], []
    for cid, r in by_id.items():
        is_gt_violation = cid in gt_violation_ids
        predicted_violation = r.get("verdict") in VIOLATION_VERDICTS
        if is_gt_violation and predicted_violation:
            tp.append(cid)
        elif is_gt_violation and not predicted_violation:
            fn.append(cid)
        elif not is_gt_violation and predicted_violation:
            fp.append(cid)
        else:
            tn.append(cid)

    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

def _law_ref_article_no(law_ref: str):
    m = re.search(r"第([\d一二三四五六七八九十百\-]+)條", law_ref)
    return m.group(1) if m else None

def law_grounding_accuracy(reports: list, ground_truth: dict) -> dict:

    gt_by_id = {v["contract_node_id"]: v for v in ground_truth["violations"]}
    by_id = {r["clause_id"]: r for r in reports}

    hits, misses = [], []
    for cid, gt in gt_by_id.items():
        r = by_id.get(cid)
        if not r or r.get("verdict") not in VIOLATION_VERDICTS:
            continue

        generated_basis_text = " ".join(r.get("law_basis") or [])
        found_any = False
        for law_ref in gt.get("law_ref", []):
            art_no = _law_ref_article_no(law_ref)
            if art_no and art_no in generated_basis_text:
                found_any = True
                break
        (hits if found_any else misses).append(cid)

    total = len(hits) + len(misses)
    accuracy = len(hits) / total if total else 0.0
    return {"hits": hits, "misses": misses, "accuracy": round(accuracy, 4)}

def mandatory_grounding_accuracy(reports: list, ground_truth: dict) -> dict:

    gt_by_id = {v["contract_node_id"]: v for v in ground_truth["violations"]}
    by_id = {r["clause_id"]: r for r in reports}

    hits, misses = [], []
    for cid, gt in gt_by_id.items():
        r = by_id.get(cid)
        if not r or r.get("verdict") not in VIOLATION_VERDICTS:
            continue

        generated_basis_text = " ".join(r.get("mandatory_item_basis") or [])
        found_any = any(
            ref in generated_basis_text for ref in gt.get("mandatory_items_ref", [])
        )
        (hits if found_any else misses).append(cid)

    total = len(hits) + len(misses)
    accuracy = len(hits) / total if total else 0.0
    return {"hits": hits, "misses": misses, "accuracy": round(accuracy, 4)}

def risk_hit_rate(classification: dict) -> float:

    return classification["recall"]

def reduction_rate(reports: list) -> dict:

    total = len(reports)
    needs_review = sum(1 for r in reports if r.get("verdict") != "符合")
    rate = 1 - (needs_review / total) if total else 0.0
    return {
        "total_clauses": total,
        "needs_review_count": needs_review,
        "reduction_rate": round(rate, 4),
    }

def decision_source_summary(reports: list) -> dict:

    counts = Counter(r.get("decision_source", "llm") for r in reports)
    total = len(reports)
    return {
        "counts": dict(counts),
        "total": total,
        "rule_based_ratio": round(counts.get("rule_based", 0) / total, 4) if total else 0.0,
    }

def grid_search(chunks_path: str = "data/processed/wrong_chunks.json",
                 ground_truth_path: str = "data/raw/ground_truth.json",
                 k_values: list = (1, 2),
                 t_values: list = (3, 5, 7),
                 theta_sim: float = DEFAULT_THETA_SIM,
                 theta_merge: float = None,
                 alpha: float = None,
                 model: str = GENERATOR_MODEL,
                 use_rule_engine: bool = True,
                 save_reports: bool = True,
                 min_k_for_selection: int = 2) -> dict:

    with open(ground_truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    from mot import DEFAULT_THETA_MERGE, DEFAULT_ALPHA
    theta_merge = DEFAULT_THETA_MERGE if theta_merge is None else theta_merge
    alpha = DEFAULT_ALPHA if alpha is None else alpha

    if save_reports:
        os.makedirs("data/reports/grid_search", exist_ok=True)

    grid_results = []
    for k in k_values:
        for t in t_values:
            print(f"\n=== Grid Search: k={k}, t={t} ===")
            reports = generate_reports_from_file(
                chunks_path, theta_sim=theta_sim, theta_merge=theta_merge,
                alpha=alpha, t=t, k=k, model=model, use_rule_engine=use_rule_engine
            )
            if save_reports:
                out_path = f"data/reports/grid_search/reports_k{k}_t{t}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(reports, f, ensure_ascii=False, indent=2)

            cls = classification_metrics(reports, ground_truth)
            rr = reduction_rate(reports)
            law_ga = law_grounding_accuracy(reports, ground_truth)
            mand_ga = mandatory_grounding_accuracy(reports, ground_truth)
            grid_results.append({
                "k": k, "t": t,
                "precision": cls["precision"],
                "recall": cls["recall"],
                "risk_hit_rate": risk_hit_rate(cls),
                "reduction_rate": rr["reduction_rate"],
                "f1": cls["f1"],
                "law_grounding_accuracy": law_ga["accuracy"],
                "mandatory_grounding_accuracy": mand_ga["accuracy"],
                "needs_review_count": rr["needs_review_count"],
            })
            print(f"  Precision={cls['precision']:.2%}  Recall={cls['recall']:.2%}  "
                  f"Reduction Rate={rr['reduction_rate']:.2%}  "
                  f"Law GA={law_ga['accuracy']:.2%}  Mandatory GA={mand_ga['accuracy']:.2%}")

    eligible = [r for r in grid_results if r["k"] >= min_k_for_selection]
    excluded = [r for r in grid_results if r["k"] < min_k_for_selection]
    if not eligible:
        print(f"\n⚠️ 沒有任何組合的 k >= {min_k_for_selection}，退回用全部組合選最佳"
              f"（請人工檢查這個結果合不合理）。")
        eligible = grid_results
    best = max(eligible, key=lambda r: r["f1"])

    if excluded:
        print(f"\n以下組合因為 k < {min_k_for_selection}（Hybrid RAG 會退化成跟單一向量檢索"
              f"相同，法條依據不可信）被排除在「最佳代表設定」候選之外，僅供對照參考：")
        for r in excluded:
            print(f"  k={r['k']}, t={r['t']}：F1={r['f1']:.2%}，"
                  f"但 Law GA 只有 {r['law_grounding_accuracy']:.2%}")

    return {"grid_results": grid_results, "best": best, "excluded_from_selection": excluded}

RAGAS_JUDGE_SYSTEM_PROMPT = """你是一個獨立、嚴格的評分者，負責評估一個法律契約審查系統單次回答的品質。
你會收到三樣東西：
1. 查詢（一段契約條款原文，系統要判斷它合不合規）
2. 檢索到的內容（系統用來輔助判斷的參考資料）
3. 系統的回答（系統對這條契約條款的審查結論）

請針對以下三個指標各給 0.0～1.0 的分數（可以有小數）：

- context_relevance：檢索到的內容裡，有多少比例是真的跟「判斷這條
  契約條款合不合規」相關的？如果檢索到的內容大部分是無關的雜訊，
  分數應該低；如果檢索到的內容精準對應到查詢的主題，分數應該高。
  如果完全沒有檢索到任何內容，給 0.0。

- faithfulness：系統回答裡提到的每一個具體主張（引用的法條、應記載
  事項、違反原因的描述），有多少比例真的能在「檢索到的內容」裡找到
  依據？如果系統引用了檢索內容裡根本沒出現的法條、或編造了檢索內容
  沒有支持的說法，這些主張要算不忠實，會拉低分數。這個指標只看回答
  有沒有憑空捏造，不評估回答本身合不合理。

- answer_relevance：系統的回答是不是真的針對「這條契約條款合不合規」
  這個問題在作答？如果回答文不對題、太籠統、或答非所問，分數應該低。

只輸出一個 JSON 物件，不要有其他文字，格式如下：
{
  "context_relevance": 0.0,
  "faithfulness": 0.0,
  "answer_relevance": 0.0,
  "notes": "簡短說明評分理由"
}"""

def _format_answer(report: dict) -> str:
    return (
        f"判定：{report.get('verdict')}\n"
        f"法律依據：{report.get('law_basis')}\n"
        f"應記載/不得記載依據：{report.get('mandatory_item_basis')}\n"
        f"理由：{report.get('violation_reason')}\n"
        f"與範本差異：{report.get('diff_from_template')}"
    )

_THROTTLE_SECONDS = float(os.getenv("JUDGE_THROTTLE_SECONDS", "3"))
_MAX_RETRIES = 6
_BASE_BACKOFF = 5.0

def _extract_retry_after(err: Exception, default: float) -> float:

    m = re.search(r"try again in ([\d.]+)s", str(err))
    if m:
        try:
            return float(m.group(1)) + 0.5
        except ValueError:
            pass
    return default

def _call_with_retry(fn, *args, **kwargs):

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            wait = _extract_retry_after(e, default=_BASE_BACKOFF * (2 ** (attempt - 1)))
            wait += random.uniform(0, 1)
            print(f"  ⚠️ Rate limit（429），第 {attempt}/{_MAX_RETRIES} 次重試，"
                  f"等待 {wait:.1f} 秒...")
            time.sleep(wait)
        except (APIConnectionError, APITimeoutError, InternalServerError) as e:
            wait = _BASE_BACKOFF * (2 ** (attempt - 1))
            print(f"  ⚠️ API 暫時性錯誤（{type(e).__name__}），第 {attempt}/{_MAX_RETRIES} "
                  f"次重試，等待 {wait:.1f} 秒...")
            time.sleep(wait)
    raise RuntimeError(
        f"超過最大重試次數（{_MAX_RETRIES}），仍然失敗。請確認 OpenAI 帳號的 "
        f"TPM/RPM 額度設定（https://platform.openai.com/account/rate-limits），"
        f"或考慮調高 JUDGE_THROTTLE_SECONDS 環境變數、換一顆額度較高的裁判模型。"
    )

def ragas_judge(query_text: str, context_text: str, answer_text: str,
                 model: str = JUDGE_MODEL) -> dict:
    if _THROTTLE_SECONDS > 0:
        time.sleep(_THROTTLE_SECONDS)

    response = _call_with_retry(
        client.chat.completions.create,
        model=model,
        messages=[
            {"role": "system", "content": RAGAS_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"查詢：\n{query_text}\n\n"
                                        f"檢索到的內容：\n{context_text}\n\n"
                                        f"系統的回答：\n{answer_text}"},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content
    try:
        text = re.sub(r"^```(json)?|```$", "", raw.strip()).strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return json.loads(match.group(0) if match else text)
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        return {"context_relevance": None, "faithfulness": None, "answer_relevance": None,
                "parse_error": str(e), "raw_response": raw}

def format_baseline_context(template_matches: list) -> str:

    if not template_matches:
        return "（未檢索到任何相似的範本條文）"
    lines = []
    for m in template_matches:
        lines.append(f"範本第{m['article_no']}條原文（相似度{m['score']:.3f}）：{m['text']}")
    return "\n".join(lines)

def generate_baseline_report(clause_id: str, clause_text: str,
                              theta_sim: float = DEFAULT_THETA_SIM,
                              model: str = GENERATOR_MODEL) -> dict:

    bfs_result = bfs_retrieve(clause_text=clause_text, k=1, theta_sim=theta_sim)
    matches = bfs_result["template_matches"]

    if not matches:
        return {
            "clause_id": clause_id, "clause_text": clause_text,
            "verdict": "無法判斷", "matched_template_article_no": None,
            "law_basis": [], "mandatory_item_basis": [],
            "violation_reason": "單一向量檢索沒有找到任何相似的範本條文。",
            "suggested_revision": None, "diff_from_template": None,
            "context_text": "（未檢索到任何相似的範本條文）",
        }

    context_text = format_baseline_context(matches)
    verdict = call_llm_for_verdict(clause_text, context_text, model=model)
    return {"clause_id": clause_id, "clause_text": clause_text, **verdict, "context_text": context_text}

def format_flat_vector_context(matches: list) -> str:

    if not matches:
        return "（未檢索到任何相似的節點）"
    label_names = {"TemplateClause": "範本", "LawArticle": "法條", "MandatoryItem": "應記載/不得記載事項"}
    lines = []
    for m in matches:
        label = label_names.get(m["label"], m["label"])
        if m["label"] == "MandatoryItem":
            ref = f"{m.get('item_type', '')}{m.get('clause_no', '')}"
        elif m["label"] == "LawArticle":
            para = f"第{m['para_no']}項" if m.get("para_no") else ""
            ref = f"第{m.get('article_no')}條{para}"
        else:
            ref = f"第{m.get('article_no')}條"
        lines.append(f"[{label}] {ref}（相似度{m['score']:.3f}）：{m['text']}")
    return "\n".join(lines)

def generate_flat_vector_baseline_report(clause_id: str, clause_text: str,
                                          theta_sim: float = DEFAULT_THETA_SIM,
                                          top_n: int = 7,
                                          model: str = GENERATOR_MODEL) -> dict:

    embedding = get_embedding(clause_text)
    with driver.session() as session:
        matches = find_similar_any_label(session, embedding, theta_sim, top_n=top_n)

    if not matches:
        return {
            "clause_id": clause_id, "clause_text": clause_text,
            "verdict": "無法判斷", "matched_template_article_no": None,
            "law_basis": [], "mandatory_item_basis": [],
            "violation_reason": "單一向量檢索（全庫版）沒有找到任何相似節點。",
            "suggested_revision": None, "diff_from_template": None,
            "context_text": "（未檢索到任何相似的節點）",
        }

    context_text = format_flat_vector_context(matches)
    verdict = call_llm_for_verdict(clause_text, context_text, model=model)
    return {"clause_id": clause_id, "clause_text": clause_text, **verdict, "context_text": context_text}

def generate_flat_vector_baseline_from_file(chunks_path: str = "data/processed/wrong_chunks.json",
                                             theta_sim: float = DEFAULT_THETA_SIM,
                                             top_n: int = 7,
                                             model: str = GENERATOR_MODEL) -> list:
    with open(chunks_path, encoding="utf-8") as f:
        contract_chunks = json.load(f)
    reports = []
    for i, chunk in enumerate(contract_chunks, start=1):
        print(f"[{i}/{len(contract_chunks)}] 單一向量檢索（全庫版）：{chunk['id']}")
        reports.append(generate_flat_vector_baseline_report(
            chunk["id"], chunk["text"], theta_sim=theta_sim, top_n=top_n, model=model
        ))
    return reports

def generate_baseline_from_file(chunks_path: str = "data/processed/wrong_chunks.json",
                                 theta_sim: float = DEFAULT_THETA_SIM,
                                 model: str = GENERATOR_MODEL) -> list:
    with open(chunks_path, encoding="utf-8") as f:
        contract_chunks = json.load(f)
    reports = []
    for i, chunk in enumerate(contract_chunks, start=1):
        print(f"[{i}/{len(contract_chunks)}] 單一向量檢索（範本限定版）：{chunk['id']}")
        reports.append(generate_baseline_report(chunk["id"], chunk["text"], theta_sim=theta_sim, model=model))
    return reports

def _hybrid_context_text(clause_text: str, theta_sim: float, theta_merge: float,
                          alpha: float, t: int, k: int = 2) -> str:

    from mot import retrieve_and_merge
    from generator import _format_context
    mot_result = retrieve_and_merge(clause_text, theta_sim=theta_sim,
                                     theta_merge=theta_merge, alpha=alpha, t=t, k=k)
    return _format_context(mot_result)

def _avg(scores, key):
    vals = [s[key] for s in scores if isinstance(s.get(key), (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None

def _score_reports_with_judge(reports: list, context_lookup, judge_model: str, label: str) -> list:

    scores = []
    for i, r in enumerate(reports, start=1):
        print(f"  [{label} {i}/{len(reports)}] {r['clause_id']}")
        ctx = context_lookup(r)
        s = ragas_judge(r["clause_text"], ctx, _format_answer(r), model=judge_model)
        s["clause_id"] = r["clause_id"]
        scores.append(s)
    return scores

def run_evaluation(chunks_path: str = "data/processed/wrong_chunks.json",
                    ground_truth_path: str = "data/raw/ground_truth.json",
                    theta_sim: float = DEFAULT_THETA_SIM,
                    theta_merge: float = None,
                    alpha: float = None,
                    t: int = 7,
                    k: int = 2,
                    model: str = GENERATOR_MODEL,
                    judge_model: str = None) -> dict:

    from mot import DEFAULT_THETA_MERGE, DEFAULT_ALPHA
    theta_merge = DEFAULT_THETA_MERGE if theta_merge is None else theta_merge
    alpha = DEFAULT_ALPHA if alpha is None else alpha
    judge_model = judge_model or JUDGE_MODEL

    with open(ground_truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    print("=== 產生 Hybrid RAG 報告 ===")
    hybrid_reports = generate_reports_from_file(
        chunks_path, theta_sim=theta_sim, theta_merge=theta_merge, alpha=alpha, t=t, k=k, model=model
    )
    print("\n=== 產生單一向量檢索・範本限定版報告 ===")
    baseline_template_reports = generate_baseline_from_file(chunks_path, theta_sim=theta_sim, model=model)
    print("\n=== 產生單一向量檢索・全庫版報告 ===")
    baseline_flat_reports = generate_flat_vector_baseline_from_file(
        chunks_path, theta_sim=theta_sim, top_n=t, model=model
    )

    cls_hybrid = classification_metrics(hybrid_reports, ground_truth)
    cls_baseline_template = classification_metrics(baseline_template_reports, ground_truth)
    cls_baseline_flat = classification_metrics(baseline_flat_reports, ground_truth)

    law_ga_hybrid = law_grounding_accuracy(hybrid_reports, ground_truth)
    law_ga_baseline_flat = law_grounding_accuracy(baseline_flat_reports, ground_truth)
    mand_ga_hybrid = mandatory_grounding_accuracy(hybrid_reports, ground_truth)
    mand_ga_baseline_flat = mandatory_grounding_accuracy(baseline_flat_reports, ground_truth)

    print(f"\n=== 用 {judge_model} 評分 RAGAS-lite 三指標 ===")
    print("Hybrid RAG:")
    ragas_hybrid = _score_reports_with_judge(
        hybrid_reports,
        lambda r: _hybrid_context_text(r["clause_text"], theta_sim, theta_merge, alpha, t, k=k),
        judge_model, "hybrid"
    )
    print("Baseline（範本限定版）:")
    ragas_baseline_template = _score_reports_with_judge(
        baseline_template_reports, lambda r: r.get("context_text", ""), judge_model, "baseline_template_only"
    )
    print("Baseline（全庫版・公平對照）:")
    ragas_baseline_flat = _score_reports_with_judge(
        baseline_flat_reports, lambda r: r.get("context_text", ""), judge_model, "baseline_flat_vector_all"
    )

    def summarize(ragas_scores):
        return {
            "context_relevance": _avg(ragas_scores, "context_relevance"),
            "faithfulness": _avg(ragas_scores, "faithfulness"),
            "answer_relevance": _avg(ragas_scores, "answer_relevance"),
        }

    return {
        "params": {"theta_sim": theta_sim, "theta_merge": theta_merge, "alpha": alpha, "t": t, "k": k,
                    "generator_model": model, "judge_model": judge_model},
        "classification": {
            "hybrid": cls_hybrid,
            "baseline_template_only": cls_baseline_template,
            "baseline_flat_vector_all": cls_baseline_flat,
        },
        "law_grounding_accuracy": {"hybrid": law_ga_hybrid, "baseline_flat_vector_all": law_ga_baseline_flat},
        "mandatory_grounding_accuracy": {"hybrid": mand_ga_hybrid, "baseline_flat_vector_all": mand_ga_baseline_flat},
        "ragas_summary": {
            "hybrid": summarize(ragas_hybrid),
            "baseline_template_only": summarize(ragas_baseline_template),
            "baseline_flat_vector_all": summarize(ragas_baseline_flat),
        },
    }

def run_judge_bias_check(chunks_path: str = "data/processed/wrong_chunks.json",
                          ground_truth_path: str = "data/raw/ground_truth.json",
                          theta_sim: float = DEFAULT_THETA_SIM,
                          theta_merge: float = None,
                          alpha: float = None,
                          t: int = 7,
                          k: int = 2,
                          model: str = GENERATOR_MODEL,
                          judge_models: list = ("gpt-4o-mini", "gpt-4o")) -> dict:

    from mot import DEFAULT_THETA_MERGE, DEFAULT_ALPHA
    theta_merge = DEFAULT_THETA_MERGE if theta_merge is None else theta_merge
    alpha = DEFAULT_ALPHA if alpha is None else alpha

    with open(ground_truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    print("=== 產生報告（僅需一次，之後用不同裁判重複評分）===")
    hybrid_reports = generate_reports_from_file(
        chunks_path, theta_sim=theta_sim, theta_merge=theta_merge, alpha=alpha, t=t, k=k, model=model
    )
    baseline_flat_reports = generate_flat_vector_baseline_from_file(
        chunks_path, theta_sim=theta_sim, top_n=t, model=model
    )

    bias_table = []
    per_judge_ragas = {}
    for judge_model in judge_models:
        print(f"\n=== 用 {judge_model} 評分 ===")
        ragas_hybrid = _score_reports_with_judge(
            hybrid_reports,
            lambda r: _hybrid_context_text(r["clause_text"], theta_sim, theta_merge, alpha, t, k=k),
            judge_model, f"hybrid/{judge_model}"
        )
        ragas_baseline = _score_reports_with_judge(
            baseline_flat_reports, lambda r: r.get("context_text", ""), judge_model, f"baseline/{judge_model}"
        )
        per_judge_ragas[judge_model] = {
            "hybrid": {k2: _avg(ragas_hybrid, k2) for k2 in ("context_relevance", "faithfulness", "answer_relevance")},
            "baseline_flat_vector_all": {k2: _avg(ragas_baseline, k2) for k2 in ("context_relevance", "faithfulness", "answer_relevance")},
        }

    for metric in ("context_relevance", "faithfulness", "answer_relevance"):
        row = {"metric": metric}
        for judge_model in judge_models:
            row[judge_model] = {
                "hybrid": per_judge_ragas[judge_model]["hybrid"][metric],
                "baseline_flat_vector_all": per_judge_ragas[judge_model]["baseline_flat_vector_all"][metric],
            }
        row["hybrid_vs_baseline_gap"] = {
            judge_model: round(
                (per_judge_ragas[judge_model]["hybrid"][metric] or 0)
                - (per_judge_ragas[judge_model]["baseline_flat_vector_all"][metric] or 0), 4
            )
            for judge_model in judge_models
        }
        bias_table.append(row)

    law_ga = law_grounding_accuracy(hybrid_reports, ground_truth)
    mand_ga = mandatory_grounding_accuracy(hybrid_reports, ground_truth)

    return {
        "params": {"theta_sim": theta_sim, "theta_merge": theta_merge, "alpha": alpha, "t": t, "k": k,
                    "generator_model": model, "judge_models": list(judge_models)},
        "classification": classification_metrics(hybrid_reports, ground_truth),
        "law_grounding_accuracy_hybrid": law_ga,
        "mandatory_grounding_accuracy_hybrid": mand_ga,
        "bias_comparison_table": bias_table,
    }

if __name__ == "__main__":
    print("=== Step 1：Grid Search（k×t）===")
    grid_result = grid_search()
    os.makedirs("data/evaluation", exist_ok=True)
    with open("data/evaluation/grid_search_results.json", "w", encoding="utf-8") as f:
        json.dump(grid_result, f, ensure_ascii=False, indent=2)

    best_k, best_t = grid_result["best"]["k"], grid_result["best"]["t"]
    print(f"\n最佳設定：k={best_k}, t={best_t}"
          f"（F1={grid_result['best']['f1']:.2%}，"
          f"Law GA={grid_result['best']['law_grounding_accuracy']:.2%}）")

    print("\n=== Step 2：正式評估（含公平版 baseline、獨立裁判）===")
    eval_result = run_evaluation(t=best_t, k=best_k, judge_model="gpt-4o")
    with open("data/evaluation/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    print("\n=== Step 3：自評 vs 獨立裁判偏誤檢查 ===")
    bias_result = run_judge_bias_check(t=best_t, k=best_k)
    with open("data/evaluation/judge_bias_check.json", "w", encoding="utf-8") as f:
        json.dump(bias_result, f, ensure_ascii=False, indent=2)

    print("\n✅ 全部評估完成，結果存於 data/evaluation/")