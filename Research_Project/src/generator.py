import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import time
import random
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from dotenv import load_dotenv
from mot import retrieve_and_merge, DEFAULT_THETA_SIM, DEFAULT_THETA_MERGE, DEFAULT_ALPHA, DEFAULT_T
from preprocess import PROHIBITED_ITEM_KEYWORDS
from build_kg import MANDATORY_LAW_MAPPING, TEMPLATE_MANDATORY_MAPPING

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")

_THROTTLE_SECONDS = float(os.getenv("GENERATOR_THROTTLE_SECONDS", "1"))
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
        f"TPM/RPM 額度設定，或調高 GENERATOR_THROTTLE_SECONDS 環境變數。"
    )

VALID_VERDICTS = {"符合", "不符合", "缺漏", "違反"}

SYSTEM_PROMPT = """你是台灣消費者保護法與健身中心定型化契約審查的專家。
你會收到一段健身中心會員契約的條款原文，以及系統透過知識圖譜檢索到的
比對依據（相似的範本條文、對應的消費者保護法條文、對應的應記載/不得
記載事項）。

請你逐項比對契約條款與檢索到的依據，判斷這條契約條款的狀態，只能是
以下四種之一：
- "符合"：契約條款內容跟範本/應記載事項一致，沒有問題。
- "不符合"：契約條款雖然有處理這個主題，但內容跟範本/法規要求有出入，
  但還不到違反不得記載事項的嚴重程度。
- "缺漏"：範本/應記載事項要求要有的內容，這條契約完全沒有處理或被拿掉。
- "違反"：契約條款的內容命中了消費者保護法或不得記載事項明文禁止的
  情形（例如免除業者責任、片面變更權利、使用「終身」「永久」等字樣）。

判斷時請把「相似範本條文」的原文當成比對基準，逐字比對契約條款跟這段
範本原文的實質差異，不要只憑法條或應記載事項的抽象文字去空想契約
「應該」長什麼樣子。特別注意以下兩種常見的誤判，不要犯：
1. 稱謂用詞差異不是違規：契約範本統一用「甲方／乙方」稱呼雙方，消費者
   保護法及應記載/不得記載事項的原文則用「消費者／業者」，這只是不同
   文件的用詞習慣不同，兩者指涉相同對象，不構成契約有問題，不要因為
   契約用「乙方」而範本引用的法條寫「業者」就判定不符合或違反。
2. 範本原文本身的留白／待填欄位不是缺漏：如果契約條款在該留白之處
   跟範本原文一樣是空格或「○」之類的待填符號（例如入會條件、金額
   欄位等制式待填欄位），這是範本本身的設計，契約逐字照抄範本的留白
   不算缺漏，只有契約把範本「有實際規定的內容」拿掉或改掉才算問題。

【重要】law_basis 與 mandatory_item_basis 只能填「檢索到的比對依據」
裡實際出現過的法條/應記載/不得記載事項，不可以自己額外引用或憑空
推測。如果這次的檢索依據裡完全沒有提供對應消費者保護法或應記載/
不得記載事項（例如只給了範本條文，沒有給法條或應記載事項的原文），
law_basis 和 mandatory_item_basis 請填空陣列 []，並在 violation_reason
裡誠實說明「本次檢索依據不包含法條/應記載事項，僅能依範本條文本身
的差異判斷，無法提供法律依據」，不要自己填一個檢索依據以外的法條
進去湊數。

請只根據提供的檢索依據跟契約條文本身做判斷，不要引用檢索依據以外的
法條或憑空推測契約沒寫出來的內容。如果檢索依據不足以支持你做出明確
判斷，也要誠實在 reason 欄位說明，verdict 選你認為最接近的一種即可。

務必只輸出一個 JSON 物件，不要有任何額外文字、不要用 markdown code
fence 包住，格式如下：
{
  "verdict": "符合 | 不符合 | 缺漏 | 違反",
  "matched_template_article_no": "跟哪一條範本比對（沒有就填 null）",
  "law_basis": ["引用到的法條，例如 消費者保護法第17條第3項"],
  "mandatory_item_basis": ["引用到的應記載/不得記載事項，例如 不得記載六"],
  "violation_reason": "為什麼判定成這個 verdict，具體指出契約條文哪裡有問題",
  "suggested_revision": "具體的修改建議，如果 verdict 是符合則填 null",
  "diff_from_template": "契約條款跟範本/應記載事項的差異說明"
}"""

def _format_context(mot_result: dict) -> str:

    clusters = mot_result["clusters"]
    if not clusters:
        return "（本次檢索沒有找到任何相似的範本條文，無比對依據）"

    lines = []
    for idx, c in enumerate(clusters, start=1):
        lines.append(f"── 候選依據 {idx}（score={c['score']}）──")
        lines.append("相似範本條文（請跟契約條款逐字比對這段原文的差異，"
                      "不要只憑法條抽象原則判斷）：")
        for m in c["members"]:
            lines.append(f"  範本第{m['article_no']}條原文：{m['text']}")
        if c["law_articles"]:
            lines.append("對應消費者保護法：")
            for l in c["law_articles"]:
                para = f"第{l['para_no']}項" if l.get("para_no") else ""
                lines.append(f"  第{l['article_no']}條{para}：{l['text']}")
        if c["mandatory_items"]:
            lines.append("對應應記載/不得記載事項：")
            for mi in c["mandatory_items"]:
                lines.append(f"  {mi['item_type']}{mi['clause_no']}：{mi['text']}")
        lines.append("")
    return "\n".join(lines)

def _parse_llm_json(raw_text: str) -> dict:

    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)

def call_llm_for_verdict(clause_text: str, context_text: str, model: str = GENERATOR_MODEL) -> dict:

    if _THROTTLE_SECONDS > 0:
        time.sleep(_THROTTLE_SECONDS)

    response = _call_with_retry(
        client.chat.completions.create,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"契約條款原文：\n{clause_text}\n\n檢索到的比對依據：\n{context_text}"},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content

    try:
        parsed = _parse_llm_json(raw)
    except (json.JSONDecodeError, TypeError) as e:
        return {
            "verdict": None,
            "parse_error": str(e),
            "raw_response": raw,
        }

    if parsed.get("verdict") not in VALID_VERDICTS:
        parsed["parse_warning"] = f"verdict 不在預期的四選一範圍內：{parsed.get('verdict')!r}"

    return parsed

def _format_law_basis(item_type: str, clause_no: str) -> list:

    refs = MANDATORY_LAW_MAPPING.get((item_type, clause_no), [])
    basis = []
    for art_no, para_no in refs:
        para = f"第{para_no}項" if para_no else ""
        basis.append(f"消費者保護法第{art_no}條{para}")
    return basis

def rule_based_keyword_scan(clause_text: str) -> list:

    hits = []
    for clause_no, keywords in PROHIBITED_ITEM_KEYWORDS.items():
        matched = [kw for kw in keywords if kw and kw in clause_text]
        if matched:
            hits.append({"clause_no": clause_no, "matched_keywords": matched})
    return hits

def _build_rule_based_report(clause_id: str, clause_text: str, rule_hits: list,
                              mot_result: dict) -> dict:

    mandatory_item_basis = [f"不得記載{h['clause_no']}" for h in rule_hits]
    law_basis = []
    for h in rule_hits:
        for ref in _format_law_basis("不得記載", h["clause_no"]):
            if ref not in law_basis:
                law_basis.append(ref)

    matched_keywords_flat = [kw for h in rule_hits for kw in h["matched_keywords"]]
    top_template = None
    if mot_result["clusters"] and mot_result["clusters"][0]["members"]:
        top_template = mot_result["clusters"][0]["members"][0]

    return {
        "clause_id": clause_id,
        "clause_text": clause_text,
        "verdict": "違反",
        "matched_template_article_no": top_template["article_no"] if top_template else None,
        "law_basis": law_basis,
        "mandatory_item_basis": mandatory_item_basis,
        "violation_reason": f"Rule-based 關鍵字掃描命中不得記載事項"
                             f"（命中關鍵字：{'、'.join(matched_keywords_flat)}），"
                             f"依設計，此類明確字樣直接判定違反，不呼叫 LLM。",
        "suggested_revision": f"請刪除或修改含「{'、'.join(matched_keywords_flat)}」字樣的約定，"
                               f"改依應記載/不得記載事項規定辦理。",
        "diff_from_template": f"契約條款出現範本/法規明文禁止的字樣（{'、'.join(matched_keywords_flat)}），"
                               f"範本原文無此類約定。",
        "decision_source": "rule_based",
        "mot_debug": {
            "k": mot_result["params"]["k"],
            "candidate_count_before_merge": mot_result["candidate_count_before_merge"],
            "cluster_count_after_merge": mot_result["cluster_count_after_merge"],
        },
    }

def generate_report(clause_id: str, clause_text: str,
                     theta_sim: float = DEFAULT_THETA_SIM,
                     theta_merge: float = DEFAULT_THETA_MERGE,
                     alpha: float = DEFAULT_ALPHA,
                     t: int = DEFAULT_T,
                     k: int = 2,
                     model: str = GENERATOR_MODEL,
                     use_rule_engine: bool = True) -> dict:

    mot_result = retrieve_and_merge(clause_text, theta_sim=theta_sim,
                                     theta_merge=theta_merge, alpha=alpha, t=t, k=k)

    if use_rule_engine:
        rule_hits = rule_based_keyword_scan(clause_text)
        if rule_hits:
            return _build_rule_based_report(clause_id, clause_text, rule_hits, mot_result)

    if not mot_result["clusters"]:

        return {
            "clause_id": clause_id,
            "clause_text": clause_text,
            "verdict": "無法判斷",
            "matched_template_article_no": None,
            "law_basis": [],
            "mandatory_item_basis": [],
            "violation_reason": "BFS 第1跳沒有找到任何相似度達門檻的範本條文，"
                                 "系統缺乏比對依據，需要人工複核或等 Rule-based "
                                 "覆蓋率檢查補上（可能是整段被改寫成語意完全不同的內容）。",
            "suggested_revision": None,
            "diff_from_template": None,
            "decision_source": "no_grounding",
            "mot_debug": {
                "k": k,
                "candidate_count_before_merge": mot_result["candidate_count_before_merge"],
                "cluster_count_after_merge": mot_result["cluster_count_after_merge"],
            },
        }

    context_text = _format_context(mot_result)
    verdict = call_llm_for_verdict(clause_text, context_text, model=model)

    return {
        "clause_id": clause_id,
        "clause_text": clause_text,
        **verdict,
        "decision_source": "llm",
        "mot_debug": {
            "k": k,
            "candidate_count_before_merge": mot_result["candidate_count_before_merge"],
            "cluster_count_after_merge": mot_result["cluster_count_after_merge"],
            "top_matched_templates": [
                m["article_no"] for c in mot_result["clusters"] for m in c["members"]
            ],
        },
    }

def generate_reports_from_file(chunks_path: str = "data/processed/wrong_chunks.json",
                                theta_sim: float = DEFAULT_THETA_SIM,
                                theta_merge: float = DEFAULT_THETA_MERGE,
                                alpha: float = DEFAULT_ALPHA,
                                t: int = DEFAULT_T,
                                k: int = 2,
                                model: str = GENERATOR_MODEL,
                                use_rule_engine: bool = True) -> list:

    with open(chunks_path, encoding="utf-8") as f:
        contract_chunks = json.load(f)

    reports = []
    for i, chunk in enumerate(contract_chunks, start=1):
        print(f"[{i}/{len(contract_chunks)}] 產生報告：{chunk['id']}")
        report = generate_report(
            chunk["id"], chunk["text"],
            theta_sim=theta_sim, theta_merge=theta_merge, alpha=alpha, t=t, k=k,
            model=model, use_rule_engine=use_rule_engine
        )
        reports.append(report)
    return reports

def check_mandatory_coverage(reports: list) -> dict:

    reports_by_article_no = {}
    for r in reports:

        m = re.search(r"_art(.+)$", r["clause_id"])
        art_no = m.group(1) if m else ("簽約注意事項" if r["clause_id"].endswith("_front") else None)
        if art_no:
            reports_by_article_no.setdefault(art_no, []).append(r)

    item_to_templates = {}
    for template_art_no, mapped_items in TEMPLATE_MANDATORY_MAPPING.items():
        for item_type, clause_no in mapped_items:
            if item_type != "應記載":
                continue
            item_to_templates.setdefault(clause_no, []).append(template_art_no)

    all_item_nos = [str(i) for i in range(1, 25)]

    coverage = {}
    for clause_no, template_art_nos in item_to_templates.items():
        matched_reports = []
        found_any_article = False
        for art_no in template_art_nos:
            if art_no in reports_by_article_no:
                found_any_article = True
                matched_reports.extend(reports_by_article_no[art_no])

        if not found_any_article:
            coverage[clause_no] = {"status": "not_found", "template_article_nos": template_art_nos}
            continue

        has_issue = any(r.get("verdict") not in ("符合",) for r in matched_reports)
        coverage[clause_no] = {
            "status": "flagged" if has_issue else "ok",
            "template_article_nos": template_art_nos,
            "contract_article_nos": [r["clause_id"] for r in matched_reports],
        }

    return coverage

if __name__ == "__main__":
    print(f"=== 用 {GENERATOR_MODEL} 對 wrong_chunks.json 產生審查報告（含 Rule-based 引擎）===\n")
    reports = generate_reports_from_file()

    os.makedirs("data/reports", exist_ok=True)
    out_path = "data/reports/generated_reports.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"\n完成，共 {len(reports)} 份報告，存於 {out_path}")

    coverage = check_mandatory_coverage(reports)
    coverage_path = "data/reports/mandatory_coverage.json"
    with open(coverage_path, "w", encoding="utf-8") as f:
        json.dump(coverage, f, ensure_ascii=False, indent=2)
    print(f"應記載覆蓋率檢查存於 {coverage_path}")

    from collections import Counter
    verdict_counts = Counter(r.get("verdict") for r in reports)
    source_counts = Counter(r.get("decision_source") for r in reports)
    print("\nverdict 分布：")
    for v, cnt in verdict_counts.most_common():
        print(f"  {v}: {cnt}")
    print("\n判定來源分布（rule_based 代表省下的 LLM 呼叫次數）：")
    for s, cnt in source_counts.most_common():
        print(f"  {s}: {cnt}")

    parse_errors = [r["clause_id"] for r in reports if r.get("verdict") is None]
    if parse_errors:
        print(f"\n⚠️ 有 {len(parse_errors)} 筆 LLM 回應解析失敗，建議檢查 raw_response 欄位："
              f"{parse_errors}")