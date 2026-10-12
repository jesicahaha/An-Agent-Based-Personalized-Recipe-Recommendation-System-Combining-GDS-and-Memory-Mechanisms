import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

def get_embedding(text: str) -> list:
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def get_contract_clause_embedding(session, contract_clause_id: str = None, clause_text: str = None) -> tuple:

    if contract_clause_id:
        result = session.run(
            "MATCH (c:ContractClause {id:$id}) RETURN c.text as text, c.embedding as embedding",
            id=contract_clause_id
        ).data()
        if not result:
            raise ValueError(f"找不到 ContractClause: {contract_clause_id}")
        return contract_clause_id, result[0]["embedding"], result[0]["text"]
    elif clause_text:
        embedding = get_embedding(clause_text)
        return clause_text, embedding, clause_text
    else:
        raise ValueError("必須提供 contract_clause_id 或 clause_text 其中之一")

def find_similar_templates(session, query_embedding: list, theta_sim: float,
                            precomputed_edges_from: str = None) -> list:

    if precomputed_edges_from:
        rows = session.run(
            """MATCH (c:ContractClause {id:$cid})-[r:SIMILAR_TO]->(t:TemplateClause)
               WHERE r.score >= $theta
               RETURN t.id as id, t.text as text, t.article_no as article_no, r.score as score
               ORDER BY r.score DESC""",
            cid=precomputed_edges_from, theta=theta_sim
        ).data()
        if rows:
            return rows

    rows = session.run(
        "MATCH (t:TemplateClause) RETURN t.id as id, t.text as text, "
        "t.article_no as article_no, t.embedding as embedding"
    ).data()
    matches = []
    for r in rows:
        score = cosine_similarity(query_embedding, r["embedding"])
        if score >= theta_sim:
            matches.append({
                "id": r["id"], "text": r["text"],
                "article_no": r["article_no"], "score": score
            })
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches

def find_similar_any_label(session, query_embedding: list, theta_sim: float,
                            labels: tuple = ("TemplateClause", "LawArticle", "MandatoryItem"),
                            top_n: int = None) -> list:

    label_filter = " OR ".join(f"n:{l}" for l in labels)
    rows = session.run(
        f"MATCH (n) WHERE {label_filter} "
        "RETURN n.id as id, n.text as text, labels(n)[0] as label, "
        "n.article_no as article_no, n.para_no as para_no, "
        "n.clause_no as clause_no, n.item_type as item_type, "
        "n.embedding as embedding"
    ).data()

    matches = []
    for r in rows:
        score = cosine_similarity(query_embedding, r["embedding"])
        if score >= theta_sim:
            matches.append({
                "id": r["id"], "text": r["text"], "label": r["label"],
                "article_no": r.get("article_no"), "para_no": r.get("para_no"),
                "clause_no": r.get("clause_no"), "item_type": r.get("item_type"),
                "score": score,
            })
    matches.sort(key=lambda x: x["score"], reverse=True)
    if top_n is not None:
        matches = matches[:top_n]
    return matches

def get_grounding_nodes(session, template_id: str) -> dict:

    law_rows = session.run(
        """MATCH (t:TemplateClause {id:$tid})-[:BASED_ON]->(l:LawArticle)
           RETURN l.id as id, l.text as text, l.article_no as article_no, l.para_no as para_no""",
        tid=template_id
    ).data()
    mandatory_rows = session.run(
        """MATCH (t:TemplateClause {id:$tid})-[:IMPLEMENTS]->(m:MandatoryItem)
           RETURN m.id as id, m.text as text, m.clause_no as clause_no, m.item_type as item_type""",
        tid=template_id
    ).data()
    return {"law_articles": law_rows, "mandatory_items": mandatory_rows}

def bfs_retrieve(contract_clause_id: str = None, clause_text: str = None,
                  k: int = 2, theta_sim: float = 0.75) -> dict:

    if k > 2:
        print(f"⚠️ k={k} 超過目前定義的兩跳上限（Contract->Template->Law），"
              f"目前沒有設計第三跳要怎麼走，本次當作 k=2 處理。")
        k = 2
    if k < 1:
        raise ValueError("k 至少要是 1")

    with driver.session() as session:
        query_id, query_embedding, query_text = get_contract_clause_embedding(
            session, contract_clause_id=contract_clause_id, clause_text=clause_text
        )

        template_matches = find_similar_templates(
            session, query_embedding, theta_sim,
            precomputed_edges_from=contract_clause_id
        )

        law_articles = {}
        mandatory_items = {}

        if k >= 2:
            for m in template_matches:
                grounding = get_grounding_nodes(session, m["id"])
                for l in grounding["law_articles"]:
                    law_articles[l["id"]] = l
                for mi in grounding["mandatory_items"]:
                    mandatory_items[mi["id"]] = mi

        return {
            "query": {"id": query_id, "text": query_text},
            "template_matches": template_matches,
            "law_articles": list(law_articles.values()),
            "mandatory_items": list(mandatory_items.values()),
            "params": {"k": k, "theta_sim": theta_sim},
        }

def bfs_retrieve_from_file(chunks_path: str = "data/processed/wrong_chunks.json",
                            k: int = 2, theta_sim: float = 0.75) -> dict:

    with open(chunks_path, encoding="utf-8") as f:
        contract_chunks = json.load(f)

    results = {}
    for chunk in contract_chunks:
        results[chunk["id"]] = bfs_retrieve(clause_text=chunk["text"], k=k, theta_sim=theta_sim)
    return results

def _summarize(all_results: dict) -> dict:

    summary = {}
    for cid, g in all_results.items():
        summary[cid] = {
            "query_text_preview": g["query"]["text"][:40],
            "matched_templates": [
                {"id": m["id"], "article_no": m.get("article_no"), "score": m.get("score")}
                for m in g["template_matches"]
            ],
            "law_articles": [l["article_no"] for l in g["law_articles"]],
            "mandatory_items": [f'{m["item_type"]}{m["clause_no"]}' for m in g["mandatory_items"]],
        }
    return summary

if __name__ == "__main__":

    os.makedirs("data/retrieval", exist_ok=True)

    for k in (1, 2):
        print(f"\n=== 對 wrong_chunks.json 執行 BFS 檢索（k={k}, θ_sim=0.75）===")
        all_results = bfs_retrieve_from_file("data/processed/wrong_chunks.json", k=k, theta_sim=0.75)
        summary = _summarize(all_results)

        out_path = f"data/retrieval/bfs_k{k}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"完成，共 {len(all_results)} 個契約條款，結果存於 {out_path}")

        no_match = [cid for cid, g in all_results.items() if not g["template_matches"]]
        if no_match:
            print(f"⚠️ 有 {len(no_match)} 個節點在 θ_sim=0.75 下完全沒找到任何相似範本條款："
                  f"{no_match}")

        if k == 2:
            no_grounding = [
                cid for cid, g in all_results.items()
                if g["template_matches"] and not g["law_articles"] and not g["mandatory_items"]
            ]
            if no_grounding:
                print(f"⚠️ 有 {len(no_grounding)} 個節點雖然第1跳配到範本，但第2跳查無法條/應記載事項"
                      f"（該範本條文不在 TEMPLATE_LAW_MAPPING/TEMPLATE_MANDATORY_MAPPING 裡）："
                      f"{no_grounding}")