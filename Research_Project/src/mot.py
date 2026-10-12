import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from retriever import driver, bfs_retrieve, get_grounding_nodes

DEFAULT_THETA_SIM = 0.75
DEFAULT_THETA_MERGE = 0.3
DEFAULT_ALPHA = 0.5
DEFAULT_T = 7

def jaccard(nodes_a: set, nodes_b: set) -> float:
    if not nodes_a and not nodes_b:
        return 0.0
    union = nodes_a | nodes_b
    if not union:
        return 0.0
    return len(nodes_a & nodes_b) / len(union)

def build_candidate_subgraphs(session, template_matches: list, k: int = 2) -> list:

    subgraphs = []
    for m in template_matches:
        if k >= 2:
            grounding = get_grounding_nodes(session, m["id"])
            law_ids = {l["id"] for l in grounding["law_articles"]}
            mandatory_ids = {mi["id"] for mi in grounding["mandatory_items"]}
            law_articles = grounding["law_articles"]
            mandatory_items = grounding["mandatory_items"]
        else:
            law_ids, mandatory_ids = set(), set()
            law_articles, mandatory_items = [], []
        nodes = {m["id"]} | law_ids | mandatory_ids
        subgraphs.append({
            "template_id": m["id"],
            "template_article_no": m.get("article_no"),

            "template_text": m.get("text", ""),
            "template_score": m.get("score") or 0.0,
            "nodes": nodes,
            "law_articles": law_articles,
            "mandatory_items": mandatory_items,
        })
    return subgraphs

def merge_subgraphs(subgraphs: list, theta_merge: float) -> list:

    n = len(subgraphs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if jaccard(subgraphs[i]["nodes"], subgraphs[j]["nodes"]) >= theta_merge:
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(subgraphs[i])
    return list(groups.values())

def score_cluster(members: list, alpha: float) -> tuple:

    relatedness = sum(m["template_score"] for m in members) / len(members)
    if len(members) < 2:
        internal_j = 0.0
    else:
        pair_scores = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair_scores.append(jaccard(members[i]["nodes"], members[j]["nodes"]))
        internal_j = sum(pair_scores) / len(pair_scores)
    score = alpha * relatedness + (1 - alpha) * internal_j
    return score, relatedness, internal_j

def mot_merge_and_rank(template_matches: list, session,
                        theta_merge: float = DEFAULT_THETA_MERGE,
                        alpha: float = DEFAULT_ALPHA,
                        t: int = None,
                        k: int = 2) -> list:

    subgraphs = build_candidate_subgraphs(session, template_matches, k=k)
    if not subgraphs:
        return []

    groups = merge_subgraphs(subgraphs, theta_merge)

    clusters = []
    for members in groups:
        score, relatedness, internal_j = score_cluster(members, alpha)
        all_law, all_mandatory = {}, {}
        for m in members:
            for l in m["law_articles"]:
                all_law[l["id"]] = l
            for mi in m["mandatory_items"]:
                all_mandatory[mi["id"]] = mi
        clusters.append({
            "members": [
                {"template_id": m["template_id"], "article_no": m["template_article_no"],
                 "text": m["template_text"], "score": m["template_score"]}
                for m in members
            ],
            "law_articles": list(all_law.values()),
            "mandatory_items": list(all_mandatory.values()),
            "score": round(score, 4),
            "relatedness": round(relatedness, 4),
            "internal_jaccard": round(internal_j, 4),
        })

    clusters.sort(key=lambda c: c["score"], reverse=True)
    if t is not None:
        return clusters[:t]
    return clusters

def retrieve_and_merge(clause_text: str,
                        theta_sim: float = DEFAULT_THETA_SIM,
                        theta_merge: float = DEFAULT_THETA_MERGE,
                        alpha: float = DEFAULT_ALPHA,
                        t: int = DEFAULT_T,
                        k: int = 2) -> dict:

    bfs_result = bfs_retrieve(clause_text=clause_text, k=1, theta_sim=theta_sim)
    with driver.session() as session:

        all_clusters = mot_merge_and_rank(
            bfs_result["template_matches"], session,
            theta_merge=theta_merge, alpha=alpha, t=None, k=k
        )
    return {
        "query": bfs_result["query"],
        "candidate_count_before_merge": len(bfs_result["template_matches"]),
        "cluster_count_after_merge": len(all_clusters),
        "clusters": all_clusters[:t],
        "params": {"theta_sim": theta_sim, "theta_merge": theta_merge, "alpha": alpha, "t": t, "k": k},
    }

def mot_retrieve_from_file(chunks_path: str = "data/processed/wrong_chunks.json",
                            theta_sim: float = DEFAULT_THETA_SIM,
                            theta_merge: float = DEFAULT_THETA_MERGE,
                            alpha: float = DEFAULT_ALPHA,
                            t: int = DEFAULT_T,
                            k: int = 2) -> dict:

    with open(chunks_path, encoding="utf-8") as f:
        contract_chunks = json.load(f)

    results = {}
    for chunk in contract_chunks:
        results[chunk["id"]] = retrieve_and_merge(
            chunk["text"], theta_sim=theta_sim, theta_merge=theta_merge, alpha=alpha, t=t, k=k
        )
    return results

def _summarize(all_results: dict) -> dict:

    summary = {}
    for cid, g in all_results.items():
        summary[cid] = {
            "query_text_preview": g["query"]["text"][:40],
            "candidate_count_before_merge": g["candidate_count_before_merge"],
            "cluster_count_after_merge": g["cluster_count_after_merge"],
            "shown_after_truncation_to_t": len(g["clusters"]),
            "clusters": [
                {
                    "members_article_no": [m["article_no"] for m in c["members"]],
                    "law_articles": [l["article_no"] for l in c["law_articles"]],
                    "mandatory_items": [f'{m["item_type"]}{m["clause_no"]}' for m in c["mandatory_items"]],
                    "score": c["score"],
                    "relatedness": c["relatedness"],
                    "internal_jaccard": c["internal_jaccard"],
                }
                for c in g["clusters"]
            ],
        }
    return summary

if __name__ == "__main__":
    print(f"=== 對 wrong_chunks.json 執行 BFS+MOT "
          f"(θ_sim={DEFAULT_THETA_SIM}, θ_merge={DEFAULT_THETA_MERGE}, "
          f"α={DEFAULT_ALPHA}, t={DEFAULT_T}) ===\n")
    all_results = mot_retrieve_from_file()
    summary = _summarize(all_results)

    os.makedirs("data/retrieval", exist_ok=True)
    out_path = "data/retrieval/mot_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"完成，共 {len(all_results)} 個契約條款，結果存於 {out_path}")

    no_reduction = [
        cid for cid, g in all_results.items()
        if g["candidate_count_before_merge"] > 1
        and g["cluster_count_after_merge"] == g["candidate_count_before_merge"]
    ]
    if no_reduction:
        print(f"\n⚠️ 有 {len(no_reduction)} 個節點合併前有多個候選範本，"
              f"但合併後群集數完全沒變少（可能是 θ_merge={DEFAULT_THETA_MERGE} "
              f"設太高，候選子圖之間節點重疊不夠多，判定不到同一群）："
              f"{no_reduction}")

    empty = [cid for cid, g in all_results.items() if g["cluster_count_after_merge"] == 0]
    if empty:
        print(f"\n⚠️ 有 {len(empty)} 個節點完全沒有候選子圖（第1跳就沒配到範本）："
              f"{empty}")