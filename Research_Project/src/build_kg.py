import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import hashlib
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

def get_embedding(text: str) -> list:
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

def clear_graph(session):
    session.run("MATCH (n) DETACH DELETE n")
    print("已清空舊圖譜")

def create_node(session, chunk: dict, embedding: list):

    label = {
        "law": "LawArticle",
        "template": "TemplateClause",
        "mandatory": "MandatoryItem",
    }[chunk["source_type"]]

    if label == "MandatoryItem":
        session.run(
            """MERGE (n:MandatoryItem {id: $id})
            SET n.text=$text, n.clause_no=$clause_no, n.item_type=$item_type,
                n.keywords=$keywords, n.doc_name=$doc_name, n.level=$level,
                n.embedding=$embedding""",
            id=chunk["id"], text=chunk["text"], clause_no=chunk.get("clause_no", ""),
            item_type=chunk.get("item_type", ""), keywords=chunk.get("keywords", []),
            doc_name=chunk["doc_name"], level=chunk.get("level", ""), embedding=embedding
        )
    else:
        session.run(
            f"""MERGE (n:{label} {{id: $id}})
            SET n.text=$text, n.article_no=$article_no, n.para_no=$para_no,
                n.doc_name=$doc_name, n.level=$level, n.embedding=$embedding""",
            id=chunk["id"], text=chunk["text"], article_no=chunk.get("article_no", ""),
            para_no=chunk.get("para_no", ""), doc_name=chunk["doc_name"],
            level=chunk.get("level", ""), embedding=embedding
        )

def create_based_on_edge(session, tid, lid):
    session.run("""MATCH (t {id:$tid}),(l:LawArticle {id:$lid}) MERGE (t)-[:BASED_ON]->(l)""",
                tid=tid, lid=lid)

def get_law_article_ids(session, article_no: str, para_no: str = None) -> list:

    if para_no is not None:
        result = session.run(
            "MATCH (l:LawArticle {article_no:$art_no, para_no:$para_no}) RETURN l.id as lid",
            art_no=article_no, para_no=para_no
        ).data()
    else:
        result = session.run(
            "MATCH (l:LawArticle {article_no:$art_no}) RETURN l.id as lid ORDER BY l.para_no",
            art_no=article_no
        ).data()
    return [r["lid"] for r in result]

def create_implements_edge(session, tid, mid):
    session.run(
        """MATCH (t:TemplateClause {id:$tid}),(m:MandatoryItem {id:$mid})
        MERGE (t)-[:IMPLEMENTS]->(m)""",
        tid=tid, mid=mid
    )

TEMPLATE_LAW_MAPPING = {
    "簽約注意事項": [("11-1", "1"), ("11-1", "2")],
    "一": [("17", "2"), ("17", "5")],
    "二": [("17", "3"), ("17", "4")],
    "三": [("11", None)],
    "四": [("17", "2")],
    "五": [("17", "3"), ("17", "4")],
    "六": [("17", "2")],
    "七": [("17", "3"), ("17", "4")],
    "八": [("18", "1")],
    "九": [("17", "2")],
    "十": [("17", "2")],
    "十一": [("17", "3"), ("17", "4")],
    "十二": [("17", "3"), ("17", "4")],
    "十三": [("19", None), ("17", "3"), ("17", "4")],
    "十四": [("17", "2")],
    "十五": [("17", "2")],
    "十六": [("17", "2")],
    "十七": [("17", "2")],
    "十八": [("17", "2")],
    "十九": [("17", "2")],
    "二十": [("17", "2")],
    "二十一": [("17", "2"), ("17", "5")],
    "二十二": [("17", "3"), ("17", "4")],
    "二十三": [("17", "2")],
    "二十四": [("7", None)],
    "二十五": [("11", None)],
    "二十七": [("7", "3")],
    "二十九": [("17", "2")],
    "三十": [("22", "1")],
    "三十四": [("47", "1")],
    "三十五": [("11", "2")],
    "三十七": [("13", "2"), ("13", "3")],

}

MANDATORY_LAW_MAPPING = {
    ("應記載", "一"): [("11-1", "1"), ("11-1", "2")],
    ("應記載", "二"): [("17", "2")],
    ("應記載", "三"): [("17", "2")],
    ("應記載", "四"): [("17", "2")],
    ("應記載", "五"): [("17", "2")],
    ("應記載", "六"): [("17", "2")],
    ("應記載", "七"): [("18", "1"), ("11", "1")],
    ("應記載", "八"): [("17", "2")],
    ("應記載", "九"): [("17", "2"), ("17", "4")],
    ("應記載", "十"): [("17", "2")],
    ("應記載", "十一"): [("17", "2")],
    ("應記載", "十二"): [("17", "2")],
    ("應記載", "十三"): [("17", "2")],
    ("應記載", "十四"): [("17", "2")],
    ("應記載", "十五"): [("17", "2")],
    ("應記載", "十六"): [("17", "2")],
    ("應記載", "十七"): [("17", "2")],
    ("應記載", "十八"): [("17", "2")],
    ("應記載", "十九"): [("17", "2")],
    ("應記載", "二十"): [("17", "2")],
    ("應記載", "二十一"): [("17", "2")],
    ("應記載", "二十二"): [("17", "2"), ("17", "5")],
    ("應記載", "二十三"): [("47", "1")],
    ("應記載", "二十四"): [("22", "1")],
    ("不得記載", "一"): [("11-1", "2"), ("11-1", "3")],
    ("不得記載", "二"): [("10-1", "1"), ("17", "3")],
    ("不得記載", "三"): [("17", "3")],
    ("不得記載", "四"): [("17", "3")],
    ("不得記載", "五"): [("12", "1")],
    ("不得記載", "六"): [("22", "1"), ("17", "3")],
    ("不得記載", "七"): [("12", "1"), ("12", "2")],
    ("不得記載", "八"): [("13", "2"), ("13", "3")],
    ("不得記載", "九"): [("17", "3")],
    ("不得記載", "十"): [("17", "3")],
    ("不得記載", "十一"): [("17", "3")],
    ("不得記載", "十二"): [("17", "3")],
    ("不得記載", "十三"): [("17", "3")],
}

TEMPLATE_MANDATORY_MAPPING = {
    "簽約注意事項": [("應記載", "一")],
    "一": [("應記載", "二"), ("應記載", "三")],
    "二": [("應記載", "六")],
    "三": [],
    "四": [("應記載", "四")],
    "五": [("應記載", "五")],
    "六": [("應記載", "五")],
    "七": [("不得記載", "四")],
    "八": [("應記載", "七")],
    "九": [],
    "十": [("應記載", "十七")],
    "十一": [("應記載", "八")],
    "十二": [("應記載", "九")],
    "十三": [("應記載", "十")],
    "十四": [("應記載", "十一")],
    "十五": [("應記載", "十二")],
    "十六": [("應記載", "十三")],
    "十七": [("應記載", "十四")],
    "十八": [("應記載", "十五")],
    "十九": [("應記載", "十六")],
    "二十": [("應記載", "十九")],
    "二十一": [("應記載", "二十二")],
    "二十二": [("應記載", "二十"), ("不得記載", "十三")],
    "二十三": [("應記載", "二十一")],
    "二十四": [],
    "二十五": [],
    "二十六": [],
    "二十七": [("不得記載", "二")],
    "二十八": [],
    "二十九": [("應記載", "十八"), ("不得記載", "十一")],
    "三十": [("應記載", "二十四")],
    "三十一": [],
    "三十二": [],
    "三十三": [],
    "三十四": [("應記載", "二十三")],
    "三十五": [("不得記載", "七")],
    "三十六": [],
    "三十七": [("不得記載", "八")],
}

def build_graph():
    with open("data/processed/law_chunks.json", encoding="utf-8") as f:
        law_chunks = json.load(f)
    with open("data/processed/template_chunks.json", encoding="utf-8") as f:
        template_chunks = json.load(f)
    with open("data/processed/mandatory_chunks.json", encoding="utf-8") as f:
        mandatory_chunks = json.load(f)

    all_chunks = law_chunks + template_chunks + mandatory_chunks
    print(f"共 {len(all_chunks)} 個節點（法條:{len(law_chunks)} / 範本:{len(template_chunks)} / "
          f"應記載不得記載:{len(mandatory_chunks)}）—— 僅靜態知識庫，不含待審契約")

    def cache_key(chunk: dict) -> str:

        return f"{chunk['id']}::{hashlib.md5(chunk['text'].encode('utf-8')).hexdigest()[:8]}"

    cache_path = "data/embeddings/embeddings.pkl"
    os.makedirs("data/embeddings", exist_ok=True)
    if os.path.exists(cache_path):
        print("讀取已快取的 embeddings...")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    missing = [c for c in all_chunks if cache_key(c) not in cache]
    if missing:
        print(f"計算 {len(missing)} 個節點的 embeddings（含新節點與內容已變動的節點）...")
        for chunk in tqdm(missing):
            cache[cache_key(chunk)] = get_embedding(chunk["text"])
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
    else:
        print("所有節點皆命中快取，無需重新計算")

    embeddings = {c["id"]: cache[cache_key(c)] for c in all_chunks}

    print("\n建立節點...")
    with driver.session() as session:
        clear_graph(session)
        for chunk in tqdm(all_chunks):
            create_node(session, chunk, embeddings[chunk["id"]])

    print("\n建立範本 -> 法條 的 BASED_ON 邊...")
    based_on_count = 0
    with driver.session() as session:
        for t_chunk in template_chunks:
            art_no = t_chunk.get("article_no", "")
            for law_art_no, law_para_no in TEMPLATE_LAW_MAPPING.get(art_no, []):
                for lid in get_law_article_ids(session, law_art_no, law_para_no):
                    create_based_on_edge(session, t_chunk["id"], lid)
                    based_on_count += 1
    print(f"建立 {based_on_count} 條 TemplateClause-[BASED_ON]->LawArticle 邊")

    print("\n建立應記載/不得記載事項 -> 法條 的 BASED_ON 邊...")
    mandatory_based_on_count = 0
    with driver.session() as session:
        for m_chunk in mandatory_chunks:
            key = (m_chunk["item_type"], m_chunk["clause_no"])
            for law_art_no, law_para_no in MANDATORY_LAW_MAPPING.get(key, []):
                for lid in get_law_article_ids(session, law_art_no, law_para_no):
                    create_based_on_edge(session, m_chunk["id"], lid)
                    mandatory_based_on_count += 1
    print(f"建立 {mandatory_based_on_count} 條 MandatoryItem-[BASED_ON]->LawArticle 邊")

    print("\n建立範本 -> 應記載/不得記載事項 的 IMPLEMENTS 邊...")
    implements_count = 0
    with driver.session() as session:
        for t_chunk in template_chunks:
            art_no = t_chunk.get("article_no", "")
            for item_type, clause_no in TEMPLATE_MANDATORY_MAPPING.get(art_no, []):
                m_id = f"mandatory_健身中心定型化契約應記載及不得記載事項_{item_type}{clause_no}"
                result = session.run(
                    "MATCH (m:MandatoryItem {id:$mid}) RETURN m.id as mid", mid=m_id
                ).data()
                if result:
                    create_implements_edge(session, t_chunk["id"], m_id)
                    implements_count += 1
                else:
                    print(f"  ⚠️ 找不到對應的 MandatoryItem 節點：{m_id}（檢查 doc_name 是否一致）")
    print(f"建立 {implements_count} 條 TemplateClause-[IMPLEMENTS]->MandatoryItem 邊")

    print("\n✅ 知識圖譜建立完成！")
    with driver.session() as session:
        for row in session.run("MATCH (n) RETURN labels(n)[0] as label,count(n) as cnt").data():
            print(f"  {row['label']}: {row['cnt']} 個節點")
        for row in session.run("MATCH ()-[r]->() RETURN type(r) as rel,count(r) as cnt").data():
            print(f"  {row['rel']}: {row['cnt']} 條邊")

if __name__ == "__main__":
    build_graph()