import re
import json
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def split_law_to_paragraph(filepath: str, source_type: str, doc_name: str) -> list[dict]:

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    article_pattern = r'^(第\s*[\d一二三四五六七八九十百]+(?:-\d+)?\s*條)'
    parts = re.split(article_pattern, text, flags=re.MULTILINE)

    chunks = []
    for i in range(1, len(parts), 2):
        article_header = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""

        raw_no = re.search(r'第\s*([\d一二三四五六七八九十百]+(?:-\d+)?)\s*條', article_header)
        article_no = raw_no.group(1).replace(" ", "") if raw_no else str(i)

        if re.search(r'章|節', article_header):
            continue

        raw_lines = [l.strip() for l in content.split('\n') if l.strip()]

        paragraphs = []
        current = []
        for line in raw_lines:
            is_sub_item = bool(re.match(r'^[一二三四五六七八九十]+、|^\d+\．|^\d+\.|^[（(][一二三四五六七八九十]+[）)]', line))
            if is_sub_item:
                if current:
                    current.append(line)
                else:
                    current = [line]
            else:
                if current:
                    paragraphs.append(' '.join(current))
                current = [line]
        if current:
            paragraphs.append(' '.join(current))

        for para_idx, para_text in enumerate(paragraphs, start=1):
            if not para_text:
                continue
            chunks.append({
                "id": f"{source_type}_{doc_name}_art{article_no}_para{para_idx}",
                "source_type": source_type,
                "doc_name": doc_name,
                "article_no": article_no,
                "para_no": str(para_idx),
                "clause_no": None,
                "text": para_text,
                "full_article": article_header + " " + content,
                "level": "paragraph"
            })

    return chunks

def split_contract_to_article(filepath: str, source_type: str, doc_name: str) -> list[dict]:

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = r'^(第[一二三四五六七八九十百]+條)'
    parts = re.split(pattern, text, flags=re.MULTILINE)

    chunks = []

    front_matter = parts[0].strip() if parts else ""
    if front_matter:

        front_lines = front_matter.split('\n')
        if front_lines and ('契約' in front_lines[0] or '範本' in front_lines[0]) and '簽約注意事項' not in front_lines[0]:
            front_lines = front_lines[1:]
        front_clean = '\n'.join(front_lines).strip()

        front_clean = re.sub(r'【[^】]*】', '', front_clean).strip()
        front_clean = re.sub(r'\s+', ' ', front_clean)
        if front_clean:
            chunks.append({
                "id": f"{source_type}_{doc_name}_front",
                "source_type": source_type,
                "doc_name": doc_name,
                "article_no": "簽約注意事項",
                "para_no": None,
                "clause_no": None,
                "text": front_clean,
                "full_article": front_clean,
                "level": "article"
            })

    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""

        raw_no = re.search(r'第([一二三四五六七八九十百]+)條', header)
        article_no = raw_no.group(1) if raw_no else str(i)

        full_text = header + " " + content

        clean_text = re.sub(r'【[^】]*】', '', full_text).strip()
        clean_text = re.sub(r'\s+', ' ', clean_text)

        if clean_text:
            chunks.append({
                "id": f"{source_type}_{doc_name}_art{article_no}",
                "source_type": source_type,
                "doc_name": doc_name,
                "article_no": article_no,
                "para_no": None,
                "clause_no": None,
                "text": clean_text,
                "full_article": clean_text,
                "level": "article"
            })

    return chunks

PROHIBITED_ITEM_KEYWORDS = {
    "一": ["拋棄審閱期", "拋棄...審閱", "已審閱契約", "同意等類此字樣", "不另給予審閱期間"],
    "二": ["免除", "限制其賠償責任", "不負任何賠償責任"],
    "三": ["片面變更", "得隨時調整", "不另行通知"],
    "四": ["片面調高", "自行調整", "得調高"],
    "五": [],
    "六": ["終身", "永久"],
    "七": [],
    "八": ["正本", "業者保管", "統一保管", "不另行交付"],
    "九": ["概不退還"],
    "十": [],
    "十一": ["自動續約", "不另通知", "不另行通知"],
    "十二": [],
    "十三": ["附條件"],
}

def split_mandatory_items(filepath: str, source_type: str, doc_name: str) -> list[dict]:

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    section_pattern = r'^(壹、應記載事項|貳、不得記載事項)'
    section_parts = re.split(section_pattern, text, flags=re.MULTILINE)

    chunks = []

    for i in range(1, len(section_parts), 2):
        section_header = section_parts[i].strip()
        section_content = section_parts[i + 1].strip() if i + 1 < len(section_parts) else ""
        item_type = "應記載" if "應記載" in section_header else "不得記載"

        item_pattern = r'^([一二三四五六七八九十]+、)'
        item_parts = re.split(item_pattern, section_content, flags=re.MULTILINE)

        for j in range(1, len(item_parts), 2):
            item_header = item_parts[j].strip().rstrip('、')
            item_content = item_parts[j + 1].strip() if j + 1 < len(item_parts) else ""

            full_text = item_header + "、" + item_content
            clean_text = re.sub(r'\s+', ' ', full_text).strip()

            keywords = PROHIBITED_ITEM_KEYWORDS.get(item_header, []) if item_type == "不得記載" else []

            chunks.append({
                "id": f"{source_type}_{doc_name}_{item_type}{item_header}",
                "source_type": source_type,
                "doc_name": doc_name,
                "article_no": None,
                "para_no": None,
                "clause_no": item_header,
                "item_type": item_type,
                "keywords": keywords,
                "text": clean_text,
                "full_article": clean_text,
                "level": "item"
            })

    return chunks

def save_chunks(chunks: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"儲存 {len(chunks)} 個節點到 {output_path}")

if __name__ == "__main__":

    law_chunks = split_law_to_paragraph(
        "data/raw/consumer_protection_law.txt",
        source_type="law",
        doc_name="消費者保護法"
    )
    save_chunks(law_chunks, "data/processed/law_chunks.json")

    template_chunks = split_contract_to_article(
        "data/raw/template.txt",
        source_type="template",
        doc_name="健身中心定型化契約範本"
    )
    save_chunks(template_chunks, "data/processed/template_chunks.json")

    mandatory_chunks = split_mandatory_items(
        "data/raw/mandatory_items.txt",
        source_type="mandatory",
        doc_name="健身中心定型化契約應記載及不得記載事項"
    )
    save_chunks(mandatory_chunks, "data/processed/mandatory_chunks.json")

    wrong_chunks = split_contract_to_article(
        "data/raw/wrong_contract.txt",
        source_type="contract",
        doc_name="錯誤契約"
    )
    save_chunks(wrong_chunks, "data/processed/wrong_chunks.json")

    print("\n── 切分完成 ──")
    print(f"消費者保護法（項級別）：      {len(law_chunks)} 個節點")
    print(f"正確範本（條級別）：          {len(template_chunks)} 個節點")
    print(f"應記載/不得記載事項（點級別）：{len(mandatory_chunks)} 個節點")
    print(f"待審查契約（條級別）：        {len(wrong_chunks)} 個節點")
    print(f"合計：                        {len(law_chunks)+len(template_chunks)+len(mandatory_chunks)+len(wrong_chunks)} 個節點")