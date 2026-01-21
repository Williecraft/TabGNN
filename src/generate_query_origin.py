import pandas as pd
from OllamaAgent import Agent as oAgent
# from GeminiAgent import Agent as gAgent
import datetime
import csv
import json
import os
import math

SOURCE = "train/spider_multitabqa"
FILENAME = "query_ollama.jsonl"
MAX_RETRIES = 5  # LLM 輸出不合規時的重試次數
FIX = True # 從頭開始檢查有沒有生成失敗的
MODEL = "Ollama"

TABLE_PATH = f"/user_data/TabGNN/data/table/{SOURCE}/table.jsonl"
QUERY_PATH = f"/user_data/TabGNN/data/generated/{SOURCE}/{FILENAME}"

os.makedirs(os.path.dirname(QUERY_PATH), exist_ok=True)

def json2csv(jtable: dict, top=10) -> str:
    import csv
    import pandas as pd

    def parse_csv_row(s: str) -> list[str]:
        return next(csv.reader([s]))

    # ---- header ----
    header = jtable.get("header", [])
    if isinstance(header, str):
        header = parse_csv_row(header)
    elif isinstance(header, list):
        if len(header) == 1 and isinstance(header[0], str) and ("," in header[0]):
            header = parse_csv_row(header[0])
        else:
            header = [("" if h is None else str(h)) for h in header]
    else:
        header = [str(header)]

    # ---- instances ----
    instances = jtable.get("instances", [])
    rows = instances if top is None else instances[:int(top)]

    parsed_rows = []
    for r in rows:
        if isinstance(r, str):
            parsed_rows.append(parse_csv_row(r))
        elif isinstance(r, list):
            parsed_rows.append([("" if x is None else str(x)) for x in r])
        else:
            parsed_rows.append([str(r)])

    # ---- 對齊欄數（避免某些列欄位數不一致）----
    ncol = len(header)
    fixed_rows = []
    for row in parsed_rows:
        if len(row) < ncol:
            row = row + [""] * (ncol - len(row))
        elif len(row) > ncol:
            row = row[:ncol]
        fixed_rows.append(row)

    df = pd.DataFrame(fixed_rows, columns=header)
    return df.to_csv(index=False)


def is_valid_output(obj):
    """
    規則：
      - 必須是 dict，含 'headers'（list[str]）與 'questions'（list[str]）
      - headers 去除空字串/ 'nan' / 'Unnamed:' 後仍需非空
      - 問題數量 > 0.5 * headers 數
    """
    if not isinstance(obj, dict):
        return False, "Not a dict"
    if "headers" not in obj or "questions" not in obj:
        return False, "Missing keys"
    headers = obj["headers"]
    questions = obj["questions"]
    if not isinstance(headers, list) or not all(isinstance(h, str) for h in headers):
        return False, "Headers not list[str]"
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        return False, "Questions not list[str]"

    # 清掉無意義表頭
    def bad(h):
        hs = h.strip()
        if hs == "": return True
        lower = hs.lower()
        return lower == "nan" or lower.startswith("unnamed")
    cleaned = [h for h in headers if not bad(h)]
    if len(cleaned) == 0:
        return False, "No valid headers after cleaning"

    # min_q = math.floor(len(cleaned) / 2) + 1  # > 0.5 * headers
    # if len(questions) < min_q:
    #     return False, f"Too few questions (< {min_q})"

    return True, ""

def parse_json_strict(s):
    """
    嘗試解析 LLM 回覆為 JSON。
    一些模型可能前後多了註解或換行，這裡直接嘗試 json.loads，失敗就拋錯。
    """
    return json.loads(s)

with open(TABLE_PATH, "r", encoding="utf-8") as f:
    table_lines = [json.loads(line) for line in f.readlines()]

if not os.path.exists(QUERY_PATH):
    with open(QUERY_PATH, "w", encoding="utf-8"): pass

with open(QUERY_PATH, "r", encoding="utf-8") as f:
    query_lines = [json.loads(line) for line in f.readlines()]

with open("/user_data/TabGNN/config/api_keys.json", "r", encoding="utf-8") as f:
    api_key = json.load(f)[MODEL]

with open("/user_data/TabGNN/results/log.txt", "w", encoding="utf-8"): pass
def write_log(text):
    with open("/user_data/TabGNN/results/log.txt", "a", encoding="utf-8") as f:
        f.write(text+"\n"+"="*50+"\n")

if not os.path.exists("/user_data/TabGNN/results/progress.json"):
    with open("/user_data/TabGNN/results/progress.json", "w", encoding="utf-8") as f: 
        json.dump({}, f)

with open("/user_data/TabGNN/results/progress.json", "r", encoding="utf-8") as f:
    progress = json.load(f)

if progress.get(SOURCE) is None:
    progress[SOURCE] = {FILENAME:0}
    with open("/user_data/TabGNN/results/progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)
elif progress[SOURCE].get(FILENAME) is None:
    progress[SOURCE][FILENAME] = 0
    with open("/user_data/TabGNN/results/progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)

def save_queries():
    with open(QUERY_PATH, "w", encoding="utf-8") as f:
        for line in query_lines:
            f.write(json.dumps(line, ensure_ascii=False)+"\n")

if MODEL == "Ollama":
    agent = oAgent(api_key=api_key)
# elif MODEL == "Gemini":
#     agent = gAgent(api_key=api_key)

PROMPT = """\
You are an expert in table data analysis.
Given a table with its file name, sheet name, and a portion of its content (first ten rows), your task is to **extract key headers and generate questions** based on the table & headers.

Important Considerations:
• The table may contain nan or Unnamed: values, which represent empty merged cells in the original table. These **should not** be considered as meaningful data points or headers.
• The **true column headers may not always be in the first row or first column**. Carefully analyze the table to identify the correct headers.
• If the table has **multi-level headers**, preserve the hierarchical structure without merging or altering the text.
• If the table has an **irregular header structure** (such as key-value formatted headers where column names are listed separately), extract the correct header names accordingly.
• **Ignore rows that contain mostly empty values (nan, Unnamed:) or placeholders without meaningful data.**
• **Do not generate python code, extract headers and questions on your own.**
• The type of Questions could be one of (lookup, calculate, visualize, reasoning).
• **Generate question using the language of the table.**

Tasks:
1. Extract Header Names:
• Identify the **true headers** by analyzing the structure of the table.
• **Exclude** placeholder values like "nan" and "Unnamed:".
• If the table contains **multi-level headers**, keep them as separate levels without merging.
• If the table has **key-value headers**, extract the correct column names.

2. Generate Questions (Context-Specific to the Table):
• Formulate **questions that can only be answered using this specific table**.
• Ensure **each question involves 1 to 3 different headers** to capture interactions between data & columns.
• Ensure the header diversity in all the questions.
• Use ” to mark the headers in the question.
• **Total number of questions should larger than the half number of extracted headers**

Avoid vague or biographical questions. Use only values and headers from the preview.

**Output Format (Strictly JSON format)**
Only return a single valid JSON object without any other text:
{{ "headers": ["header1", "header2", "..."], "questions": ["question1", "question2", "..."] }}

**Table Meta**
- File: {file_name}
- Sheet: {sheet_name}

**Table Preview**
{table_csv_preview}
"""



def main():
    start = 0 if FIX else progress[SOURCE][FILENAME]
    for i in range(start, len(table_lines)):
        table:dict = table_lines[i]
        if FIX:
            if query_lines[i].pop("error", None) is None: continue
            else:
                print(f"Fixing table index {i+1} (ID:{table['id']})")


        file_name = table["file_name"]
        sheet_name = table["sheet_name"]
        table_id = table["id"]

        now = datetime.datetime.now()+datetime.timedelta(hours=8)
        print(now)
        print(f"Progress: {i+1}/{len(table_lines)}")
        write_log(f"Progress: {i+1}/{len(table_lines)}")
        print(f"Generating query for table: (ID:{table_id}) {sheet_name}")
        write_log(f"Generating query for table: (ID:{table_id}) {sheet_name}")

        base_prompt = PROMPT.format(
            file_name=file_name,
            sheet_name=sheet_name,
            table_csv_preview=json2csv(table, top=10),
        )

        # 查詢 LLM + 重試
        attempt = 0
        final_obj = None
        while attempt <= MAX_RETRIES:
            prompt = base_prompt if attempt == 0 else (
                base_prompt + "\n\nRespond with **only** a valid JSON object matching the required schema."
            )

            write_log(prompt)
            response = agent.query(prompt)
            write_log(response)
            # response = response.replace("\\\"", "")
            response = response.replace("```json", "").replace("```", "").strip()
            print(response)

            try:
                obj = parse_json_strict(response)
                ok, msg = is_valid_output(obj)
                if ok:
                    final_obj = obj
                    break
                else:
                    write_log(f"[VALIDATION FAIL] table_id={table_id}: {msg}")
            except Exception as e:
                write_log(f"[JSON PARSE ERROR] table_id={table_id}: {e}")

            attempt += 1

        if final_obj is None:
            # 仍失敗，寫入一筆帶錯誤訊息的紀錄（避免卡住）
            result = {
                "table_id": table_id,
                "sheet_name": sheet_name,
                "headers": [],
                "questions": [],
                "error": "LLM output invalid after retries"
            }
        else:
            # 正常寫入結果
            result = {
                "table_id": table_id,
                "sheet_name": sheet_name,
                "headers": final_obj["headers"],
                "questions": final_obj["questions"]
            }

        while i >= len(query_lines): query_lines.append({})
        query_lines[i] = result
        
        save_queries()

        if not FIX:
            progress[SOURCE][FILENAME] = i+1
            with open("/user_data/TabGNN/results/progress.json", "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=4)

        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()