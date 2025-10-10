import pandas as pd
from GeminiAgent import Agent
import csv
import json
import os

table_path = r"table\train\feta\table.jsonl"
query_path = r"data\train\feta\generate_query.jsonl"
max_query_counts = 5

os.makedirs(os.path.dirname(query_path), exist_ok=True)

def json2md(jtable, top = 10):

    header = next(csv.reader(jtable["header"]))
    reader = csv.reader(jtable["instances"])
    instans = [next(reader) for i in range(len(jtable["instances"]))]

    jtable = pd.DataFrame(data=instans, columns=header)

    top_n = jtable.head(top)
    return top_n.to_markdown(index=False)

with open(table_path, "r", encoding="utf-8") as f:
    table_lines = [json.loads(line) for line in f.readlines()]

def save_queries():
    with open(query_path, "w", encoding="utf-8") as f:
        for line in query_lines:
            f.write(json.dumps(line, ensure_ascii=False)+"\n")

if not os.path.exists(query_path):
    with open(query_path, "w", encoding="utf-8"): pass

with open(query_path, "r", encoding="utf-8") as f:
    query_lines = [json.loads(line) for line in f.readlines()]

with open("api_keys.json", "r", encoding="utf-8") as f:
    api_keys = json.load(f)

agent = Agent(api_keys=api_keys)

PROMPT = """\
You are a data retrieval assistant. I will give you:
1) A partial table preview in Markdown (first rows only).
2) Existing queries already generated for this table.

Your task:
- Propose exactly ONE NEW user query in English.
- It must be answerable using this table (or the same dataset family) without inventing columns that don’t exist.
- Make it semantically different from all existing queries (change focus, time window, entity, aggregation, comparison, or constraint).
- Keep it natural and concise (≤ 25 words). No explanations.

### Table Name
{table_name}

### Table Preview (Markdown)
{table_markdown_preview}

### Existing Queries
{existing_queries_markdown_list}

### Additional rules
- Use column names/entities consistent with the preview.
- Prefer diversity: different filters, grouping, aggregation, timeframe, superlative, or comparison.
- Do NOT repeat any existing query intent.
- Do NOT ask multi-hop questions needing external knowledge beyond this dataset family.
- Do NOT output any other text or explanations.
"""

def get_existing_queries(line):
    if line >= len(query_lines):
        return "- None"
     
    existing_queries = []
    for q in query_lines[line]["queries"]:
        existing_queries.append(q)
    if len(existing_queries) == 0:
        return "- None"
    else:
        return "\n".join([f"- {q}" for q in existing_queries])

def write_log(text):
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(text+"\n"+"="*50+"\n")

def main():
    with open("progress.json", "r", encoding="utf-8") as f:
        progress = json.load(f)["progress"]
    

    for i in range(progress+1, len(table_lines)):
        table = table_lines[i]
        table_name = table["sheet_name"]
        table_id = table["id"]

        print(f"Progress: {i+1}/{len(table_lines)}")
        print(f"Generating query for table: (ID:{table_id}) {table_name}")

        while True:
            prompt = PROMPT.format(
                table_name=table_name,
                table_markdown_preview=json2md(table, top=10),
                existing_queries_markdown_list=get_existing_queries(i)
            )

            write_log(prompt)

            response = agent.query(prompt)
            print(f"- ", response)

            write_log(response)

            if i < len(query_lines):
                if response not in query_lines[i]["queries"]:
                    query_lines[i]["queries"].append(response)
                    if len(query_lines[i]["queries"]) > max_query_counts: break
            else:
                query_lines.append({
                    "table_id": table_id,
                    "table_name": table_name,
                    "queries": [response]
                })
        
        save_queries()
        with open("progress.json", "w", encoding="utf-8") as f:
            json.dump({"progress": i}, f, ensure_ascii=False)

        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()