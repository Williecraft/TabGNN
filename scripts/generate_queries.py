import os, io, json, csv

def load_jsonl(p):
    with io.open(p, 'r', encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if l:
                yield json.loads(l)

def parse_csv_line(s):
    return next(csv.reader([s]))

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(base)
    table_path = os.path.join(root, 'table', 'train', 'feta', 'table.jsonl')
    query_path = os.path.join(root, 'table', 'train', 'feta', 'query.jsonl')
    out_path = os.path.join(root, 'data', 'train', 'feta', 'generate_query.jsonl')

    # Build mapping: (file_name, sheet_name) -> table_id
    mapping = {}
    count_tables = 0
    for o in load_jsonl(table_path):
        tid = o.get('id')
        fn = o.get('file_name')
        sh = o.get('sheet_name')
        if tid is None or fn is None or sh is None:
            # try metadata fallback
            md = o.get('metadata', {})
            fn = md.get('table_source_json', fn)
            sh = o.get('sheet_name', sh)
        if fn is not None and sh is not None:
            mapping[(str(fn), str(sh))] = tid
            count_tables += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = io.open(out_path, 'w', encoding='utf-8')

    written = 0
    missed = 0
    for q in load_jsonl(query_path):
        question = q.get('question')
        gts = q.get('ground_truth_list', [])
        # support either id present or map by file_name/sheet_name
        for gt in gts:
            tid = gt.get('id')
            if tid is None:
                fn = gt.get('file_name') or (q.get('metadata', {}).get('table_source_json'))
                sh = gt.get('sheet_name')
                tid = mapping.get((str(fn), str(sh)))
            if tid is None:
                missed += 1
                continue
            rec = {"table_id": tid, "queries": [question]}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    out.close()
    print(f"Tables indexed: {count_tables}")
    print(f"Queries written: {written}")
    print(f"Ground truths missed: {missed}")
    print(f"Output: {out_path}")

if __name__ == '__main__':
    main()