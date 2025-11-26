import json
import pandas as pd
import csv

SOURCE = "test/feta"
TABLE_PATH = "/user_data/TabGNN/data/table/test/feta/table.jsonl"
i = 10

with open(TABLE_PATH, "r", encoding="utf-8") as f:
    table_lines = [json.loads(line) for line in f.readlines()]

def json2csv(jtable, top = 10):

    header = next(csv.reader(jtable["header"]))
    reader = csv.reader(jtable["instances"])
    instans = [next(reader) for i in range(len(jtable["instances"]))]

    jtable = pd.DataFrame(data=instans, columns=header)

    top_n = jtable.head(top)
    return top_n.to_csv(index=False)



table:dict = table_lines[i]

print(json2csv(table, top=10))