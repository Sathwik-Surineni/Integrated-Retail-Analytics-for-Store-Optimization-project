# inspect_rules.py
import pandas as pd
from pathlib import Path
p = Path("models/association_rules_sparse.csv")
if not p.exists():
    print("File not found:", p)
else:
    df = pd.read_csv(p)
    print("Total rules:", len(df))
    # show top 10 by lift
    display_cols = ["dept_A","dept_B","count_cooccurrence","support","confidence_A_to_B","confidence_B_to_A","lift"]
    print(df.sort_values(["lift","support"], ascending=[False,False]).head(10)[display_cols].to_string(index=False))
