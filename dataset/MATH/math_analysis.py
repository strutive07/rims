import pandas as pd
import jsonlines as jsl
from pprint import pprint

# count the number of examples in each type 
df = pd.DataFrame(jsl.open("MATH-full.jsonl"))
counts = df.type.value_counts()
pprint(counts)

"""
type
Algebra                   1187
Intermediate Algebra       903
Prealgebra                 871
Precalculus                546
Number Theory              540
Geometry                   479
Counting & Probability     474

total = 5000
"""