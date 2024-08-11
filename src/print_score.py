import io
import os
import glob

import pandas as pd
from tabulate import tabulate

def load_scores():
    files = glob.glob("outputs/*/*/processed_indiv_scored.md")

    df_dict = {}
    
    for file in files:
        dataset_name, model_name, df = read_md(file)
        if dataset_name not in df_dict:
            df_dict[dataset_name] = {}
            
        df_dict[dataset_name][model_name] = df

    for dataset_name in df_dict:
        df_dict[dataset_name] = pd.concat(df_dict[dataset_name].values())
    return df_dict


def load_sg(dataset_name, model_name):
    fname = f'outputs/{dataset_name}/{model_name}/simple_greedy/processed_sg_scored.txt'
    if os.path.exists(fname):
        with open(fname) as f:
            return f.read()
    else:
        return ''

def load_rims(dataset_name, model_name):
    fname = f'outputs/{dataset_name}/{model_name}/rims/*/processed_rims_scored.txt'

    fnames = glob.glob(fname)
    res = ''
    
    for fname in fnames:
        with open(fname) as f:
            res += f"""{fname}

{f.read()}
"""
    return res

def read_md(fname):
    with open(fname) as f:
        markdown_table = f.read()
    df = pd.read_table(io.StringIO(markdown_table), sep='|', header=0, index_col=0)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.reset_index().drop(columns=['index', '']).drop(0)

    dataset_name = fname.split('/')[1]
    model_name = fname.split('/')[2]
    
    df['model_name'] = [model_name]
    df['simple greedy'] = [str(load_sg(dataset_name, model_name))]
    df['rims'] = [str(load_rims(dataset_name, model_name))]
    df = df[['model_name', 'cot', 'pal', 'p2c', 'simple greedy', 'rims']]

    return dataset_name, model_name, df


dfs = load_scores()
for dataset_name in dfs:
    with open(f'scores/{dataset_name}.md', 'w') as f:
        f.write(tabulate(dfs[dataset_name]))