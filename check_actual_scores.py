import json
import os
from pathlib import Path

res_dir = Path("evaluation/results")
MODELS = ['gemini-2.5-flash', 'deepseek-chat', 'Qwen2.5-72B-Instruct']
BASELINES = ['flat_text', 'vector_rag', 'graph_aug', 'tool_use', 'react_code']

# Tier 1 datasets for membership
t1_ds = ['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm', 'syn_logistics']

print("=== MEMBERSHIP (Tier 1) Accuracy ===")
for bl in BASELINES:
    for model in MODELS:
        correct = 0
        total = 0
        for ds in t1_ds:
            fpath = res_dir / f"{bl}_original_{ds}_{model}.json"
            if fpath.exists():
                data = json.load(open(fpath))
                for r in data['results']:
                    if r.get('subtype') == 'membership':
                        total += 1
                        if r['scores']['exact_match']:
                            correct += 1
        if total > 0:
            print(f"{bl} | {model} | {correct}/{total} = {correct/total*100:.1f}%")

print("\n=== CASCADE_COUNT (Tier 2) Accuracy ===")
t2_ds = ['adventureworks', 'syn_logistics']
t2_bl = [b + "_v2" for b in BASELINES]
for bl in t2_bl:
    for model in MODELS:
        correct = 0
        total = 0
        for ds in t2_ds:
            fpath = res_dir / f"{bl}_tier2_{ds}_{model}.json"
            if fpath.exists():
                data = json.load(open(fpath))
                for r in data['results']:
                    if r.get('subtype') == 'cascade_count':
                        total += 1
                        if r['scores']['exact_match']:
                            correct += 1
        if total > 0:
            print(f"{bl} | {model} | {correct}/{total} = {correct/total*100:.1f}%")
