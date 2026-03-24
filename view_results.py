"""Comprehensive Gemini analysis for paper."""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
from collections import Counter, defaultdict

results_dir = Path("evaluation/results")

# Load ALL Gemini original results
all_data = {}
all_results = []
for bl in ["flat_text", "vector_rag", "graph_aug"]:
    for ds in ["adventureworks", "tpc-ds", "tpc-di", "omop_cdm"]:
        f = results_dir / f"{bl}_original_{ds}_gemini-2.5-flash.json"
        if f.exists() and f.stat().st_size > 100:
            d = json.load(open(f, encoding='utf-8'))
            if 'results' in d:
                all_data[(bl, ds)] = d
                for r in d['results']:
                    r['_baseline'] = bl
                    r['_dataset'] = ds
                    all_results.append(r)

# ======================================================================
# TABLE 1: Overall results (for the paper)
# ======================================================================
print("=" * 80)
print("TABLE 1: OVERALL RESULTS BY BASELINE × DATASET")
print("=" * 80)
print(f"{'Baseline':<12} {'AW':>9} {'TPC-DS':>9} {'TPC-DI':>9} {'OMOP':>9} {'AVG':>9}")
print("-" * 60)
for bl in ["flat_text", "vector_rag", "graph_aug"]:
    vals = []
    for ds in ["adventureworks", "tpc-ds", "tpc-di", "omop_cdm"]:
        if (bl, ds) in all_data:
            em = all_data[(bl, ds)]["metrics"]["exact_match"] * 100
            vals.append(em)
        else:
            vals.append(0)
    avg = sum(vals) / len(vals)
    short = {"flat_text": "Flat Text", "vector_rag": "Vector RAG", "graph_aug": "Graph-Aug"}
    print(f"{short[bl]:<12} {vals[0]:>8.1f}% {vals[1]:>8.1f}% {vals[2]:>8.1f}% {vals[3]:>8.1f}% {avg:>8.1f}%")

# F1
print(f"\n{'Baseline':<12} {'AW':>9} {'TPC-DS':>9} {'TPC-DI':>9} {'OMOP':>9} {'AVG':>9}")
print("-" * 60)
for bl in ["flat_text", "vector_rag", "graph_aug"]:
    vals = []
    for ds in ["adventureworks", "tpc-ds", "tpc-di", "omop_cdm"]:
        if (bl, ds) in all_data:
            f1 = all_data[(bl, ds)]["metrics"]["avg_f1"] * 100
            vals.append(f1)
        else:
            vals.append(0)
    avg = sum(vals) / len(vals)
    short = {"flat_text": "Flat Text", "vector_rag": "Vector RAG", "graph_aug": "Graph-Aug"}
    print(f"{short[bl]:<12} {vals[0]:>8.1f}% {vals[1]:>8.1f}% {vals[2]:>8.1f}% {vals[3]:>8.1f}% {avg:>8.1f}%")

# ======================================================================
# TABLE 2: By difficulty
# ======================================================================
print(f"\n{'=' * 80}")
print("TABLE 2: RESULTS BY DIFFICULTY")
print("=" * 80)
for diff in ["easy", "medium", "hard"]:
    print(f"\n  {diff.upper()}:")
    print(f"  {'Baseline':<12} {'EM':>8} {'F1':>8} {'n':>5}")
    print(f"  {'-' * 35}")
    for bl in ["flat_text", "vector_rag", "graph_aug"]:
        correct, total, f1_sum = 0, 0, 0
        for ds in ["adventureworks", "tpc-ds", "tpc-di", "omop_cdm"]:
            if (bl, ds) in all_data:
                for r in all_data[(bl, ds)]["results"]:
                    if r["difficulty"] == diff:
                        total += 1
                        correct += int(r["scores"]["exact_match"])
                        f1_sum += r["scores"]["f1"]
        if total:
            short = {"flat_text": "FT", "vector_rag": "VR", "graph_aug": "GA"}
            print(f"  {short[bl]:<12} {correct/total*100:>7.1f}% {f1_sum/total*100:>7.1f}% {total:>5}")

# ======================================================================
# TABLE 3: By subtype (the key table)
# ======================================================================
print(f"\n{'=' * 80}")
print("TABLE 3: RESULTS BY SUBTYPE (ALL DATASETS)")
print("=" * 80)
subtypes = sorted(set(r["subtype"] for r in all_results))
print(f"{'Subtype':<22} {'Diff':<8} {'n':>4} {'FT EM':>8} {'VR EM':>8} {'GA EM':>8} {'FT F1':>8} {'VR F1':>8} {'GA F1':>8} {'Best':>5}")
print("-" * 100)
for st in subtypes:
    diff = ""
    n = 0
    vals_em = {}
    vals_f1 = {}
    for bl in ["flat_text", "vector_rag", "graph_aug"]:
        correct, total, f1_sum = 0, 0, 0
        for r in all_results:
            if r["subtype"] == st and r["_baseline"] == bl:
                total += 1
                correct += int(r["scores"]["exact_match"])
                f1_sum += r["scores"]["f1"]
                diff = r["difficulty"]
        if total:
            vals_em[bl] = correct / total * 100
            vals_f1[bl] = f1_sum / total * 100
            n = total
    best = max(vals_em, key=vals_em.get) if vals_em else "?"
    short = {"flat_text": "FT", "vector_rag": "VR", "graph_aug": "GA"}
    print(f"{st:<22} {diff:<8} {n:>4} "
          f"{vals_em.get('flat_text',0):>7.1f}% {vals_em.get('vector_rag',0):>7.1f}% {vals_em.get('graph_aug',0):>7.1f}% "
          f"{vals_f1.get('flat_text',0):>7.1f}% {vals_f1.get('vector_rag',0):>7.1f}% {vals_f1.get('graph_aug',0):>7.1f}% "
          f"{short.get(best, '?'):>5}")

# ======================================================================
# TABLE 4: Error analysis — where does each baseline fail?
# ======================================================================
print(f"\n{'=' * 80}")
print("TABLE 4: ERROR ANALYSIS")
print("=" * 80)
for bl in ["flat_text", "vector_rag", "graph_aug"]:
    short = {"flat_text": "Flat Text", "vector_rag": "Vector RAG", "graph_aug": "Graph-Aug"}
    wrong = [r for r in all_results if r["_baseline"] == bl and not r["scores"]["exact_match"]]
    total = sum(1 for r in all_results if r["_baseline"] == bl)
    print(f"\n  {short[bl]} ({len(wrong)}/{total} wrong):")
    by_subtype = Counter(r["subtype"] for r in wrong)
    for st, cnt in by_subtype.most_common():
        st_total = sum(1 for r in all_results if r["_baseline"] == bl and r["subtype"] == st)
        pct = cnt / st_total * 100
        print(f"    {st:<22}: {cnt:>3}/{st_total} wrong ({pct:.0f}%)")

# ======================================================================
# TABLE 5: Where GA beats FT (the key insight)
# ======================================================================
print(f"\n{'=' * 80}")
print("TABLE 5: WHERE GRAPH-AUG BEATS FLAT TEXT")
print("=" * 80)
print(f"{'Subtype':<22} {'Diff':<8} {'FT EM':>8} {'GA EM':>8} {'Delta':>8} {'Verdict'}")
print("-" * 65)
for st in subtypes:
    ft_correct, ft_total = 0, 0
    ga_correct, ga_total = 0, 0
    diff = ""
    for r in all_results:
        if r["subtype"] == st:
            diff = r["difficulty"]
            if r["_baseline"] == "flat_text":
                ft_total += 1
                ft_correct += int(r["scores"]["exact_match"])
            elif r["_baseline"] == "graph_aug":
                ga_total += 1
                ga_correct += int(r["scores"]["exact_match"])
    if ft_total and ga_total:
        ft_em = ft_correct / ft_total * 100
        ga_em = ga_correct / ga_total * 100
        delta = ga_em - ft_em
        verdict = "GA++" if delta > 5 else ("FT++" if delta < -5 else "~TIE")
        print(f"{st:<22} {diff:<8} {ft_em:>7.1f}% {ga_em:>7.1f}% {delta:>+7.1f}% {verdict}")

# ======================================================================
# KEY STATISTICS for the Abstract
# ======================================================================
print(f"\n{'=' * 80}")
print("KEY STATISTICS FOR THE PAPER ABSTRACT")
print("=" * 80)
for bl in ["flat_text", "vector_rag", "graph_aug"]:
    bl_results = [r for r in all_results if r["_baseline"] == bl]
    em = sum(int(r["scores"]["exact_match"]) for r in bl_results) / len(bl_results) * 100
    f1 = sum(r["scores"]["f1"] for r in bl_results) / len(bl_results) * 100
    hard = [r for r in bl_results if r["difficulty"] == "hard"]
    hard_em = sum(int(r["scores"]["exact_match"]) for r in hard) / len(hard) * 100 if hard else 0
    short = {"flat_text": "FT", "vector_rag": "VR", "graph_aug": "GA"}
    print(f"  {short[bl]}: EM={em:.1f}%, F1={f1:.1f}%, Hard EM={hard_em:.1f}%")

print(f"\n  Total Qs per baseline: {len([r for r in all_results if r['_baseline']=='flat_text'])}")
print(f"  Datasets: 4")
print(f"  Subtypes: {len(subtypes)}")
