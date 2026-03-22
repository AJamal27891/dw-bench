#!/usr/bin/env python3
"""DW-Bench: Comprehensive integrity check of ALL result files.

Linux/macOS version with rich Unicode output.

Checks:
1. File exists and is valid JSON
2. Has required keys (results, metrics)
3. Each result has required fields (id, question, gold_answer, predicted_answer, scores)
4. Question counts match expected N per dataset
5. Scores match recomputed scores (spot check)
6. No duplicate question IDs
7. Cross-baseline question count consistency
8. Table 3 micro-EM cross-check
9. Table 5 obfuscation cross-check

Usage:
    python integrity_check_full.py
    python integrity_check_full.py --model gemini-2.5-flash
    python integrity_check_full.py --verbose
"""
import json
import sys
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, 'evaluation')
from metrics import score_answer

results_dir = Path('evaluation/results')

MODELS = ['gemini-2.5-flash', 'deepseek-chat', 'Qwen2.5-72B-Instruct']
MODEL_SHORT = {'gemini-2.5-flash': 'Gem', 'deepseek-chat': 'DS',
               'Qwen2.5-72B-Instruct': 'Qw'}
BASELINES = ['flat_text', 'vector_rag', 'graph_aug', 'tool_use',
             'react_code', 'oracle']
BL_LABELS = {'flat_text': 'FT', 'vector_rag': 'VR', 'graph_aug': 'GA',
             'tool_use': 'TU', 'react_code': 'RC', 'oracle': 'Or'}
DATASETS = ['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm', 'syn_logistics']
DS_EXPECTED = {'adventureworks': 208, 'tpc-ds': 127, 'tpc-di': 181,
               'omop_cdm': 158, 'syn_logistics': 372}

parser = argparse.ArgumentParser(
    description='DW-Bench: Validate result file integrity (rich output)')
parser.add_argument('--model', choices=MODELS, default=None,
                    help='Check a single model (default: all)')
parser.add_argument('--verbose', '-v', action='store_true',
                    help='Show per-file details')
args = parser.parse_args()

check_models = [args.model] if args.model else MODELS
errors = []
warnings = []
sep = '═' * 70

print(f'╔{sep}╗')
print(f'║  DW-BENCH INTEGRITY CHECK (Full)')
print(f'╚{sep}╝')

# ── 1. Check each file ──────────────────────────────────────────────
file_summary = {}
for model in check_models:
    ms = MODEL_SHORT.get(model, model[:8])
    for suffix in ['original', 'obfuscated']:
        for bl in BASELINES:
            for ds in DATASETS:
                if bl == 'oracle' and suffix == 'obfuscated':
                    continue
                if ds == 'syn_logistics' and suffix == 'obfuscated':
                    continue

                fname = f'{bl}_{suffix}_{ds}_{model}.json'
                fpath = results_dir / fname

                if not fpath.exists():
                    if suffix == 'original':
                        errors.append(f'🚫 MISSING: {fname}')
                    continue

                try:
                    data = json.load(open(fpath, encoding='utf-8'))
                except json.JSONDecodeError as e:
                    errors.append(f'💥 CORRUPT JSON: {fname}: {e}')
                    continue

                if 'results' not in data:
                    errors.append(f'❌ NO RESULTS KEY: {fname}')
                    continue

                results = data['results']
                n = len(results)
                expected = DS_EXPECTED[ds]

                if n != expected:
                    warnings.append(
                        f'⚠️  {fname}: {n} Qs (expected {expected})')

                # Required fields
                required = ['id', 'question', 'gold_answer',
                            'predicted_answer', 'scores', 'subtype',
                            'difficulty']
                for i, r in enumerate(results):
                    for field in required:
                        if field not in r:
                            errors.append(
                                f"❌ {fname}[{i}]: missing '{field}'")

                # Duplicate IDs
                ids = [r['id'] for r in results]
                dupes = [id_ for id_, cnt in Counter(ids).items()
                         if cnt > 1]
                if dupes:
                    errors.append(f'❌ {fname}: duplicate IDs: {dupes}')

                # Score spot check (first 10)
                score_mismatches = 0
                for r in results[:10]:
                    if (r.get('answer_type') == 'ordered_list'
                            and r['scores'].get('valid_alt_path', False)):
                        continue
                    recomputed = score_answer(
                        r['predicted_answer'], r['gold_answer'],
                        r.get('answer_type', 'list'))
                    if recomputed['exact_match'] != \
                            r['scores']['exact_match']:
                        score_mismatches += 1
                if score_mismatches > 0:
                    errors.append(
                        f'❌ {fname}: {score_mismatches}/10 score '
                        f'mismatches')

                em = data['metrics']['exact_match'] * 100
                f1 = data['metrics']['avg_f1'] * 100
                file_summary[(suffix, bl, ds, model)] = {
                    'n': n, 'em': em, 'f1': f1}

                if args.verbose:
                    status = '✅' if score_mismatches == 0 else '❌'
                    print(f'  {status} {fname}: {n} Qs, '
                          f'EM={em:.1f}%')

# ── 2. Cross-baseline consistency ────────────────────────────────────
print(f'\n{"─" * 70}')
print('📊 CROSS-BASELINE CONSISTENCY (original)')
print(f'{"─" * 70}')
for model in check_models:
    ms = MODEL_SHORT.get(model, model[:8])
    for ds in DATASETS:
        counts = {}
        for bl in BASELINES:
            key = ('original', bl, ds, model)
            if key in file_summary:
                counts[bl] = file_summary[key]['n']
        unique = set(counts.values())
        if len(unique) > 1:
            errors.append(f'{ms}/{ds}: count mismatch: {counts}')
            print(f'  ❌ {ms}/{ds}: {counts}')
        elif counts:
            print(f'  ✅ {ms}/{ds}: all baselines = '
                  f'{list(counts.values())[0]} Qs')

# ── 3. Table 3 micro-EM ─────────────────────────────────────────────
print(f'\n{"─" * 70}')
print('📈 TABLE 3: Micro-EM (original, 5 datasets, N=1046)')
print(f'{"─" * 70}')
for model in check_models:
    ms = MODEL_SHORT.get(model, model[:8])
    parts = []
    for bl in BASELINES:
        correct = 0
        total = 0
        for ds in DATASETS:
            fname = f'{bl}_original_{ds}_{model}.json'
            fpath = results_dir / fname
            if fpath.exists():
                data = json.load(open(fpath, encoding='utf-8'))
                for r in data['results']:
                    total += 1
                    if r['scores']['exact_match']:
                        correct += 1
        if total > 0:
            em = round(correct / total * 100, 1)
            parts.append(f'{BL_LABELS[bl]}={em}%')
    print(f'  {ms}: {", ".join(parts)}')

# ── 4. Table 5 obfuscation ──────────────────────────────────────────
print(f'\n{"─" * 70}')
print('🔒 TABLE 5: Obfuscation (4 datasets excl. Syn-Logistics)')
print(f'{"─" * 70}')
obf_ds = ['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm']
for model in check_models:
    ms = MODEL_SHORT.get(model, model[:8])
    print(f'  {ms}:')
    for bl in ['flat_text', 'vector_rag', 'graph_aug', 'tool_use',
               'react_code']:
        vals = {}
        for cond in ['original', 'obfuscated']:
            correct = 0
            total = 0
            for ds in obf_ds:
                fname = f'{bl}_{cond}_{ds}_{model}.json'
                fpath = results_dir / fname
                if fpath.exists():
                    data = json.load(open(fpath, encoding='utf-8'))
                    for r in data['results']:
                        total += 1
                        if r['scores']['exact_match']:
                            correct += 1
            if total > 0:
                vals[cond] = round(correct / total * 100, 1)
        if 'original' in vals and 'obfuscated' in vals:
            delta = round(vals['obfuscated'] - vals['original'], 1)
            sign = '+' if delta >= 0 else ''
            print(f'    {BL_LABELS.get(bl, bl):3s}: '
                  f'orig={vals["original"]}%  '
                  f'obf={vals["obfuscated"]}%  '
                  f'Δ={sign}{delta}pp')

# ── 5. Summary ───────────────────────────────────────────────────────
print(f'\n{"═" * 70}')
print('📋 SUMMARY')
print(f'{"═" * 70}')

if errors:
    print(f'\n❌ {len(errors)} ERRORS:')
    for e in errors:
        print(f'  {e}')
else:
    print('\n✅ No errors found!')

if warnings:
    print(f'\n⚠️  {len(warnings)} WARNINGS:')
    for w in warnings:
        print(f'  {w}')
else:
    print('✅ No warnings!')

total_files = len(file_summary)
print(f'\n📁 Total files checked: {total_files}')
print(f'\n💡 Run "python view_results.py" for detailed results tables.')
