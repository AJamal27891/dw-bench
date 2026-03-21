"""DW-Bench Evaluation Harness.

Runs baselines on Q&A pairs and computes metrics.

Usage:
    python evaluate.py --baseline flat_text --dataset adventureworks
    python evaluate.py --baseline flat_text --dataset all --obfuscated
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics import score_answer


def compute_aggregate_metrics(results: list) -> dict:
    """Compute aggregate metrics across all results.

    Returns both micro-averaged (overall) and macro-averaged (mean of
    per-subtype averages) metrics. Macro-averaging weights all 13 subtypes
    equally, preventing high-N subtypes like join_path from dominating.
    """
    if not results:
        return {}

    total = len(results)
    exact_matches = sum(1 for r in results if r['scores']['exact_match'])
    avg_f1 = sum(r['scores']['f1'] for r in results) / total

    # By type
    by_type = {}
    for r in results:
        t = r['type']
        if t not in by_type:
            by_type[t] = {'total': 0, 'exact': 0, 'f1_sum': 0}
        by_type[t]['total'] += 1
        by_type[t]['exact'] += int(r['scores']['exact_match'])
        by_type[t]['f1_sum'] += r['scores']['f1']

    # By difficulty
    by_diff = {}
    for r in results:
        d = r['difficulty']
        if d not in by_diff:
            by_diff[d] = {'total': 0, 'exact': 0, 'f1_sum': 0}
        by_diff[d]['total'] += 1
        by_diff[d]['exact'] += int(r['scores']['exact_match'])
        by_diff[d]['f1_sum'] += r['scores']['f1']

    # By subtype
    by_subtype = {}
    for r in results:
        st = r['subtype']
        if st not in by_subtype:
            by_subtype[st] = {'total': 0, 'exact': 0, 'f1_sum': 0}
        by_subtype[st]['total'] += 1
        by_subtype[st]['exact'] += int(r['scores']['exact_match'])
        by_subtype[st]['f1_sum'] += r['scores']['f1']

    def summarize(d):
        return {k: {
            'total': v['total'],
            'exact_match': round(v['exact'] / v['total'], 4),
            'avg_f1': round(v['f1_sum'] / v['total'], 4),
        } for k, v in d.items()}

    subtype_summary = summarize(by_subtype)

    # Macro-average: mean of per-subtype EM and F1
    subtype_ems = [v['exact_match'] for v in subtype_summary.values()]
    subtype_f1s = [v['avg_f1'] for v in subtype_summary.values()]
    macro_em = round(sum(subtype_ems) / len(subtype_ems), 4) if subtype_ems else 0
    macro_f1 = round(sum(subtype_f1s) / len(subtype_f1s), 4) if subtype_f1s else 0

    return {
        'total': total,
        # Micro (overall)
        'micro_em': round(exact_matches / total, 4),
        'micro_f1': round(avg_f1, 4),
        # Macro (subtype-averaged)
        'macro_em': macro_em,
        'macro_f1': macro_f1,
        # Legacy aliases
        'exact_match': round(exact_matches / total, 4),
        'avg_f1': round(avg_f1, 4),
        # Breakdowns
        'by_type': summarize(by_type),
        'by_difficulty': summarize(by_diff),
        'by_subtype': subtype_summary,
    }


def run_evaluation(baseline_name: str, dataset: str, obfuscated: bool,
                   api_key: str, api_base: str, model: str,
                   extended: bool = False) -> dict:
    """Run a single evaluation: baseline × dataset × obfuscated flag."""
    repo_root = Path(__file__).parent.parent
    ds_dir = repo_root / 'datasets' / dataset

    suffix = '_obfuscated' if obfuscated else '_original'
    if extended:
        suffix = '_extended'
    print(f"\n{'='*60}")
    print(f"Running: {baseline_name}{suffix} on {dataset}")
    print(f"{'='*60}")

    # Determine QA file
    qa_file = None
    if extended:
        qa_file = 'qa_pairs_extended.json'
    elif obfuscated:
        qa_file = 'qa_pairs_obfuscated.json'

    # Load baseline
    if baseline_name == 'flat_text':
        from baselines.flat_text import run_flat_text
        raw_results = run_flat_text(ds_dir, api_key,
                                    api_base=api_base, model=model,
                                    obfuscated=obfuscated,
                                    qa_file=qa_file)
    elif baseline_name == 'vector_rag':
        from baselines.vector_rag import run_vector_rag
        raw_results = run_vector_rag(ds_dir, api_key,
                                     api_base=api_base, model=model,
                                     obfuscated=obfuscated,
                                     qa_file=qa_file)
    elif baseline_name in ('graph_aug', 'gnn_llm'):
        from baselines.graph_aug import run_gnn_llm
        raw_results = run_gnn_llm(ds_dir, api_key,
                                  api_base=api_base, model=model,
                                  obfuscated=obfuscated,
                                  qa_file=qa_file)
    elif baseline_name == 'oracle':
        from baselines.oracle import run_oracle
        raw_results = run_oracle(ds_dir, api_key,
                                 api_base=api_base, model=model,
                                 obfuscated=obfuscated,
                                 qa_file=qa_file)
    elif baseline_name == 'tool_use':
        from baselines.tool_use import run_tool_use
        raw_results = run_tool_use(ds_dir, api_key,
                                   api_base=api_base, model=model,
                                   obfuscated=obfuscated,
                                   qa_file=qa_file)
    elif baseline_name == 'react_code':
        from baselines.react_code import run_react_code
        raw_results = run_react_code(ds_dir, api_key,
                                     api_base=api_base, model=model,
                                     obfuscated=obfuscated,
                                     qa_file=qa_file)
    elif baseline_name == 'flat_text_v2':
        from baselines.flat_text import run_flat_text
        if qa_file is None:
            qa_file = 'qa_pairs_tier2.json'
        raw_results = run_flat_text(ds_dir, api_key,
                                    api_base=api_base, model=model,
                                    obfuscated=False,
                                    qa_file=qa_file)
    elif baseline_name == 'graph_aug_v2':
        from baselines.graph_aug_v2 import run_graph_aug_v2
        if qa_file is None:
            qa_file = 'qa_pairs_tier2.json'
        raw_results = run_graph_aug_v2(ds_dir, api_key,
                                       api_base=api_base, model=model,
                                       qa_file=qa_file)
    elif baseline_name == 'tool_use_v2':
        from baselines.tool_use_v2 import run_tool_use_v2
        if qa_file is None:
            qa_file = 'qa_pairs_tier2.json'
        raw_results = run_tool_use_v2(ds_dir, api_key,
                                      api_base=api_base, model=model,
                                      qa_file=qa_file)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    # Load FK adjacency for path validation
    import torch
    fk_adj = None
    name_to_idx = None
    graph_path = ds_dir / 'schema_graph.pt'
    if obfuscated:
        obf_graph = ds_dir / 'obfuscated_schema_graph.pt'
        if obf_graph.exists():
            graph_path = obf_graph
    if graph_path.exists():
        gdata = torch.load(graph_path, weights_only=False)
        gnames = gdata['table'].table_names
        name_to_idx = {nm: i for i, nm in enumerate(gnames)}
        gn = len(gnames)
        fk_adj = {i: set() for i in range(gn)}
        if ('table', 'fk_to', 'table') in gdata.edge_types:
            ei = gdata['table', 'fk_to', 'table'].edge_index
            for j in range(ei.shape[1]):
                fk_adj[ei[0, j].item()].add(ei[1, j].item())

    # Score each result
    scored_results = []
    for r in raw_results:
        scores = score_answer(r['predicted_answer'], r['gold_answer'],
                              r['answer_type'],
                              fk_adj=fk_adj, name_to_idx=name_to_idx)
        scored_r = {**r, 'scores': scores}
        scored_results.append(scored_r)

    # Compute aggregates
    metrics = compute_aggregate_metrics(scored_results)

    # Print summary
    print(f"\n  Results for {baseline_name}{suffix} on {dataset}:")
    print(f"    Total: {metrics['total']}")
    print(f"    Micro-EM: {metrics['micro_em']:.1%}  |  Macro-EM: {metrics['macro_em']:.1%}")
    print(f"    Micro-F1: {metrics['micro_f1']:.1%}  |  Macro-F1: {metrics['macro_f1']:.1%}")
    print(f"\n    By difficulty:")
    for diff in ['easy', 'medium', 'hard']:
        if diff in metrics['by_difficulty']:
            d = metrics['by_difficulty'][diff]
            print(f"      {diff:8s}: EM={d['exact_match']:.1%}  "
                  f"F1={d['avg_f1']:.1%}  (n={d['total']})")
    print(f"\n    By type:")
    for t, d in sorted(metrics['by_type'].items()):
        print(f"      {t:20s}: EM={d['exact_match']:.1%}  "
              f"F1={d['avg_f1']:.1%}  (n={d['total']})")

    # Save results — include model tag so different models don't overwrite
    results_dir = repo_root / 'evaluation' / 'results'
    results_dir.mkdir(exist_ok=True)
    # Derive short model tag: "qwen/qwen3-vl-4b" -> "qwen3-vl-4b"
    model_tag = model.split('/')[-1].replace(' ', '_')
    out_name = f'{baseline_name}{suffix}_{dataset}_{model_tag}.json'
    out_path = results_dir / out_name
    output = {
        'baseline': baseline_name,
        'dataset': dataset,
        'obfuscated': obfuscated,
        'model': model,
        'model_tag': model_tag,
        'metrics': metrics,
        'results': scored_results,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n    Saved to: {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description='DW-Bench Evaluation Harness')
    parser.add_argument('--baseline', type=str, required=True,
                        choices=['flat_text', 'vector_rag', 'graph_aug', 'gnn_llm',
                                 'oracle', 'tool_use', 'react_code',
                                 'flat_text_v2', 'graph_aug_v2', 'tool_use_v2'])
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm', 'syn_logistics', 'all'])
    parser.add_argument('--obfuscated', action='store_true',
                        help='Use obfuscated schema and Q&A')
    parser.add_argument('--extended', action='store_true',
                        help='Use extended hard/medium Q&A set')
    parser.add_argument('--api-key', type=str,
                        default=os.environ.get('GOOGLE_API_KEY',
                                os.environ.get('DEEPSEEK_API_KEY', '')),
                        help='API key (reads GOOGLE_API_KEY or DEEPSEEK_API_KEY from env)')
    parser.add_argument('--api-base', type=str,
                        default='',
                        help='API base URL (e.g. https://generativelanguage.googleapis.com/v1beta/openai)')
    parser.add_argument('--model', type=str,
                        default='',
                        help='Model name (e.g., gemini-2.5-flash, deepseek-chat)')
    args = parser.parse_args()

    datasets = (['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm', 'syn_logistics']
                if args.dataset == 'all' else [args.dataset])

    all_results = []
    for ds in datasets:
        result = run_evaluation(
            args.baseline, ds, args.obfuscated, args.api_key,
            args.api_base, args.model, extended=args.extended)
        all_results.append(result)

    # Combined summary if multiple datasets
    if len(all_results) > 1:
        all_scored = []
        for r in all_results:
            all_scored.extend(r['results'])
        combined_metrics = compute_aggregate_metrics(all_scored)
        print(f"\n{'='*60}")
        print("COMBINED RESULTS")
        print(f"{'='*60}")
        print(f"  Total: {combined_metrics['total']}")
        print(f"  Exact Match: {combined_metrics['exact_match']:.1%}")
        print(f"  Avg F1:      {combined_metrics['avg_f1']:.1%}")
        for diff in ['easy', 'medium', 'hard']:
            if diff in combined_metrics['by_difficulty']:
                d = combined_metrics['by_difficulty'][diff]
                print(f"  {diff:8s}: EM={d['exact_match']:.1%}  "
                      f"F1={d['avg_f1']:.1%}  (n={d['total']})")


if __name__ == '__main__':
    main()
