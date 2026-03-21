#!/usr/bin/env python3
"""DW-Bench: Unified evaluation runner.

Runs all baselines (Flat Text, Vector RAG, Graph-Augmented) across all
datasets with a specified LLM via an OpenAI-compatible API.

Usage:
  # Original evaluation (Gemini Flash via Google AI)
  python run.py --api-base https://generativelanguage.googleapis.com/v1beta/openai \
                --model gemini-2.5-flash --api-key YOUR_KEY

  # Local model evaluation (e.g., LM Studio)
  python run.py --api-base http://localhost:1234/v1 \
                --model microsoft/phi-4-mini-reasoning

  # Run only obfuscated condition
  python run.py --api-base ... --model ... --condition obfuscated

  # Run only extended (hard+medium) questions
  python run.py --api-base ... --model ... --condition extended

  # Run a single baseline on a single dataset
  python run.py --api-base ... --model ... --baseline flat_text --dataset tpc-ds
"""
import subprocess
import sys
import os
import argparse
import time


BASELINES = ['flat_text', 'vector_rag', 'graph_aug', 'tool_use', 'react_code', 'oracle']
DATASETS = ['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm', 'syn_logistics']
T2_BASELINES = ['flat_text_v2', 'graph_aug_v2', 'tool_use_v2', 'react_code_v2']
CONDITIONS = ['original', 'obfuscated', 'extended']


def run_evaluation(baseline, dataset, condition, api_base, model, api_key):
    """Run a single evaluation and return (success, elapsed_seconds)."""
    cmd = [
        sys.executable, 'evaluation/evaluate.py',
        '--baseline', baseline,
        '--dataset', dataset,
        '--api-base', api_base,
        '--model', model,
    ]
    if api_key:
        cmd += ['--api-key', api_key]
    if condition == 'obfuscated':
        cmd += ['--obfuscated']
    elif condition == 'extended':
        cmd += ['--extended']

    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=os.path.dirname(os.path.abspath(__file__)) or '.')
    elapsed = time.time() - t0
    return result.returncode == 0, elapsed


def main():
    parser = argparse.ArgumentParser(
        description='DW-Bench: Run LLM evaluation on data warehouse schemas')
    parser.add_argument('--api-base', required=True,
                        help='OpenAI-compatible API base URL')
    parser.add_argument('--model', required=True,
                        help='Model name (e.g., gemini-2.5-flash)')
    parser.add_argument('--api-key', default='',
                        help='API key (optional for local models)')
    parser.add_argument('--baseline', choices=BASELINES + T2_BASELINES,
                        default=None,
                        help='Run a single baseline (default: all Tier 1)')
    parser.add_argument('--dataset', choices=DATASETS + ['all'], default='all',
                        help='Run a single dataset (default: all 5)')
    parser.add_argument('--condition', choices=CONDITIONS + ['all'],
                        default='original',
                        help='Evaluation condition (default: original)')
    parser.add_argument('--tier', choices=['1', '2'], default='1',
                        help='Tier 1 (schema-level) or Tier 2 (value-level)')
    parser.add_argument('--verify', action='store_true',
                        help='Run 5-question Oracle sample to verify pipeline')
    args = parser.parse_args()

    # Resolve API key from env if not provided
    if not args.api_key:
        for env_var in ['GOOGLE_API_KEY', 'DEEPSEEK_API_KEY', 'OPENAI_API_KEY']:
            val = os.environ.get(env_var, '')
            if val:
                args.api_key = val
                break

    # --verify: quick pipeline check
    if args.verify:
        print("\n  Running verification (Oracle on tpc-ds, original)...")
        ok, elapsed = run_evaluation(
            'oracle', 'tpc-ds', 'original',
            args.api_base, args.model, args.api_key)
        if ok:
            print(f"  ✓ Pipeline verified in {elapsed:.0f}s")
            print("  Run 'python view_results.py' to see results.")
        else:
            print(f"  ✗ Verification failed after {elapsed:.0f}s")
        return

    if args.tier == '2':
        baselines = [args.baseline] if args.baseline else T2_BASELINES
    else:
        baselines = [args.baseline] if args.baseline else BASELINES
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    conditions = CONDITIONS if args.condition == 'all' else [args.condition]

    total = len(baselines) * len(datasets) * len(conditions)
    done = 0
    failed = []

    print(f"\n{'=' * 70}")
    print(f"  DW-Bench Evaluation")
    print(f"  Model:      {args.model}")
    print(f"  API:        {args.api_base}")
    print(f"  Baselines:  {', '.join(baselines)}")
    print(f"  Datasets:   {', '.join(datasets)}")
    print(f"  Conditions: {', '.join(conditions)}")
    print(f"  Total runs: {total}")
    print(f"{'=' * 70}\n")

    t_start = time.time()

    for condition in conditions:
        for bl in baselines:
            for ds in datasets:
                done += 1
                print(f"\n{'#' * 70}")
                print(f"# [{done}/{total}] {condition.upper()}: "
                      f"{bl} | {ds} | {args.model}")
                print(f"{'#' * 70}\n")

                ok, elapsed = run_evaluation(
                    bl, ds, condition, args.api_base, args.model, args.api_key)

                if ok:
                    print(f"\n  ✓ Done in {elapsed:.0f}s")
                else:
                    print(f"\n  ✗ FAILED after {elapsed:.0f}s — continuing")
                    failed.append(f"{condition}/{bl}/{ds}")

    total_time = time.time() - t_start

    print(f"\n{'=' * 70}")
    print(f"  COMPLETE: {done - len(failed)}/{total} succeeded "
          f"({total_time/60:.1f} min)")
    if failed:
        print(f"  FAILED ({len(failed)}):")
        for f in failed:
            print(f"    ✗ {f}")
    print(f"{'=' * 70}")
    print(f"\n  View results: python view_results.py")
    print(f"  Check integrity: python integrity_check.py\n")


if __name__ == '__main__':
    main()
