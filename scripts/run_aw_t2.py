"""Run ALL AdventureWorks T2 evaluations in parallel.

9 runs total: 3 baselines × 3 models
Each model gets 2 workers (staggered baseline launches).

Usage: python scripts/run_aw_t2.py
"""
import subprocess
import sys
import os
import threading
import time
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

ROOT = Path(__file__).parent.parent
BASELINES = ['flat_text_v2', 'graph_aug_v2', 'tool_use_v2']
DATASET = 'adventureworks'

MODELS = [
    {
        'name': 'gemini-2.5-flash',
        'api_base': 'https://generativelanguage.googleapis.com/v1beta/openai',
        'api_key': os.environ.get('GOOGLE_API_KEY', ''),
    },
    {
        'name': 'deepseek-chat',
        'api_base': 'https://api.deepseek.com/v1',
        'api_key': os.environ.get('DEEPSEEK_API_KEY', ''),
    },
    {
        'name': 'Qwen/Qwen2.5-72B-Instruct',
        'api_base': 'https://api.deepinfra.com/v1/openai',
        'api_key': os.environ.get('DEEPINFRA_API_KEY', ''),
    },
]

results_lock = threading.Lock()
completed = []
failed = []


def run_one(baseline, model_cfg):
    """Run a single evaluation."""
    model = model_cfg['name']
    tag = f"{baseline}|{model.split('/')[-1]}"
    print(f"  ▶ Starting: {tag}")
    t0 = time.time()

    cmd = [
        sys.executable, 'evaluation/evaluate.py',
        '--baseline', baseline,
        '--dataset', DATASET,
        '--api-base', model_cfg['api_base'],
        '--model', model_cfg['name'],
        '--api-key', model_cfg['api_key'],
    ]

    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT),
            capture_output=True, text=True,
            timeout=7200,  # 2 hour timeout per run
            encoding='utf-8', errors='replace',
        )
        elapsed = time.time() - t0
        with results_lock:
            if result.returncode == 0:
                completed.append((tag, elapsed))
                print(f"  ✓ Completed: {tag} ({elapsed:.0f}s)")
            else:
                failed.append((tag, result.stderr[-500:] if result.stderr else ""))
                print(f"  ✗ FAILED: {tag} ({elapsed:.0f}s)")
                if result.stderr:
                    print(f"    stderr: {result.stderr[-300:]}")
    except subprocess.TimeoutExpired:
        with results_lock:
            failed.append((tag, "TIMEOUT (2h)"))
            print(f"  ✗ TIMEOUT: {tag}")


def main():
    total = len(BASELINES) * len(MODELS)

    # Verify API keys
    missing = []
    for m in MODELS:
        if not m['api_key']:
            missing.append(m['name'])
    if missing:
        print(f"ERROR: Missing API keys for: {missing}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  AdventureWorks T2 Parallel Evaluation")
    print(f"  Baselines: {', '.join(BASELINES)}")
    print(f"  Models:    {', '.join(m['name'].split('/')[-1] for m in MODELS)}")
    print(f"  Total runs: {total}")
    print(f"  Strategy:  2 workers per model (staggered)")
    print(f"{'='*60}\n")

    threads = []
    t_start = time.time()

    # Launch strategy: stagger by model to avoid rate limits
    # Worker 1: flat_text_v2  for each model (fast, schema-only)
    # Worker 2: graph_aug_v2  for each model (needs lineage context)
    # Later:    tool_use_v2   for each model (most API calls)

    # Wave 1: flat_text_v2 for all 3 models (quick, ~30 min each)
    for model_cfg in MODELS:
        t = threading.Thread(target=run_one,
                             args=('flat_text_v2', model_cfg))
        threads.append(t)
        t.start()
        time.sleep(3)  # 3s stagger between model launches

    # Wave 2: graph_aug_v2 (5s after wave 1 start)
    time.sleep(5)
    for model_cfg in MODELS:
        t = threading.Thread(target=run_one,
                             args=('graph_aug_v2', model_cfg))
        threads.append(t)
        t.start()
        time.sleep(3)

    # Wave 3: tool_use_v2 (10s after wave 2 start)
    time.sleep(10)
    for model_cfg in MODELS:
        t = threading.Thread(target=run_one,
                             args=('tool_use_v2', model_cfg))
        threads.append(t)
        t.start()
        time.sleep(3)

    # Wait for all to complete
    for t in threads:
        t.join()

    total_time = time.time() - t_start

    # Print summary
    print(f"\n{'='*60}")
    print(f"  RESULTS: {len(completed)}/{total} completed "
          f"({total_time/60:.1f} min)")
    print(f"{'='*60}")
    for tag, elapsed in sorted(completed):
        print(f"  ✓ {tag} ({elapsed/60:.1f} min)")
    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for tag, err in failed:
            print(f"  ✗ {tag}: {err[:200]}")

    # Show results summary
    print(f"\n{'='*60}")
    print(f"  SCORE SUMMARY")
    print(f"{'='*60}")
    results_dir = ROOT / 'evaluation' / 'results'
    for bl in BASELINES:
        print(f"\n  {bl}:")
        for mc in MODELS:
            model_tag = mc['name'].split('/')[-1].replace(' ', '_')
            fname = f"{bl}_original_{DATASET}_{model_tag}.json"
            fpath = results_dir / fname
            if fpath.exists():
                try:
                    d = json.load(open(fpath, encoding='utf-8'))
                    m = d.get('metrics', {})
                    em = m.get('macro_em', 0)
                    f1 = m.get('macro_f1', 0)
                    n = m.get('total', 0)
                    print(f"    {model_tag:25s}: Macro-EM={em:5.1%} "
                          f"F1={f1:5.1%} (n={n})")
                except Exception as e:
                    print(f"    {model_tag:25s}: Error reading: {e}")
            else:
                print(f"    {model_tag:25s}: NOT YET")


if __name__ == '__main__':
    main()
