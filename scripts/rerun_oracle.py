"""Threaded Oracle Re-Run Script.

Runs the hardened Oracle baseline across all 5 datasets, both models,
using ThreadPoolExecutor for maximum throughput.

Usage:
  python scripts/rerun_oracle.py --gemini-key <KEY> --deepseek-key <KEY>
"""
import json
import sys
import argparse
import concurrent.futures
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / 'evaluation' / 'baselines'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'evaluation'))
from baselines.oracle import run_oracle, _format_gold_evidence, SYSTEM_PROMPT
from baselines.flat_text import call_llm
from metrics import score_answer

DS_ALL = ['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm', 'syn_logistics']
RESULTS_DIR = Path('evaluation/results')


def run_single_question(qa, api_key, api_base, model):
    """Run a single Oracle question — thread-safe."""
    oracle_evidence = _format_gold_evidence(qa)
    user_prompt = (
        f"Context:\n{oracle_evidence}\n\n"
        f"Question: {qa['question']}\n"
        f"Expected answer type: {qa['answer_type']}."
    )

    response = call_llm(SYSTEM_PROMPT, user_prompt, api_key,
                        api_base=api_base, model=model)

    predicted = response.get('answer', response.get('Answer', ''))

    return {
        'id': qa['id'],
        'question': qa['question'],
        'gold_answer': qa['answer'],
        'predicted_answer': predicted,
        'answer_type': qa['answer_type'],
        'type': qa.get('type', qa.get('category', '')),
        'subtype': qa['subtype'],
        'difficulty': qa['difficulty'],
        'raw_response': response.get('_raw_text', ''),
    }


def rerun_oracle(api_key, api_base, model, model_tag, max_workers=20):
    """Run Oracle across all datasets with threading."""
    for ds in DS_ALL:
        ds_dir = Path(f'datasets/{ds}')
        qa_path = ds_dir / 'qa_pairs.json'
        if not qa_path.exists():
            print(f"  Skipping {ds}: no qa_pairs.json")
            continue

        qa_pairs = json.load(open(qa_path, encoding='utf-8'))
        print(f"\n{'='*60}")
        print(f"  Oracle [{model_tag}] {ds}: {len(qa_pairs)} questions, {max_workers} threads")
        print(f"{'='*60}")

        results = []
        errors = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single_question, qa, api_key, api_base, model): qa
                for qa in qa_pairs
            }
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    qa = futures[future]
                    print(f"    ERROR on {qa['id']}: {e}")
                    results.append({
                        'id': qa['id'],
                        'question': qa['question'],
                        'gold_answer': qa['answer'],
                        'predicted_answer': [],
                        'answer_type': qa['answer_type'],
                        'type': qa.get('type', qa.get('category', '')),
                        'subtype': qa['subtype'],
                        'difficulty': qa['difficulty'],
                        'raw_response': f'ERROR: {e}',
                    })
                    errors += 1

                if (i + 1) % 20 == 0 or (i + 1) == len(qa_pairs):
                    em = sum(1 for r in results if r['predicted_answer'] == r['gold_answer'])
                    print(f"    [{i+1}/{len(qa_pairs)}] EM: {em}/{len(results)} "
                          f"({em/len(results)*100:.1f}%) errors: {errors}")

        # Sort results by ID to match original order
        id_order = {qa['id']: idx for idx, qa in enumerate(qa_pairs)}
        results.sort(key=lambda r: id_order.get(r['id'], 999999))

        # Score all results
        import torch
        graph_path = ds_dir / 'schema_graph.pt'
        fk_adj, name_to_idx = None, None
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

        for r in results:
            scores = score_answer(
                r['predicted_answer'], r['gold_answer'], r['answer_type'],
                fk_adj=fk_adj, name_to_idx=name_to_idx,
                question=r.get('question', ''), subtype=r.get('subtype', '')
            )
            r['scores'] = scores

        # Compute final EM
        total_em = sum(1 for r in results if r['scores']['exact_match'])
        print(f"  FINAL: {total_em}/{len(results)} = {total_em/len(results)*100:.1f}% EM")

        # Save
        out_path = RESULTS_DIR / f'oracle_original_{ds}_{model_tag}.json'
        output = {
            'baseline': 'oracle',
            'condition': 'original',
            'dataset': ds,
            'model': model_tag,
            'results': results
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_path}")


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')

    parser = argparse.ArgumentParser(description='Threaded Oracle Re-Run')
    parser.add_argument('--gemini-key', type=str,
                        default=os.environ.get('GOOGLE_API_KEY', ''))
    parser.add_argument('--gemini-base', type=str,
                        default='https://generativelanguage.googleapis.com/v1beta/openai')
    parser.add_argument('--deepseek-key', type=str,
                        default=os.environ.get('DEEPSEEK_API_KEY', ''))
    parser.add_argument('--deepseek-base', type=str,
                        default='https://api.deepseek.com/v1')
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()

    if args.gemini_key:
        print("\n" + "=" * 60)
        print("GEMINI 2.5 FLASH ORACLE RE-RUN")
        print("=" * 60)
        rerun_oracle(args.gemini_key, args.gemini_base,
                     'gemini-2.5-flash', 'gemini-2.5-flash',
                     max_workers=args.workers)

    if args.deepseek_key:
        print("\n" + "=" * 60)
        print("DEEPSEEK-V3 ORACLE RE-RUN")
        print("=" * 60)
        rerun_oracle(args.deepseek_key, args.deepseek_base,
                     'deepseek-chat', 'deepseek-chat',
                     max_workers=args.workers)

    if not args.gemini_key and not args.deepseek_key:
        print("No API keys provided. Set GOOGLE_API_KEY and DEEPSEEK_API_KEY in .env")
        print("Or pass: --gemini-key <KEY> --deepseek-key <KEY>")
