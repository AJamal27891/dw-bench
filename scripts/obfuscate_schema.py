"""Phase 3: Obfuscate schema names to prevent text memorization shortcuts.

Scrambles all table and column names while preserving:
  - Graph topology (edge_index unchanged)
  - Structural features (degree, betweenness, etc.)
  - Edge types (fk_to, derived_from)

Produces:
  - obfuscated_schema_graph.pt  (graph with scrambled names)
  - qa_pairs_obfuscated.json    (Q&A with scrambled names)
  - obfuscation_map.json        (mapping for debugging)

Usage:
    python obfuscate_schema.py                    # all datasets
    python obfuscate_schema.py --dataset tpc-di   # single dataset
"""
import argparse
import json
import random
import re
import string
from pathlib import Path

import torch

SEED = 42
random.seed(SEED)


def generate_obfuscated_name(prefix: str, idx: int) -> str:
    """Generate a deterministic obfuscated name like Table_A3, col_x7."""
    letter = string.ascii_uppercase[idx % 26]
    number = idx // 26
    return f"{prefix}_{letter}{number}" if number > 0 else f"{prefix}_{letter}"


def build_table_mapping(table_names: list) -> dict:
    """Create mapping: original_name → obfuscated_name."""
    # Shuffle indices for randomness
    indices = list(range(len(table_names)))
    random.shuffle(indices)

    mapping = {}
    for rank, orig_idx in enumerate(indices):
        orig_name = table_names[orig_idx]
        mapping[orig_name] = generate_obfuscated_name('Table', rank)
    return mapping


def obfuscate_graph(graph_path: Path, output_dir: Path) -> dict:
    """Obfuscate a schema_graph.pt file."""
    data = torch.load(graph_path, weights_only=False)
    table_names = data['table'].table_names

    # Build mapping
    mapping = build_table_mapping(table_names)

    # Replace table names
    obf_names = [mapping[name] for name in table_names]
    data['table'].table_names = obf_names

    # Save obfuscated graph
    out_path = output_dir / 'obfuscated_schema_graph.pt'
    torch.save(data, out_path)

    # Save mapping
    map_path = output_dir / 'obfuscation_map.json'
    reverse_mapping = {v: k for k, v in mapping.items()}
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump({
            'original_to_obfuscated': mapping,
            'obfuscated_to_original': reverse_mapping,
        }, f, indent=2, ensure_ascii=False)

    print(f"    Obfuscated {len(mapping)} table names")
    print(f"    Saved graph to: {out_path}")
    print(f"    Saved mapping to: {map_path}")

    return mapping


def obfuscate_qa(qa_path: Path, mapping: dict, output_dir: Path,
                 original_names: list) -> None:
    """Apply obfuscation mapping to Q&A pairs and verify no leakage."""
    with open(qa_path, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    obfuscated_pairs = []
    for q in qa_pairs:
        oq = q.copy()

        # Obfuscate question text — protect common phrases that contain
        # table names as English words (e.g., 'relationship' in 'foreign key relationship')
        question = q['question']
        
        # Protect phrases that should NOT be touched
        protected_phrases = [
            'foreign key relationship',
            'data lineage relationship', 
        ]
        placeholders = {}
        for idx_ph, phrase in enumerate(protected_phrases):
            ph = f'__PROTECTED_{idx_ph}__'
            if phrase in question:
                question = question.replace(phrase, ph)
                placeholders[ph] = phrase
        
        # Now replace table names (longest first to avoid partial matches)
        for orig, obf in sorted(mapping.items(), key=lambda x: -len(x[0])):
            question = re.sub(r'(?<![a-zA-Z_.])' + re.escape(orig) + r'(?![a-zA-Z_])',
                              obf, question)
        
        # Restore protected phrases
        for ph, phrase in placeholders.items():
            question = question.replace(ph, phrase)
        
        oq['question'] = question

        # Obfuscate answer
        answer = q['answer']
        if isinstance(answer, list):
            oq['answer'] = sorted([
                mapping.get(a, a) for a in answer
            ])
        elif isinstance(answer, str) and answer in mapping:
            oq['answer'] = mapping[answer]
        # Leave booleans and integers unchanged

        oq['id'] = q['id'].replace(q.get('dataset', ''), q.get('dataset', ''))

        obfuscated_pairs.append(oq)

    # Save obfuscated Q&A
    out_path = output_dir / 'qa_pairs_obfuscated.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(obfuscated_pairs, f, indent=2, ensure_ascii=False)

    # ── Leakage check: verify no original names remain ──────────────
    obf_text = json.dumps(obfuscated_pairs)
    leaked = []
    for orig_name in original_names:
        # Skip very short names that could match substrings
        if len(orig_name) < 5:
            continue
        # Check if original name appears in obfuscated text
        if orig_name in obf_text:
            leaked.append(orig_name)

    if leaked:
        print(f"    ⚠️  LEAKAGE: {len(leaked)} original names found in "
              f"obfuscated Q&A: {leaked[:5]}")
    else:
        print(f"    ✅ No name leakage detected")

    print(f"    Saved {len(obfuscated_pairs)} obfuscated Q&A pairs to: "
          f"{out_path}")


def obfuscate_enriched_graph(enriched_path: Path, mapping: dict,
                             output_dir: Path) -> None:
    """Obfuscate enriched_schema_graph.pt by re-embedding obfuscated DDLs.

    Replaces DDL embeddings (first 384 dims) with embeddings of
    obfuscated table names. Preserves structural features (last 6 dims).
    """
    if not enriched_path.exists():
        print(f"    ⚠️  No enriched graph found, skipping")
        return

    data = torch.load(enriched_path, weights_only=False)
    original_names = list(data['table'].table_names)
    obf_names = [mapping[name] for name in original_names]

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer('all-MiniLM-L6-v2')
        ddl_texts = [f"CREATE TABLE {name} ( /* columns obfuscated */ );"
                     for name in obf_names]
        embeddings = model.encode(ddl_texts, show_progress_bar=False)

        x = data['table'].x
        new_x = torch.zeros_like(x)
        new_x[:, :384] = torch.from_numpy(np.array(embeddings)).float()
        new_x[:, 384:] = x[:, 384:]
        data['table'].x = new_x
    except ImportError:
        print("    ⚠️  sentence-transformers not available, zeroing DDL dims")
        x = data['table'].x
        data['table'].x = torch.cat([torch.zeros(x.shape[0], 384),
                                      x[:, 384:]], dim=1)

    data['table'].table_names = obf_names
    out_path = output_dir / 'obfuscated_enriched_schema_graph.pt'
    torch.save(data, out_path)
    print(f"    Saved obfuscated enriched graph: {out_path}")
    print(f"    Features: {list(data['table'].x.shape)}")


def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: Obfuscate schema names',
    )
    parser.add_argument(
        '--dataset', type=str, default='all',
        choices=['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm', 'all'],
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    datasets = (
        ['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm']
        if args.dataset == 'all'
        else [args.dataset]
    )

    print("=" * 60)
    print("Phase 3: Schema Obfuscation")
    print("=" * 60)

    for ds in datasets:
        ds_dir = repo_root / 'datasets' / ds
        graph_path = ds_dir / 'schema_graph.pt'
        enriched_path = ds_dir / 'enriched_schema_graph.pt'
        qa_path = ds_dir / 'qa_pairs.json'

        if not graph_path.exists():
            print(f"\n  ❌ {graph_path} not found, skipping {ds}")
            continue

        print(f"\n  Processing: {ds}")

        # Get original names for leakage check
        data = torch.load(graph_path, weights_only=False)
        original_names = list(data['table'].table_names)

        # Obfuscate graph
        mapping = obfuscate_graph(graph_path, ds_dir)

        # Obfuscate enriched graph
        obfuscate_enriched_graph(enriched_path, mapping, ds_dir)

        # Obfuscate Q&A (if exists)
        if qa_path.exists():
            obfuscate_qa(qa_path, mapping, ds_dir, original_names)
        else:
            print(f"    ⚠️  No Q&A pairs found at {qa_path}")

    print(f"\n{'=' * 60}")
    print("Obfuscation complete ✅")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
