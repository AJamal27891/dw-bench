"""Generate Tier 2 value-level QA pairs for AdventureWorks.

Produces questions across 8 subtypes in 3 categories:
  Data Provenance:     row_provenance, value_origin, multi_hop_trace
  Forward Impact:      row_impact, value_propagation, cascade_count
  Cross-Silo:          cross_silo_reachability, shared_source

All answers are programmatically derived from lineage_map.json + value_data CSVs.

Usage:
    python scripts/generate_aw_value_qa.py
"""
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

REPO_ROOT = Path(__file__).parent.parent
DS_DIR = REPO_ROOT / 'datasets' / 'adventureworks'
VALUE_DIR = DS_DIR / 'value_data'


def load_data():
    with open(DS_DIR / 'lineage_map.json', 'r', encoding='utf-8') as f:
        lineage_map = json.load(f)
    with open(VALUE_DIR / '_manifest.json', 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    return lineage_map, manifest


def table_layer(manifest, table_name):
    return manifest.get(table_name, {}).get('layer', 'unknown')


def build_reverse_lineage(lineage_map):
    """Build reverse index: source_id → [(target_table, target_row)]."""
    reverse = defaultdict(list)
    for target_table, rows_map in lineage_map.items():
        for row_key, row_info in rows_map.items():
            for src in row_info['sources']:
                for src_row_idx in src['rows']:
                    src_id = f"{src['table']}:row_{src_row_idx}"
                    reverse[src_id].append((target_table, row_key))
    return reverse


def trace_full_provenance(lineage_map, target_table, target_row):
    """Trace full backward provenance. Returns {source_table: set(row_ids)}."""
    all_sources = defaultdict(set)
    queue = [(target_table, target_row)]
    visited = set()
    while queue:
        tbl, row = queue.pop()
        if (tbl, row) in visited:
            continue
        visited.add((tbl, row))
        if tbl in lineage_map and row in lineage_map[tbl]:
            for src in lineage_map[tbl][row]['sources']:
                src_table = src['table']
                for src_row_idx in src['rows']:
                    src_row = f'row_{src_row_idx}'
                    all_sources[src_table].add(src_row)
                    queue.append((src_table, src_row))
    return dict(all_sources)


def trace_full_impact(reverse_lineage, source_table, source_row):
    """Trace forward impact. Returns {target_table: set(row_keys)}."""
    all_targets = defaultdict(set)
    queue = [f'{source_table}:{source_row}']
    visited = set()
    while queue:
        src_id = queue.pop()
        if src_id in visited:
            continue
        visited.add(src_id)
        for target_table, target_row in reverse_lineage.get(src_id, []):
            all_targets[target_table].add(target_row)
            queue.append(f'{target_table}:{target_row}')
    return dict(all_targets)


def count_lineage_depth(lineage_map, target_table, target_row):
    """Count the max depth of backward provenance chain."""
    max_depth = 0
    queue = [(target_table, target_row, 0)]
    visited = set()
    while queue:
        tbl, row, depth = queue.pop()
        if (tbl, row) in visited:
            continue
        visited.add((tbl, row))
        max_depth = max(max_depth, depth)
        if tbl in lineage_map and row in lineage_map[tbl]:
            for src in lineage_map[tbl][row]['sources']:
                for src_row_idx in src['rows']:
                    queue.append((src['table'], f'row_{src_row_idx}', depth + 1))
    return max_depth


def generate_qa():
    lineage_map, manifest = load_data()
    reverse_lineage = build_reverse_lineage(lineage_map)

    rng = random.Random(42)
    qa_pairs = []
    qa_id = 0
    seen_questions = set()

    def add_qa(q_type, subtype, question, answer, answer_type,
               difficulty, reasoning):
        nonlocal qa_id
        if question in seen_questions:
            return
        seen_questions.add(question)
        qa_pairs.append({
            'id': f'adventureworks_tier2_{subtype}_{qa_id:03d}',
            'dataset': 'adventureworks',
            'tier': 2,
            'type': q_type,
            'subtype': subtype,
            'question': question,
            'answer': answer,
            'answer_type': answer_type,
            'difficulty': difficulty,
            'reasoning': reasoning,
        })
        qa_id += 1

    # Classify tables
    oltp_tables = sorted([t for t in manifest if manifest[t]['layer'] == 'oltp'])
    stg_tables = sorted([t for t in manifest if manifest[t]['layer'] == 'staging'])
    dw_tables = sorted([t for t in manifest if manifest[t]['layer'] == 'dw'])
    dim_tables = sorted([t for t in dw_tables if t.startswith('Dim')])
    fact_tables = sorted([t for t in dw_tables if t.startswith('Fact')])

    # ══════════════════════════════════════════════════════════
    # 1. ROW PROVENANCE — Which rows directly contributed to this DW row?
    # ══════════════════════════════════════════════════════════
    print('  Generating row_provenance questions...')

    # DW dim tables: 4 rows each
    for dw_table in dim_tables:
        rows_map = lineage_map.get(dw_table, {})
        if not rows_map:
            continue
        sample_rows = rng.sample(list(rows_map.keys()),
                                  min(4, len(rows_map)))
        for row_key in sample_rows:
            row_info = rows_map[row_key]
            direct_sources = []
            for src in row_info['sources']:
                for r in src['rows']:
                    direct_sources.append(f"{src['table']}:row_{r}")
            add_qa('data_provenance', 'row_provenance',
                   f"Which rows directly contributed to {dw_table} {row_key}?",
                   sorted(direct_sources), 'list', 'hard',
                   f'Direct lookup in lineage_map[{dw_table}][{row_key}]')

    # Fact tables: 3 rows each
    for dw_table in fact_tables:
        rows_map = lineage_map.get(dw_table, {})
        if not rows_map:
            continue
        sample_rows = rng.sample(list(rows_map.keys()),
                                  min(3, len(rows_map)))
        for row_key in sample_rows:
            row_info = rows_map[row_key]
            direct_sources = []
            for src in row_info['sources']:
                for r in src['rows']:
                    direct_sources.append(f"{src['table']}:row_{r}")
            add_qa('data_provenance', 'row_provenance',
                   f"Which rows directly contributed to {dw_table} {row_key}?",
                   sorted(direct_sources), 'list', 'hard',
                   f'Direct lookup in lineage_map[{dw_table}][{row_key}]')

    # ══════════════════════════════════════════════════════════
    # 2. VALUE ORIGIN — Which OLTP tables does this DW row originate from?
    # ══════════════════════════════════════════════════════════
    print('  Generating value_origin questions...')

    for dw_table in dim_tables + fact_tables:
        rows_map = lineage_map.get(dw_table, {})
        if not rows_map:
            continue
        sample_rows = rng.sample(list(rows_map.keys()),
                                  min(3, len(rows_map)))
        for row_key in sample_rows:
            prov = trace_full_provenance(lineage_map, dw_table, row_key)
            oltp_sources = sorted([t for t in prov if t in oltp_tables])
            if oltp_sources:
                depth = count_lineage_depth(lineage_map, dw_table, row_key)
                diff = 'hard' if depth >= 2 else 'medium'
                add_qa('data_provenance', 'value_origin',
                       f"Which OLTP source tables does {dw_table} {row_key} "
                       f"originate from?",
                       oltp_sources, 'list', diff,
                       f'Transitive backward trace from {dw_table} {row_key} '
                       f'(depth={depth})')

    # ══════════════════════════════════════════════════════════
    # 3. MULTI-HOP TRACE — Which staging tables processed data?
    # ══════════════════════════════════════════════════════════
    print('  Generating multi_hop_trace questions...')

    for dw_table in dim_tables + fact_tables:
        rows_map = lineage_map.get(dw_table, {})
        if not rows_map:
            continue
        sample_rows = rng.sample(list(rows_map.keys()),
                                  min(4, len(rows_map)))
        for row_key in sample_rows:
            prov = trace_full_provenance(lineage_map, dw_table, row_key)
            stg_in_chain = sorted([t for t in prov if t in stg_tables])
            if stg_in_chain:
                depth = count_lineage_depth(lineage_map, dw_table, row_key)
                add_qa('data_provenance', 'multi_hop_trace',
                       f"Which staging tables processed data that ended up in "
                       f"{dw_table} {row_key}?",
                       stg_in_chain, 'list', 'hard',
                       f'Intermediate staging tables in provenance chain of '
                       f'{dw_table} {row_key} (depth={depth})')

    # Full chain trace: ALL intermediate tables
    for dw_table in dim_tables[:6]:
        rows_map = lineage_map.get(dw_table, {})
        if not rows_map:
            continue
        sample_rows = rng.sample(list(rows_map.keys()),
                                  min(2, len(rows_map)))
        for row_key in sample_rows:
            prov = trace_full_provenance(lineage_map, dw_table, row_key)
            all_tables = sorted(prov.keys())
            if len(all_tables) >= 2:
                add_qa('data_provenance', 'multi_hop_trace',
                       f"List ALL source tables in the full provenance chain "
                       f"of {dw_table} {row_key}.",
                       all_tables, 'list', 'hard',
                       f'Complete provenance chain with {len(all_tables)} '
                       f'tables for {dw_table} {row_key}')

    # ══════════════════════════════════════════════════════════
    # 4. ROW IMPACT — If OLTP row deleted, which DW tables lose data?
    # ══════════════════════════════════════════════════════════
    print('  Generating row_impact questions...')

    for oltp_table in oltp_tables:
        n_rows = manifest[oltp_table]['rows']
        sample_idxs = rng.sample(range(n_rows), min(3, n_rows))
        for row_idx in sample_idxs:
            impact = trace_full_impact(reverse_lineage, oltp_table,
                                        f'row_{row_idx}')
            dw_affected = sorted([t for t in impact if t in dw_tables])
            if dw_affected:
                add_qa('forward_impact', 'row_impact',
                       f"If {oltp_table} row_{row_idx} is deleted, "
                       f"which DW tables lose data?",
                       dw_affected, 'list', 'hard',
                       f'Forward trace from {oltp_table}:row_{row_idx}')

    # Staging→DW impact
    for stg_table in stg_tables[:8]:
        n_rows = manifest[stg_table]['rows']
        sample_idxs = rng.sample(range(n_rows), min(2, n_rows))
        for row_idx in sample_idxs:
            impact = trace_full_impact(reverse_lineage, stg_table,
                                        f'row_{row_idx}')
            if not impact:
                continue
            all_affected = sorted(impact.keys())
            if len(all_affected) >= 1:
                add_qa('forward_impact', 'row_impact',
                       f"If {stg_table} row_{row_idx} is deleted, "
                       f"which DW tables are affected?",
                       all_affected, 'list', 'hard',
                       f'Forward trace from mid-pipeline '
                       f'{stg_table}:row_{row_idx}')

    # ══════════════════════════════════════════════════════════
    # 5. VALUE PROPAGATION — Does data from OLTP X flow to DW Y?
    # ══════════════════════════════════════════════════════════
    print('  Generating value_propagation questions...')

    # Precompute OLTP sources per DW table
    dw_provenance_cache = {}
    for dw_table in dw_tables:
        rows_map = lineage_map.get(dw_table, {})
        if rows_map:
            first_row = next(iter(rows_map))
            prov = trace_full_provenance(lineage_map, dw_table, first_row)
            dw_provenance_cache[dw_table] = set(prov.keys())
        else:
            dw_provenance_cache[dw_table] = set()

    all_combos = list(itertools.product(oltp_tables, dw_tables))
    rng.shuffle(all_combos)

    true_combos = [(o, d) for o, d in all_combos
                   if o in dw_provenance_cache.get(d, set())]
    false_combos = [(o, d) for o, d in all_combos
                    if o not in dw_provenance_cache.get(d, set())]

    selected = (rng.sample(true_combos, min(20, len(true_combos))) +
                rng.sample(false_combos, min(20, len(false_combos))))
    rng.shuffle(selected)

    for oltp_table, dw_table in selected:
        reaches = oltp_table in dw_provenance_cache.get(dw_table, set())
        add_qa('forward_impact', 'value_propagation',
               f"Does data from {oltp_table} propagate to {dw_table}?",
               reaches, 'boolean', 'medium',
               f'Check if {oltp_table} is in transitive provenance of '
               f'{dw_table}')

    # ══════════════════════════════════════════════════════════
    # 6. CASCADE COUNT — Total downstream rows affected
    # ══════════════════════════════════════════════════════════
    print('  Generating cascade_count questions...')

    for oltp_table in oltp_tables:
        n_rows = manifest[oltp_table]['rows']
        sample_idxs = rng.sample(range(n_rows), min(3, n_rows))
        for row_idx in sample_idxs:
            impact = trace_full_impact(reverse_lineage, oltp_table,
                                        f'row_{row_idx}')
            total_affected = sum(len(rows) for rows in impact.values())
            if total_affected > 0:
                add_qa('forward_impact', 'cascade_count',
                       f"How many total downstream rows across all tables "
                       f"are affected if {oltp_table} row_{row_idx} is deleted?",
                       total_affected, 'integer', 'hard',
                       f'Sum of downstream rows from '
                       f'{oltp_table}:row_{row_idx}')

    # DW-only cascade count
    for oltp_table in oltp_tables[:10]:
        n_rows = manifest[oltp_table]['rows']
        row_idx = rng.randint(0, n_rows - 1)
        impact = trace_full_impact(reverse_lineage, oltp_table,
                                    f'row_{row_idx}')
        dw_rows = sum(len(rows) for t, rows in impact.items()
                      if t in dw_tables)
        if dw_rows > 0:
            add_qa('forward_impact', 'cascade_count',
                   f"How many DW-layer rows are affected if {oltp_table} "
                   f"row_{row_idx} is deleted?",
                   dw_rows, 'integer', 'hard',
                   f'DW-only cascade from {oltp_table}:row_{row_idx}')

    # ══════════════════════════════════════════════════════════
    # 7. CROSS-SILO REACHABILITY — Does data flow between schemas?
    # ══════════════════════════════════════════════════════════
    print('  Generating cross_silo_reachability questions...')

    # AW has schema-based silos: Person, Sales, Production, Purchasing, HR
    schema_silos = defaultdict(list)
    for t in oltp_tables:
        schema = t.split('.')[0]
        schema_silos[schema].append(t)
    silo_names = sorted(schema_silos.keys())

    for src_silo, tgt_silo in itertools.permutations(silo_names, 2):
        src_tables = schema_silos[src_silo]
        # Check if any src table feeds any DW table
        flows = False
        for dw_table in dw_tables:
            if flows:
                break
            prov_set = dw_provenance_cache.get(dw_table, set())
            for src_t in src_tables:
                if src_t in prov_set:
                    flows = True
                    break

        add_qa('cross_silo', 'cross_silo_reachability',
               f"Does any data from the {src_silo} schema's OLTP tables "
               f"flow into any DW dimension or fact table?",
               flows, 'boolean', 'hard',
               f'Check {src_silo} OLTP → DW reachability')

    # Per-table cross-schema questions
    cross_schema_combos = [(o, d) for o, d in all_combos]
    rng.shuffle(cross_schema_combos)

    true_xs = [(o, d) for o, d in cross_schema_combos
               if o in dw_provenance_cache.get(d, set())]
    false_xs = [(o, d) for o, d in cross_schema_combos
                if o not in dw_provenance_cache.get(d, set())]

    xs_sample = (rng.sample(true_xs, min(10, len(true_xs))) +
                 rng.sample(false_xs, min(10, len(false_xs))))
    rng.shuffle(xs_sample)

    for oltp_table, dw_table in xs_sample:
        reaches = oltp_table in dw_provenance_cache.get(dw_table, set())
        src_schema = oltp_table.split('.')[0]
        add_qa('cross_silo', 'cross_silo_reachability',
               f"Does {oltp_table} ({src_schema} schema) contribute data to "
               f"{dw_table}?",
               reaches, 'boolean', 'hard',
               f'Cross-schema: {oltp_table} → {dw_table}')

    # ══════════════════════════════════════════════════════════
    # 8. SHARED SOURCE — Which OLTP tables feed BOTH DW tables?
    # ══════════════════════════════════════════════════════════
    print('  Generating shared_source questions...')

    # DW dim pairs
    for dim_a, dim_b in itertools.combinations(dim_tables, 2):
        oltp_a = {t for t in dw_provenance_cache.get(dim_a, set())
                  if t in oltp_tables}
        oltp_b = {t for t in dw_provenance_cache.get(dim_b, set())
                  if t in oltp_tables}
        shared = sorted(oltp_a & oltp_b)
        add_qa('cross_silo', 'shared_source',
               f"Which OLTP source tables feed data into BOTH "
               f"{dim_a} AND {dim_b}?",
               shared, 'list', 'medium',
               f'Intersection of OLTP provenance for {dim_a} and {dim_b}')

    # Dim × Fact pairs (harder)
    dim_fact_pairs = list(itertools.product(dim_tables[:6], fact_tables))
    rng.shuffle(dim_fact_pairs)
    for dim_t, fact_t in dim_fact_pairs[:15]:
        oltp_dim = {t for t in dw_provenance_cache.get(dim_t, set())
                    if t in oltp_tables}
        oltp_fact = {t for t in dw_provenance_cache.get(fact_t, set())
                     if t in oltp_tables}
        shared = sorted(oltp_dim & oltp_fact)
        add_qa('cross_silo', 'shared_source',
               f"Which OLTP source tables feed data into BOTH "
               f"{dim_t} AND {fact_t}?",
               shared, 'list', 'hard',
               f'Cross-type shared sources: {dim_t} ∩ {fact_t}')

    # ──────────────────────────────────────────────────────────
    # Save QA pairs
    # ──────────────────────────────────────────────────────────
    out_path = DS_DIR / 'qa_pairs_tier2.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    # Summary
    print(f'\n  Generated {len(qa_pairs)} QA pairs:')
    by_subtype = defaultdict(lambda: defaultdict(int))
    by_type = defaultdict(int)
    by_difficulty = defaultdict(int)
    for q in qa_pairs:
        by_subtype[q['subtype']][q['difficulty']] += 1
        by_type[q['type']] += 1
        by_difficulty[q['difficulty']] += 1

    print(f'\n  {"Subtype":30s} {"medium":>8s} {"hard":>8s} {"total":>8s}')
    print(f'  {"-"*56}')
    for st in sorted(by_subtype.keys()):
        d = by_subtype[st]
        total = sum(d.values())
        print(f'  {st:30s} {d.get("medium",0):>8d} {d.get("hard",0):>8d} '
              f'{total:>8d}')
    print(f'  {"-"*56}')
    print(f'  {"TOTAL":30s} {by_difficulty.get("medium",0):>8d} '
          f'{by_difficulty.get("hard",0):>8d} {len(qa_pairs):>8d}')

    print(f'\n  By type:')
    for t, cnt in sorted(by_type.items()):
        print(f'    {t:30s}: {cnt}')

    print(f'\n  Saved to: {out_path}')
    return qa_pairs


if __name__ == '__main__':
    print('Generating Tier 2 QA pairs for AdventureWorks...')
    generate_qa()
    print('\n✅ Done!')
