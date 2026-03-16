"""Generate Tier 2 value-level QA pairs for Syn-Logistics.

Produces questions across 8 subtypes in 3 categories:
  Data Provenance:     row_provenance, value_origin, multi_hop_trace
  Forward Impact:      row_impact, value_propagation, cascade_count
  Cross-Silo:          cross_silo_reachability, shared_source

All answers are programmatically derived from lineage_map.json + value_data CSVs.

Target: ~40 questions per subtype, majority hard difficulty.

Usage:
    python scripts/generate_value_qa.py
"""
import csv
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

REPO_ROOT = Path(__file__).parent.parent
DS_DIR = REPO_ROOT / 'datasets' / 'syn_logistics'
VALUE_DIR = DS_DIR / 'value_data'


# ──────────────────────────────────────────────────────────────────
# SILOS definition (same as in generate_value_data.py)
# ──────────────────────────────────────────────────────────────────
SILOS = {
    'logistics': {
        'raw': ['raw_purchase_orders', 'raw_shipment_tracking', 'raw_supplier_catalog',
                'raw_warehouse_inventory', 'raw_freight_invoices'],
        'staging': ['stg_orders_cleaned', 'stg_shipments_validated', 'stg_suppliers_deduped',
                    'stg_inventory_snapshot', 'stg_freight_reconciled'],
        'core': ['dim_supplier', 'dim_product', 'dim_warehouse', 'dim_carrier',
                 'fact_shipment', 'fact_purchase_order'],
        'mart': ['mart_delivery_performance', 'mart_procurement_cost',
                 'mart_supplier_scorecard', 'mart_inventory_turnover'],
    },
    'hr': {
        'raw': ['raw_employee_records', 'raw_department_hierarchy',
                'raw_compensation_data'],
        'staging': ['stg_employees_cleaned', 'stg_departments_mapped',
                    'stg_compensation_normalized'],
        'core': ['dim_employee', 'dim_department', 'fact_payroll'],
        'mart': ['mart_headcount_report', 'mart_attrition_analysis'],
    },
    'healthcare': {
        'raw': ['raw_patient_intake', 'raw_lab_results', 'raw_prescriptions'],
        'staging': ['stg_patients_validated', 'stg_labs_standardized',
                    'stg_prescriptions_coded'],
        'core': ['dim_patient', 'dim_provider', 'fact_encounter'],
        'mart': ['mart_readmission_risk', 'mart_treatment_outcomes'],
    },
    'ecommerce': {
        'raw': ['raw_cart_sessions', 'raw_product_catalog', 'raw_payment_events'],
        'staging': ['stg_sessions_enriched', 'stg_products_normalized',
                    'stg_payments_validated'],
        'core': ['dim_customer', 'dim_product_listing', 'fact_transaction'],
        'mart': ['mart_conversion_funnel', 'mart_revenue_dashboard'],
    },
    'finance': {
        'raw': ['raw_ledger_entries', 'raw_bank_feeds', 'raw_tax_filings'],
        'staging': ['stg_ledger_reconciled', 'stg_bank_matched',
                    'stg_tax_validated'],
        'core': ['dim_account', 'dim_cost_center', 'fact_journal_entry'],
        'mart': ['mart_trial_balance', 'mart_cash_flow_statement'],
    },
}

SILO_NAMES = list(SILOS.keys())


def table_to_silo(table_name: str) -> str:
    for silo_name, layers in SILOS.items():
        for tables in layers.values():
            if table_name in tables:
                return silo_name
    return 'unknown'


def table_to_layer(table_name: str) -> str:
    for silo_name, layers in SILOS.items():
        for layer_name, tables in layers.items():
            if table_name in tables:
                return layer_name
    return 'unknown'


def load_data():
    with open(DS_DIR / 'lineage_map.json', 'r', encoding='utf-8') as f:
        lineage_map = json.load(f)
    with open(VALUE_DIR / '_manifest.json', 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    return lineage_map, manifest


def build_reverse_lineage(lineage_map: dict) -> dict:
    reverse = defaultdict(list)
    for target_table, rows_map in lineage_map.items():
        for row_key, row_info in rows_map.items():
            for src in row_info['sources']:
                for src_row_idx in src['rows']:
                    src_id = f"{src['table']}:row_{src_row_idx}"
                    reverse[src_id].append((target_table, row_key))
    return reverse


def trace_full_provenance(lineage_map: dict, target_table: str,
                          target_row: str) -> dict:
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


def trace_full_impact(reverse_lineage: dict, source_table: str,
                      source_row: str) -> dict:
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


def count_lineage_depth(lineage_map: dict, target_table: str,
                        target_row: str) -> int:
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


def read_csv_row(table_name: str, row_idx: int) -> dict:
    csv_path = VALUE_DIR / f'{table_name}.csv'
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            if i == row_idx:
                return dict(zip(header, row))
    return {}


def generate_qa():
    """Generate all Tier 2 QA pairs. Target: ~40 per subtype."""
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
            return  # skip duplicates
        seen_questions.add(question)
        qa_pairs.append({
            'id': f'syn_logistics_tier2_{subtype}_{qa_id:03d}',
            'dataset': 'syn_logistics',
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

    mart_tables = sorted([t for t in manifest if t.startswith('mart_')])
    dim_tables = sorted([t for t in manifest if t.startswith('dim_')])
    fact_tables = sorted([t for t in manifest if t.startswith('fact_')])
    stg_tables = sorted([t for t in manifest if t.startswith('stg_')])
    raw_tables = sorted([t for t in manifest if t.startswith('raw_')])

    # ══════════════════════════════════════════════════════════
    # 1. ROW PROVENANCE — target ~45 (hard)
    #    Direct source rows for mart, fact, and staging rows
    # ══════════════════════════════════════════════════════════
    print('  Generating row_provenance questions...')

    # Mart rows: 4 rows × 12 mart tables = ~48
    for mart_table in mart_tables:
        rows_map = lineage_map.get(mart_table, {})
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
                   f"Which rows directly contributed to {mart_table} {row_key}?",
                   sorted(direct_sources), 'list', 'hard',
                   f'Direct lookup in lineage_map[{mart_table}][{row_key}]')

    # Fact/dim rows for more coverage: 2 rows × ~15 core tables
    for core_table in fact_tables + dim_tables:
        rows_map = lineage_map.get(core_table, {})
        if not rows_map:
            continue
        sample_rows = rng.sample(list(rows_map.keys()),
                                 min(2, len(rows_map)))
        for row_key in sample_rows:
            row_info = rows_map[row_key]
            direct_sources = []
            for src in row_info['sources']:
                for r in src['rows']:
                    direct_sources.append(f"{src['table']}:row_{r}")
            add_qa('data_provenance', 'row_provenance',
                   f"Which rows directly contributed to {core_table} {row_key}?",
                   sorted(direct_sources), 'list', 'hard',
                   f'Direct lookup in lineage_map[{core_table}][{row_key}]')

    # ══════════════════════════════════════════════════════════
    # 2. VALUE ORIGIN — target ~40 (medium + hard)
    #    Which raw tables does a row originate from?
    # ══════════════════════════════════════════════════════════
    print('  Generating value_origin questions...')

    # Medium: dim table origins (3 rows × 12 dims)
    for dim_table in dim_tables:
        rows_map = lineage_map.get(dim_table, {})
        if not rows_map:
            continue
        sample_rows = rng.sample(list(rows_map.keys()), min(3, len(rows_map)))
        for row_key in sample_rows:
            prov = trace_full_provenance(lineage_map, dim_table, row_key)
            raw_sources = sorted([t for t in prov if t.startswith('raw_')])
            if raw_sources:
                add_qa('data_provenance', 'value_origin',
                       f"Which raw source tables does {dim_table} {row_key} originate from?",
                       raw_sources, 'list', 'medium',
                       f'Transitive backward trace from {dim_table} {row_key}')

    # Hard: mart table origins (deeper chain, 2 rows × 12 marts)
    for mart_table in mart_tables:
        rows_map = lineage_map.get(mart_table, {})
        if not rows_map:
            continue
        sample_rows = rng.sample(list(rows_map.keys()), min(2, len(rows_map)))
        for row_key in sample_rows:
            prov = trace_full_provenance(lineage_map, mart_table, row_key)
            raw_sources = sorted([t for t in prov if t.startswith('raw_')])
            if raw_sources:
                add_qa('data_provenance', 'value_origin',
                       f"Which raw source tables does {mart_table} {row_key} originate from?",
                       raw_sources, 'list', 'hard',
                       f'Deep transitive backward trace (3+ hops) from '
                       f'{mart_table} {row_key}')

    # ══════════════════════════════════════════════════════════
    # 3. MULTI-HOP TRACE — target ~45 (medium + hard)
    #    Which staging/core tables are in the provenance chain?
    # ══════════════════════════════════════════════════════════
    print('  Generating multi_hop_trace questions...')

    # Medium: staging tables in mart provenance (4 rows × 12 marts)
    for mart_table in mart_tables:
        rows_map = lineage_map.get(mart_table, {})
        if not rows_map:
            continue
        sample_rows = rng.sample(list(rows_map.keys()), min(4, len(rows_map)))
        for row_key in sample_rows:
            prov = trace_full_provenance(lineage_map, mart_table, row_key)
            stg_in_chain = sorted([t for t in prov if t.startswith('stg_')])
            if stg_in_chain:
                depth = count_lineage_depth(lineage_map, mart_table, row_key)
                diff = 'hard' if depth >= 3 else 'medium'
                add_qa('data_provenance', 'multi_hop_trace',
                       f"Which staging tables processed data that ended up in "
                       f"{mart_table} {row_key}?",
                       stg_in_chain, 'list', diff,
                       f'Intermediate staging tables in provenance chain of '
                       f'{mart_table} {row_key} (depth={depth})')

    # Hard: ALL intermediate tables (not just staging)
    for mart_table in mart_tables[:6]:
        rows_map = lineage_map.get(mart_table, {})
        if not rows_map:
            continue
        sample_rows = rng.sample(list(rows_map.keys()), min(2, len(rows_map)))
        for row_key in sample_rows:
            prov = trace_full_provenance(lineage_map, mart_table, row_key)
            all_tables = sorted(prov.keys())
            if len(all_tables) >= 3:
                add_qa('data_provenance', 'multi_hop_trace',
                       f"List ALL source tables (raw, staging, core) in the "
                       f"full provenance chain of {mart_table} {row_key}.",
                       all_tables, 'list', 'hard',
                       f'Complete provenance chain with {len(all_tables)} '
                       f'table types for {mart_table} {row_key}')

    # ══════════════════════════════════════════════════════════
    # 4. ROW IMPACT — target ~50 (hard)
    #    Forward: if source row deleted, which tables lose data?
    # ══════════════════════════════════════════════════════════
    print('  Generating row_impact questions...')

    # 4 rows per raw table × 14 raw tables = ~56
    for raw_table in raw_tables:
        n_rows = manifest[raw_table]['rows']
        sample_rows = rng.sample(range(n_rows), min(4, n_rows))
        for raw_row_idx in sample_rows:
            impact = trace_full_impact(reverse_lineage, raw_table,
                                       f'row_{raw_row_idx}')
            if not impact:
                continue
            mart_affected = sorted([t for t in impact if t.startswith('mart_')])
            if mart_affected:
                add_qa('forward_impact', 'row_impact',
                       f"If {raw_table} row_{raw_row_idx} is deleted, "
                       f"which mart tables lose data?",
                       mart_affected, 'list', 'hard',
                       f'Forward trace from {raw_table}:row_{raw_row_idx}')

    # Also: staging→mart impact (harder to reason about)
    for stg_table in stg_tables[:8]:
        rows_map = lineage_map.get(stg_table, {})
        if not rows_map:
            continue
        n_rows = manifest[stg_table]['rows']
        sample_rows = rng.sample(range(n_rows), min(2, n_rows))
        for row_idx in sample_rows:
            impact = trace_full_impact(reverse_lineage, stg_table,
                                       f'row_{row_idx}')
            if not impact:
                continue
            all_affected_tables = sorted(impact.keys())
            if len(all_affected_tables) >= 2:
                add_qa('forward_impact', 'row_impact',
                       f"If {stg_table} row_{row_idx} is deleted, "
                       f"which tables (core and mart) are affected?",
                       all_affected_tables, 'list', 'hard',
                       f'Forward trace from mid-pipeline '
                       f'{stg_table}:row_{row_idx}')

    # ══════════════════════════════════════════════════════════
    # 5. VALUE PROPAGATION — target ~40 (medium + hard)
    #    Does data from raw table X flow to mart Y?
    # ══════════════════════════════════════════════════════════
    print('  Generating value_propagation questions...')

    # Precompute which raw tables feed into each mart
    mart_provenance_cache = {}
    for mart_table in mart_tables:
        rows_map = lineage_map.get(mart_table, {})
        if rows_map:
            first_row = next(iter(rows_map))
            prov = trace_full_provenance(lineage_map, mart_table, first_row)
            mart_provenance_cache[mart_table] = set(prov.keys())
        else:
            mart_provenance_cache[mart_table] = set()

    # All raw × mart combinations (14 × 12 = 168 possible, sample ~40)
    all_combos = list(itertools.product(raw_tables, mart_tables))
    rng.shuffle(all_combos)

    # Ensure balanced true/false answers
    true_combos = [(r, m) for r, m in all_combos
                   if r in mart_provenance_cache.get(m, set())]
    false_combos = [(r, m) for r, m in all_combos
                    if r not in mart_provenance_cache.get(m, set())]

    # Take 20 true + 20 false
    selected = rng.sample(true_combos, min(20, len(true_combos))) + \
               rng.sample(false_combos, min(20, len(false_combos)))
    rng.shuffle(selected)

    for raw_table, target_mart in selected:
        reaches = raw_table in mart_provenance_cache.get(target_mart, set())
        same_silo = table_to_silo(raw_table) == table_to_silo(target_mart)
        diff = 'medium' if same_silo else 'hard'

        add_qa('forward_impact', 'value_propagation',
               f"Does data from {raw_table} propagate to {target_mart}?",
               reaches, 'boolean', diff,
               f'Check if {raw_table} is in transitive provenance of '
               f'{target_mart} (same_silo={same_silo})')

    # ══════════════════════════════════════════════════════════
    # 6. CASCADE COUNT — target ~42 (hard)
    #    Count of ALL transitively affected downstream rows
    # ══════════════════════════════════════════════════════════
    print('  Generating cascade_count questions...')

    # 3 rows per raw table × 14 raw tables = 42
    for raw_table in raw_tables:
        n_rows = manifest[raw_table]['rows']
        sample_rows = rng.sample(range(n_rows), min(3, n_rows))
        for row_idx in sample_rows:
            impact = trace_full_impact(reverse_lineage, raw_table,
                                        f'row_{row_idx}')
            total_affected = sum(len(rows) for rows in impact.values())
            if total_affected > 0:
                add_qa('forward_impact', 'cascade_count',
                       f"How many total downstream rows across all tables "
                       f"are affected if {raw_table} row_{row_idx} is deleted?",
                       total_affected, 'integer', 'hard',
                       f'Sum of all downstream rows from '
                       f'{raw_table}:row_{row_idx}')

    # Also: cascade from staging (shallower but still complex)
    for stg_table in stg_tables[:8]:
        n_rows = manifest[stg_table]['rows']
        row_idx = rng.randint(0, n_rows - 1)
        impact = trace_full_impact(reverse_lineage, stg_table,
                                    f'row_{row_idx}')
        total_affected = sum(len(rows) for rows in impact.values())
        if total_affected > 0:
            add_qa('forward_impact', 'cascade_count',
                   f"How many total downstream rows are affected if "
                   f"{stg_table} row_{row_idx} is deleted?",
                   total_affected, 'integer', 'hard',
                   f'Cascade count from mid-pipeline {stg_table}:row_{row_idx}')

    # Cascade by layer: count affected rows per layer
    for raw_table in raw_tables[:5]:
        n_rows = manifest[raw_table]['rows']
        row_idx = rng.randint(0, n_rows - 1)
        impact = trace_full_impact(reverse_lineage, raw_table,
                                    f'row_{row_idx}')
        mart_rows = sum(len(rows) for t, rows in impact.items()
                        if t.startswith('mart_'))
        if mart_rows > 0:
            add_qa('forward_impact', 'cascade_count',
                   f"How many mart-layer rows are affected if {raw_table} "
                   f"row_{row_idx} is deleted?",
                   mart_rows, 'integer', 'hard',
                   f'Mart-only cascade count from {raw_table}:row_{row_idx}')

    # ══════════════════════════════════════════════════════════
    # 7. CROSS-SILO REACHABILITY — target ~40 (hard)
    #    Does data flow between silos?
    # ══════════════════════════════════════════════════════════
    print('  Generating cross_silo_reachability questions...')

    # All 20 ordered silo pairs (5 × 4)
    for src_silo, tgt_silo in itertools.permutations(SILO_NAMES, 2):
        src_raw_tables = SILOS[src_silo]['raw']
        tgt_mart_tables = SILOS[tgt_silo]['mart']

        flows = False
        for tgt_mart in tgt_mart_tables:
            if flows:
                break
            prov_set = mart_provenance_cache.get(tgt_mart, set())
            for src_raw in src_raw_tables:
                if src_raw in prov_set:
                    flows = True
                    break

        add_qa('cross_silo', 'cross_silo_reachability',
               f"Does any data from the {src_silo} silo's raw tables "
               f"flow into the {tgt_silo} silo's mart tables?",
               flows, 'boolean', 'hard',
               f'Check {src_silo} raw → {tgt_silo} mart reachability')

    # Per-table cross-silo questions: does a specific raw
    # table flow to a specific mart in a different silo?
    # Balance True/False explicitly for statistical validity
    cross_silo_combos = [(r, m) for r, m in all_combos
                         if table_to_silo(r) != table_to_silo(m)]
    rng.shuffle(cross_silo_combos)

    true_combos = [(r, m) for r, m in cross_silo_combos
                   if r in mart_provenance_cache.get(m, set())]
    false_combos = [(r, m) for r, m in cross_silo_combos
                    if r not in mart_provenance_cache.get(m, set())]

    # Sample 10 True + 10 False for balanced per-table questions
    per_table_sample = (
        rng.sample(true_combos, min(10, len(true_combos))) +
        rng.sample(false_combos, min(10, len(false_combos)))
    )
    rng.shuffle(per_table_sample)

    for raw_table, mart_table in per_table_sample:
        reaches = raw_table in mart_provenance_cache.get(mart_table, set())
        src_silo = table_to_silo(raw_table)
        tgt_silo = table_to_silo(mart_table)
        add_qa('cross_silo', 'cross_silo_reachability',
               f"Does {raw_table} ({src_silo} silo) contribute data to "
               f"{mart_table} ({tgt_silo} silo)?",
               reaches, 'boolean', 'hard',
               f'Cross-silo: {raw_table} ({src_silo}) → '
               f'{mart_table} ({tgt_silo})')

    # ══════════════════════════════════════════════════════════
    # 8. SHARED SOURCE — target ~40 (medium + hard)
    #    Which raw tables feed BOTH marts?
    # ══════════════════════════════════════════════════════════
    print('  Generating shared_source questions...')

    # Within-silo mart pairs
    for silo_name, layers in SILOS.items():
        silo_marts = layers.get('mart', [])
        for mart_a, mart_b in itertools.combinations(silo_marts, 2):
            raw_a = {t for t in mart_provenance_cache.get(mart_a, set())
                     if t.startswith('raw_')}
            raw_b = {t for t in mart_provenance_cache.get(mart_b, set())
                     if t.startswith('raw_')}
            shared = sorted(raw_a & raw_b)
            add_qa('cross_silo', 'shared_source',
                   f"Which raw source tables feed data into BOTH "
                   f"{mart_a} AND {mart_b}?",
                   shared, 'list', 'medium',
                   f'Intersection of raw provenance for {mart_a} and {mart_b} '
                   f'(within {silo_name})')

    # Cross-silo mart pairs (harder — different business domains)
    cross_mart_pairs = []
    for s1, s2 in itertools.combinations(SILO_NAMES, 2):
        marts1 = SILOS[s1].get('mart', [])
        marts2 = SILOS[s2].get('mart', [])
        for m1, m2 in itertools.product(marts1, marts2):
            cross_mart_pairs.append((m1, m2, s1, s2))

    rng.shuffle(cross_mart_pairs)
    for mart_a, mart_b, silo_a, silo_b in cross_mart_pairs[:25]:
        raw_a = {t for t in mart_provenance_cache.get(mart_a, set())
                 if t.startswith('raw_')}
        raw_b = {t for t in mart_provenance_cache.get(mart_b, set())
                 if t.startswith('raw_')}
        shared = sorted(raw_a & raw_b)
        add_qa('cross_silo', 'shared_source',
               f"Which raw source tables feed data into BOTH "
               f"{mart_a} AND {mart_b}?",
               shared, 'list', 'hard',
               f'Cross-silo shared sources: {mart_a} ({silo_a}) ∩ '
               f'{mart_b} ({silo_b})')

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
    print('Generating Tier 2 QA pairs for Syn-Logistics...')
    generate_qa()
    print('\n✅ Done!')
