"""Build Syn-Logistics: A controlled synthetic schema for ablation study.

Generates a deterministic 50-table schema with forced topology properties:
  - 5 disconnected silos (Logistics, HR, Healthcare, E-commerce, Finance)
  - 3+ parallel 4-hop lineage chains per silo
  - Multi-source merge points
  - FK cross-links within silos

Outputs:
  datasets/syn_logistics/schema_graph.pt   — PyG HeteroData
  datasets/syn_logistics/qa_pairs.json     — Deterministic QA pairs
  datasets/syn_logistics/schema_ddl.json   — DDL text for baselines

Usage:
    python scripts/build_syn_logistics.py
"""
import json
import random
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

random.seed(42)

# ──────────────────────────────────────────────────────────────────────
# Domain Dictionary: realistic table names per silo per layer
# ──────────────────────────────────────────────────────────────────────
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

LAYERS = ['raw', 'staging', 'core', 'mart']
OUT_DIR = Path(__file__).parent.parent / 'datasets' / 'syn_logistics'


def build_topology():
    """Build the graph topology with FK and lineage edges."""
    all_tables = []
    silo_membership = {}  # table -> silo name
    layer_membership = {}  # table -> layer name

    # Collect all table names
    for silo_name, layers in SILOS.items():
        for layer, tables in layers.items():
            for t in tables:
                all_tables.append(t)
                silo_membership[t] = silo_name
                layer_membership[t] = layer

    name_to_idx = {n: i for i, n in enumerate(all_tables)}
    n = len(all_tables)

    # ── Lineage edges (derived_from): raw→stg→core→mart ──────────────
    lineage_edges = []  # (source, target) meaning target derives from source

    for silo_name, layers in SILOS.items():
        raw = layers['raw']
        stg = layers['staging']
        core = layers['core']
        mart = layers['mart']

        # raw → staging (1:1 where possible, then fan-in for multi_source)
        for i, s in enumerate(stg):
            # Primary source
            primary = raw[i % len(raw)]
            lineage_edges.append((primary, s))
            # Secondary source for multi_source (every other staging table)
            if i > 0:
                secondary = raw[(i + 1) % len(raw)]
                if secondary != primary:
                    lineage_edges.append((secondary, s))

        # staging → core (fan-in: multiple staging feed each core table)
        for i, c in enumerate(core):
            primary = stg[i % len(stg)]
            lineage_edges.append((primary, c))
            secondary = stg[(i + 1) % len(stg)]
            if secondary != primary:
                lineage_edges.append((secondary, c))

        # core → mart
        for i, m in enumerate(mart):
            primary = core[i % len(core)]
            lineage_edges.append((primary, m))
            if len(core) > 1:
                secondary = core[(i + 1) % len(core)]
                if secondary != primary:
                    lineage_edges.append((secondary, m))

    # ── FK edges (within silos, within layers) ────────────────────────
    fk_edges = []

    for silo_name, layers in SILOS.items():
        core = layers['core']
        # FK: fact tables → dim tables
        dims = [t for t in core if t.startswith('dim_')]
        facts = [t for t in core if t.startswith('fact_')]
        for fact in facts:
            for dim in dims:
                fk_edges.append((fact, dim))

    return all_tables, name_to_idx, silo_membership, layer_membership, \
        lineage_edges, fk_edges


def build_heterodata(all_tables, name_to_idx, silo_membership,
                     lineage_edges, fk_edges):
    """Build PyG HeteroData from topology."""
    n = len(all_tables)

    # Build NetworkX graph for structural features
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    fk_src, fk_dst = [], []
    for child, parent in fk_edges:
        s, d = name_to_idx[child], name_to_idx[parent]
        fk_src.append(s)
        fk_dst.append(d)
        G.add_edge(s, d)

    lin_src, lin_dst = [], []
    for source, target in lineage_edges:
        s, d = name_to_idx[source], name_to_idx[target]
        lin_src.append(s)
        lin_dst.append(d)
        G.add_edge(s, d)

    # Structural features
    G_undir = G.to_undirected()
    deg_c = nx.degree_centrality(G)
    betw = nx.betweenness_centrality(G)
    pr = nx.pagerank(G_undir) if len(G_undir.edges) > 0 else \
        {i: 1.0/n for i in range(n)}
    components = list(nx.connected_components(G_undir))
    silo_id = {}
    for ci, comp in enumerate(components):
        for node in comp:
            silo_id[node] = ci

    features = []
    for i in range(n):
        features.append([
            G.in_degree(i), G.out_degree(i),
            deg_c.get(i, 0), betw.get(i, 0),
            pr.get(i, 0), float(silo_id.get(i, 0)),
        ])

    data = HeteroData()
    data['table'].x = torch.tensor(features, dtype=torch.float32)
    data['table'].num_nodes = n
    data['table'].table_names = all_tables

    if fk_src:
        data['table', 'fk_to', 'table'].edge_index = torch.tensor(
            [fk_src, fk_dst], dtype=torch.long)
    if lin_src:
        data['table', 'derived_from', 'table'].edge_index = torch.tensor(
            [lin_src, lin_dst], dtype=torch.long)

    data.dataset_name = 'syn_logistics'
    data.num_silos = len(components)
    data.has_lineage = len(lin_src) > 0

    return data, components


def build_ddl(all_tables, silo_membership, layer_membership, fk_edges,
              lineage_edges):
    """Generate realistic DDL text for each table."""
    # Column templates per layer
    COLUMN_TEMPLATES = {
        'raw': ['id INT PRIMARY KEY', 'source_system VARCHAR(50)',
                'load_timestamp TIMESTAMP', 'raw_data TEXT',
                'file_name VARCHAR(255)', 'batch_id INT'],
        'staging': ['id INT PRIMARY KEY', 'source_id INT',
                    'validated_flag BOOLEAN', 'cleaned_value VARCHAR(200)',
                    'etl_timestamp TIMESTAMP', 'quality_score DECIMAL(5,2)'],
        'core': ['surrogate_key INT PRIMARY KEY', 'business_key VARCHAR(100)',
                 'effective_date DATE', 'expiry_date DATE',
                 'is_current BOOLEAN', 'hash_diff VARCHAR(64)'],
        'mart': ['metric_key INT PRIMARY KEY', 'dimension_key INT',
                 'measure_value DECIMAL(18,4)', 'period_date DATE',
                 'aggregation_level VARCHAR(20)', 'refresh_timestamp TIMESTAMP'],
    }

    ddl_entries = []
    for table in all_tables:
        layer = layer_membership[table]
        silo = silo_membership[table]
        cols = COLUMN_TEMPLATES[layer]

        # Add FK columns
        fk_cols = []
        for child, parent in fk_edges:
            if child == table:
                fk_cols.append(f'{parent}_key INT REFERENCES {parent}(surrogate_key)')

        all_cols = cols + fk_cols
        col_str = ',\n  '.join(all_cols)
        ddl = f"CREATE TABLE {table} (\n  {col_str}\n);"

        ddl_entries.append({
            'table_name': table,
            'silo': silo,
            'layer': layer,
            'ddl': ddl,
        })

    return ddl_entries


def generate_qa(all_tables, name_to_idx, silo_membership, layer_membership,
                lineage_edges, fk_edges, components):
    """Generate QA pairs using deterministic graph algorithms."""
    n = len(all_tables)
    idx_to_name = {i: name for name, i in name_to_idx.items()}

    # Build adjacency structures
    lineage_fwd = {i: set() for i in range(n)}   # source → targets
    lineage_rev = {i: set() for i in range(n)}   # target → sources
    fk_adj = {i: set() for i in range(n)}
    fk_rev = {i: set() for i in range(n)}

    for src, tgt in lineage_edges:
        s, d = name_to_idx[src], name_to_idx[tgt]
        lineage_fwd[s].add(d)
        lineage_rev[d].add(s)

    for child, parent in fk_edges:
        s, d = name_to_idx[child], name_to_idx[parent]
        fk_adj[s].add(d)
        fk_rev[d].add(s)

    # All-edges adjacency (undirected)
    all_adj = {i: set() for i in range(n)}
    for i in range(n):
        all_adj[i] = lineage_fwd[i] | lineage_rev[i] | fk_adj[i] | fk_rev[i]

    # Component lookup
    comp_map = {}  # node_idx → component_idx
    for ci, comp in enumerate(components):
        for node in comp:
            comp_map[node] = ci

    qa_pairs = []
    qid = 0

    def add_q(subtype, question, answer, answer_type, difficulty, reasoning=''):
        nonlocal qid
        qa_pairs.append({
            'id': f'syn_logistics_{subtype}_{qid:03d}',
            'dataset': 'syn_logistics',
            'type': 'lineage' if subtype in ['forward', 'reverse', 'transitive',
                'combined_impact', 'multi_source'] else 'routing' if subtype in
                ['join_path', 'hop_count', 'direct_fk'] else 'silo',
            'subtype': subtype,
            'question': question,
            'answer': answer,
            'answer_type': answer_type,
            'difficulty': difficulty,
            'reasoning': reasoning,
        })
        qid += 1

    # ── TRANSITIVE (target: n≥20) ────────────────────────────────────
    # For each mart table, find ALL upstream raw tables (multi-hop)
    def find_all_upstream(start_idx, depth=0, max_depth=10):
        """BFS backward through lineage to find all upstream tables."""
        visited = set()
        queue = deque([(start_idx, 0)])
        upstream = []
        while queue:
            node, d = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if d > 0:
                upstream.append((idx_to_name[node], d))
            if d < max_depth:
                for pred in lineage_rev[node]:
                    if pred not in visited:
                        queue.append((pred, d + 1))
        return upstream

    mart_tables = [t for t in all_tables if layer_membership[t] == 'mart']
    for mart in mart_tables:
        midx = name_to_idx[mart]
        upstream = find_all_upstream(midx)
        raw_upstream = [name for name, _ in upstream if layer_membership[name] == 'raw']
        if raw_upstream:
            answer = sorted(raw_upstream)
            add_q('transitive',
                  f'Which raw source tables transitively feed into {mart} through the ETL pipeline?',
                  answer, 'list', 'hard',
                  f'BFS backward from {mart} through derived_from edges to raw layer')

    # For each raw table, find ALL downstream mart tables
    raw_tables = [t for t in all_tables if layer_membership[t] == 'raw']
    for raw in raw_tables:
        ridx = name_to_idx[raw]
        # BFS forward
        visited = set()
        queue = deque([(ridx, 0)])
        downstream = []
        while queue:
            node, d = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if d > 0:
                downstream.append(idx_to_name[node])
            for succ in lineage_fwd[node]:
                if succ not in visited:
                    queue.append((succ, d + 1))
        mart_downstream = sorted([t for t in downstream if layer_membership[t] == 'mart'])
        if mart_downstream:
            add_q('transitive',
                  f'If {raw} is modified, which mart/reporting tables will be affected through the lineage chain?',
                  mart_downstream, 'list', 'hard',
                  f'BFS forward from {raw} through derived_from edges to mart layer')

    # ── MULTI_SOURCE (target: n≥15) ──────────────────────────────────
    for silo_name, layers in SILOS.items():
        for layer_name in ['staging', 'core', 'mart']:
            for table in layers[layer_name]:
                tidx = name_to_idx[table]
                sources = sorted([idx_to_name[s] for s in lineage_rev[tidx]])
                if len(sources) >= 2:
                    add_q('multi_source',
                          f'Which tables directly feed into {table}?',
                          sources, 'list', 'medium',
                          f'Direct predecessors in derived_from edges')

    # ── CONNECTED (target: n≥20) ─────────────────────────────────────
    comp_tables = {}
    for ci, comp in enumerate(components):
        comp_tables[ci] = sorted([idx_to_name[node] for node in comp])

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            t1 = comp_tables[i][0]
            t2 = comp_tables[j][0]
            add_q('connected',
                  f'Are {t1} and {t2} in the same connected component?',
                  'no', 'boolean', 'medium',
                  f'Component {i} and {j} are disconnected')

    # Same component pairs
    for ci, comp in enumerate(components):
        tables = comp_tables[ci]
        if len(tables) >= 2:
            pairs = [(tables[a], tables[b]) for a in range(min(3, len(tables)))
                     for b in range(a+1, min(5, len(tables)))]
            for t1, t2 in pairs[:3]:
                add_q('connected',
                      f'Are {t1} and {t2} in the same connected component?',
                      'yes', 'boolean', 'easy',
                      f'Both in component {ci}')

    # ── COUNT (target: n≥20) ──────────────────────────────────────────
    add_q('count',
          'How many disconnected components (data silos) exist in the schema?',
          str(len(components)), 'number', 'easy',
          f'{len(components)} connected components')

    for ci, comp in enumerate(components):
        tables = comp_tables[ci]
        silo = silo_membership[tables[0]]
        add_q('count',
              f'How many tables are in the {silo} data silo?',
              str(len(tables)), 'number', 'easy',
              f'Component {ci} has {len(tables)} tables')

    # Per-layer counts per silo
    for silo_name, layers_dict in SILOS.items():
        for lyr in LAYERS:
            count = len(layers_dict[lyr])
            add_q('count',
                  f'How many {lyr}-layer tables are in the {silo_name} domain?',
                  str(count), 'number', 'easy',
                  f'{count} {lyr} tables in {silo_name}')

    # ── FULL_ENUMERATION (target: n≥20) ──────────────────────────────
    for ci, comp in enumerate(components):
        tables = comp_tables[ci]
        silo = silo_membership[tables[0]]
        add_q('full_enumeration',
              f'List all tables in the {silo} data silo.',
              sorted(tables), 'list', 'medium',
              f'All tables in component {ci}')

    # Per-layer enumeration for ALL layers
    for silo_name, layers_dict in SILOS.items():
        for layer_name in LAYERS:
            tables = sorted(layers_dict[layer_name])
            add_q('full_enumeration',
                  f'List all {layer_name} tables in the {silo_name} domain.',
                  tables, 'list', 'easy',
                  f'{layer_name} tables in {silo_name}')

    # ── FORWARD lineage ──────────────────────────────────────────────
    for src, tgt in lineage_edges[:20]:
        add_q('forward',
              f'Which tables are directly derived from {src}?',
              sorted([idx_to_name[d] for d in lineage_fwd[name_to_idx[src]]]),
              'list', 'easy',
              f'Direct successors of {src} in derived_from')

    # ── REVERSE lineage ──────────────────────────────────────────────
    for src, tgt in lineage_edges[:20]:
        sources = sorted([idx_to_name[s] for s in lineage_rev[name_to_idx[tgt]]])
        if sources:
            add_q('reverse',
                  f'What are the direct source tables for {tgt}?',
                  sources, 'list', 'easy',
                  f'Direct predecessors of {tgt}')

    # ── ISOLATION (target: n≥20) ──────────────────────────────────────
    for ci, comp in enumerate(components):
        tables = comp_tables[ci]
        silo = silo_membership[tables[0]]
        other_tables = [t for t in all_tables if t not in tables]
        if other_tables:
            sample = random.sample(other_tables, min(5, len(other_tables)))
            for other in sample:
                add_q('isolation',
                      f'Is {tables[0]} connected to {other} through any FK or lineage relationship?',
                      'no', 'boolean', 'medium',
                      f'{tables[0]} is in {silo}, {other} is in a different silo')

    # ── HOP_COUNT ────────────────────────────────────────────────────
    # Shortest paths within silos
    for silo_name, layers in SILOS.items():
        raw = layers['raw']
        mart = layers['mart']
        for r in raw:
            for m in mart:
                # BFS shortest path
                ri, mi = name_to_idx[r], name_to_idx[m]
                visited = {ri}
                queue = deque([(ri, [r])])
                found = False
                while queue and not found:
                    node, path = queue.popleft()
                    for nb in all_adj[node]:
                        if nb not in visited:
                            new_path = path + [idx_to_name[nb]]
                            if nb == mi:
                                hops = len(new_path) - 1
                                add_q('hop_count',
                                      f'What is the minimum number of hops between {r} and {m}?',
                                      str(hops), 'number', 'hard',
                                      f'Shortest path: {" → ".join(new_path)}')
                                found = True
                                break
                            visited.add(nb)
                            queue.append((nb, new_path))

    # ── DIRECT_FK ────────────────────────────────────────────────────
    for child, parent in fk_edges:
        add_q('direct_fk',
              f'Does {child} have a direct foreign key to {parent}?',
              'yes', 'boolean', 'easy',
              f'FK edge exists: {child} → {parent}')

    # Non-FK pairs
    for silo_name, layers in SILOS.items():
        raw = layers['raw']
        core = layers['core']
        dims = [t for t in core if t.startswith('dim_')]
        for r in raw[:2]:
            for d in dims[:2]:
                if (r, d) not in fk_edges:
                    add_q('direct_fk',
                          f'Does {r} have a direct foreign key to {d}?',
                          'no', 'boolean', 'easy',
                          f'No direct FK between {r} and {d}')

    # ── COMBINED_IMPACT (target: n≥20) ────────────────────────────────
    # From all raw AND staging tables (more starting points = more questions)
    impact_sources = [t for t in all_tables
                      if layer_membership[t] in ('raw', 'staging')]
    for src in impact_sources:
        sidx = name_to_idx[src]
        # Forward lineage
        visited = set()
        queue = deque([sidx])
        affected = set()
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            affected.add(node)
            for succ in lineage_fwd[node]:
                queue.append(succ)
        # Now add FK-connected from affected
        fk_affected = set()
        for a in affected:
            for fk in fk_adj[a] | fk_rev[a]:
                fk_affected.add(fk)
        total = sorted(set(idx_to_name[a] for a in (affected | fk_affected)) - {src})
        if total and len(total) >= 3:  # Only meaningful combined impacts
            add_q('combined_impact',
                  f'If {src} is corrupted, which tables are affected through both lineage AND foreign key dependencies?',
                  total, 'list', 'hard',
                  f'Lineage forward from {src} + FK expansion')

    # ── MEMBERSHIP ───────────────────────────────────────────────────
    for ci, comp in enumerate(components):
        tables = comp_tables[ci]
        silo = silo_membership[tables[0]]
        for t in tables[:4]:
            add_q('membership',
                  f'Which data silo does {t} belong to?',
                  silo, 'string', 'medium',
                  f'{t} is in the {silo} silo')

    # ── JOIN_PATH (target: n≥20) ──────────────────────────────────────
    for silo_name, layers_dict in SILOS.items():
        core = layers_dict['core']
        facts = [t for t in core if t.startswith('fact_')]
        dims = [t for t in core if t.startswith('dim_')]
        for f in facts:
            for d in dims:
                add_q('join_path',
                      f'What is the shortest join path between {f} and {d}?',
                      [f, d], 'list', 'easy',
                      f'Direct FK: {f} → {d}')
        # Cross-dim paths (dim to dim via shared fact)
        for i, d1 in enumerate(dims):
            for d2 in dims[i+1:]:
                if facts:  # Path: d1 ← fact → d2
                    add_q('join_path',
                          f'What is the shortest join path between {d1} and {d2}?',
                          [d1, facts[0], d2], 'list', 'medium',
                          f'Via shared fact table: {d1} ← {facts[0]} → {d2}')

    return qa_pairs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Building Syn-Logistics topology...')
    all_tables, name_to_idx, silo_membership, layer_membership, \
        lineage_edges, fk_edges = build_topology()

    print(f'  Tables: {len(all_tables)}')
    print(f'  Lineage edges: {len(lineage_edges)}')
    print(f'  FK edges: {len(fk_edges)}')
    print(f'  Silos: {len(SILOS)}')

    print('Building HeteroData...')
    data, components = build_heterodata(
        all_tables, name_to_idx, silo_membership, lineage_edges, fk_edges)

    print(f'  Components: {len(components)}')
    for ci, comp in enumerate(components):
        names = sorted([all_tables[i] for i in comp])
        silo = silo_membership[names[0]]
        print(f'    Silo {ci} ({silo}): {len(comp)} tables')

    print('Generating DDL...')
    ddl_entries = build_ddl(all_tables, silo_membership, layer_membership,
                            fk_edges, lineage_edges)

    print('Generating QA pairs...')
    qa_pairs = generate_qa(all_tables, name_to_idx, silo_membership,
                           layer_membership, lineage_edges, fk_edges, components)

    # Count by subtype
    from collections import Counter
    by_subtype = Counter(q['subtype'] for q in qa_pairs)
    by_difficulty = Counter(q['difficulty'] for q in qa_pairs)
    print(f'\n  Total questions: {len(qa_pairs)}')
    print(f'  By subtype:')
    for st in sorted(by_subtype):
        flag = ' ✓' if by_subtype[st] >= 20 else ' ← needs more'
        print(f'    {st:20s}: {by_subtype[st]:3d}{flag}')
    print(f'  By difficulty: {dict(by_difficulty)}')

    # Save
    torch.save(data, OUT_DIR / 'schema_graph.pt')
    print(f'\n  Saved: {OUT_DIR / "schema_graph.pt"}')

    json.dump(qa_pairs, open(OUT_DIR / 'qa_pairs.json', 'w', encoding='utf-8'),
              indent=2)
    print(f'  Saved: {OUT_DIR / "qa_pairs.json"} ({len(qa_pairs)} questions)')

    json.dump(ddl_entries, open(OUT_DIR / 'schema_ddl.json', 'w', encoding='utf-8'),
              indent=2)
    print(f'  Saved: {OUT_DIR / "schema_ddl.json"} ({len(ddl_entries)} tables)')

    print('\n✅ Syn-Logistics dataset built successfully!')


if __name__ == '__main__':
    main()
