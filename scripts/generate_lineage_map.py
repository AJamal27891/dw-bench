"""Generate row-level lineage map for Syn-Logistics.

For each derived_from edge (source_table → target_table) in the schema graph,
creates row-to-row mappings: each target row maps to 1+ source rows.

The lineage_map enables Tier 2 questions like:
  "Which raw rows contributed to this mart aggregation?"
  "If raw row 42 is deleted, which downstream rows are affected?"

Usage:
    python scripts/generate_lineage_map.py
    python scripts/generate_lineage_map.py --validate
"""
import argparse
import json
import random
from pathlib import Path

import torch

random.seed(42)

REPO_ROOT = Path(__file__).parent.parent
DS_DIR = REPO_ROOT / 'datasets' / 'syn_logistics'
VALUE_DIR = DS_DIR / 'value_data'


def load_derived_from_edges():
    """Load derived_from edges from schema_graph.pt + cross-silo edges."""
    data = torch.load(DS_DIR / 'schema_graph.pt', weights_only=False)
    table_names = data['table'].table_names

    edges = []  # (source_table, target_table)
    if ('table', 'derived_from', 'table') in data.edge_types:
        ei = data['table', 'derived_from', 'table'].edge_index
        for j in range(ei.shape[1]):
            src = table_names[ei[0, j].item()]
            tgt = table_names[ei[1, j].item()]
            edges.append((src, tgt))

    # ── Cross-silo "Conformed Dimensions" edges ──────────────
    # Simulate Enterprise Data Lake integrations:
    #   Healthcare stays FULLY ISOLATED (for valid False answers)
    #   HR, Logistics, Finance form a triangle
    #   E-commerce feeds into Logistics
    CROSS_SILO_EDGES = [
        # HR → Logistics: employee/driver records for shipment validation
        ('stg_employees_cleaned', 'stg_shipments_validated'),
        # Logistics → Finance: carrier cost data for financial reconciliation
        ('dim_carrier', 'stg_ledger_reconciled'),
        # Finance → HR: account classifications inform compensation normalization
        ('dim_account', 'stg_compensation_normalized'),
        # E-commerce → Logistics: payment validation feeds freight matching
        ('stg_payments_validated', 'stg_freight_reconciled'),
    ]

    n_original = len(edges)
    edges.extend(CROSS_SILO_EDGES)
    print(f'  Cross-silo edges injected: {len(CROSS_SILO_EDGES)} '
          f'(total: {n_original} + {len(CROSS_SILO_EDGES)} = {len(edges)})')

    return table_names, edges


def load_manifest():
    """Load the value data manifest to get row counts."""
    with open(VALUE_DIR / '_manifest.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_lineage_map():
    """Generate row-to-row lineage mappings for all derived_from edges.

    Strategy per layer transition:
    - raw → staging: each staging row maps to 1-3 raw rows (fan-in from cleaning/merging)
    - staging → core: each core row maps to 1-5 staging rows (joins/enrichment)
    - core → mart: each mart row aggregates from 5-50 core rows (aggregation)
    """
    table_names, edges = load_derived_from_edges()
    manifest = load_manifest()

    print(f'  Schema: {len(table_names)} tables, {len(edges)} derived_from edges')

    # Group edges by target to handle multi-source tables
    target_sources = {}  # target_table -> [source_tables]
    for src, tgt in edges:
        if tgt not in target_sources:
            target_sources[tgt] = []
        target_sources[tgt].append(src)

    lineage_map = {}
    total_mappings = 0

    for target_table, source_tables in sorted(target_sources.items()):
        if target_table not in manifest:
            print(f'  ⚠️  Target {target_table} not in manifest, skipping')
            continue

        target_rows = manifest[target_table]['rows']
        target_layer = manifest[target_table]['layer']
        rng = random.Random(hash(target_table) & 0xFFFFFFFF)

        table_map = {}

        for row_idx in range(target_rows):
            sources = []
            for src_table in source_tables:
                if src_table not in manifest:
                    continue
                src_rows = manifest[src_table]['rows']

                # Determine how many source rows this target row maps to
                if target_layer == 'staging':
                    # Staging rows derive from 1-3 raw rows
                    n_source = rng.randint(1, 3)
                elif target_layer == 'core':
                    if target_table.startswith('fact_'):
                        # Fact rows reference 1-2 rows per source
                        n_source = rng.randint(1, 2)
                    else:
                        # Dim rows derive from 1-3 staging rows
                        n_source = rng.randint(1, 3)
                elif target_layer == 'mart':
                    # Mart rows aggregate from 5-25 core rows
                    n_source = rng.randint(5, 25)
                else:
                    n_source = rng.randint(1, 3)

                # Pick source row indices (with replacement allowed for aggregations)
                if n_source >= src_rows:
                    source_row_ids = list(range(src_rows))
                else:
                    source_row_ids = sorted(rng.sample(range(src_rows), n_source))

                sources.append({
                    'table': src_table,
                    'rows': source_row_ids,
                })

            table_map[f'row_{row_idx}'] = {'sources': sources}
            total_mappings += sum(len(s['rows']) for s in sources)

        lineage_map[target_table] = table_map
        n_rows = len(table_map)
        print(f'    {target_table:40s}: {n_rows:>6,} rows mapped '
              f'← {", ".join(source_tables)}')

    # Save
    out_path = DS_DIR / 'lineage_map.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(lineage_map, f, indent=None)  # No indent — file would be huge

    print(f'\n  Total: {len(lineage_map)} target tables, {total_mappings:,} row-level mappings')
    print(f'  Saved to: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)')

    return lineage_map


def validate(lineage_map: dict = None):
    """Validate lineage map against schema graph and value data."""
    if lineage_map is None:
        lm_path = DS_DIR / 'lineage_map.json'
        if not lm_path.exists():
            print('❌ No lineage_map.json found. Run generation first.')
            return False
        with open(lm_path, 'r', encoding='utf-8') as f:
            lineage_map = json.load(f)

    table_names, edges = load_derived_from_edges()
    manifest = load_manifest()

    errors = []

    # Check all derived_from target tables have entries
    target_tables = set(tgt for _, tgt in edges)
    missing_targets = target_tables - set(lineage_map.keys())
    for m in missing_targets:
        if m in manifest:  # Only error if we have data for it
            errors.append(f'Missing target table in lineage_map: {m}')

    # Check row counts match
    for table_name, rows_map in lineage_map.items():
        if table_name not in manifest:
            errors.append(f'{table_name} in lineage_map but not in manifest')
            continue
        expected_rows = manifest[table_name]['rows']
        actual_rows = len(rows_map)
        if actual_rows != expected_rows:
            errors.append(f'{table_name}: {actual_rows} mapped rows != '
                          f'{expected_rows} CSV rows')

    # Check source row references are in range
    sample_tables = list(lineage_map.keys())[:5]
    for table_name in sample_tables:
        rows_map = lineage_map[table_name]
        for row_key, row_info in list(rows_map.items())[:10]:
            for src in row_info['sources']:
                src_table = src['table']
                if src_table not in manifest:
                    errors.append(f'{table_name}.{row_key} references '
                                  f'unknown source {src_table}')
                    continue
                max_row = manifest[src_table]['rows']
                for row_id in src['rows']:
                    if row_id >= max_row:
                        errors.append(f'{table_name}.{row_key} references '
                                      f'{src_table} row {row_id} >= {max_row}')

    if errors:
        print(f'❌ {len(errors)} validation errors:')
        for e in errors[:20]:
            print(f'  {e}')
        return False

    total_rows = sum(len(v) for v in lineage_map.values())
    print(f'✅ Lineage map validated: {len(lineage_map)} tables, '
          f'{total_rows:,} rows mapped')
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate Syn-Logistics lineage map')
    parser.add_argument('--validate', action='store_true',
                        help='Validate existing lineage map')
    args = parser.parse_args()

    if args.validate:
        ok = validate()
        exit(0 if ok else 1)

    print('Generating lineage map...')
    lineage_map = generate_lineage_map()
    print('\nValidating...')
    validate(lineage_map)
    print('\n✅ Done!')


if __name__ == '__main__':
    main()
