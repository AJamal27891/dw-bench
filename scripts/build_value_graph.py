"""Build multi-scale PyG HeteroData with table + row nodes.

Extends the existing schema_graph.pt by adding:
  - row node type with value embeddings
  - (row, belongs_to, table) edges
  - (row, fk_ref, row) edges  
  - (row, derived_from_row, row) edges from lineage_map

The result is saved as value_schema_graph.pt alongside the original.

Usage:
    python scripts/build_value_graph.py
"""
import csv
import json
import hashlib
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

REPO_ROOT = Path(__file__).parent.parent
DS_DIR = REPO_ROOT / 'datasets' / 'syn_logistics'
VALUE_DIR = DS_DIR / 'value_data'


def hash_row(values: list) -> list:
    """Create a fixed-length feature vector from row values via hashing."""
    # Simple approach: hash each value to get a deterministic float feature
    features = []
    for val in values:
        h = hashlib.md5(str(val).encode()).hexdigest()
        features.append(int(h[:8], 16) / 0xFFFFFFFF)  # Normalize to [0, 1]
    return features


def build_value_graph():
    """Build multi-scale graph with table and row nodes."""
    # Load existing schema graph
    schema_graph = torch.load(DS_DIR / 'schema_graph.pt', weights_only=False)
    table_names = schema_graph['table'].table_names
    n_tables = len(table_names)

    print(f'  Schema: {n_tables} tables')

    # Load manifest and lineage map
    with open(VALUE_DIR / '_manifest.json', 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    with open(DS_DIR / 'lineage_map.json', 'r', encoding='utf-8') as f:
        lineage_map = json.load(f)

    # ──────────────────────────────────────────────────────────
    # Build row node registry
    # ──────────────────────────────────────────────────────────
    row_ids = []       # Global row ID strings: "table_name:row_N"
    row_features = []  # Feature vectors for each row
    row_to_idx = {}    # "table_name:row_N" -> global index
    table_to_rows = {} # table_name -> list of global row indices

    # Fixed feature dimension (pad/truncate all rows to this)
    FEAT_DIM = 8

    for table_name in sorted(manifest.keys()):
        info = manifest[table_name]
        csv_path = VALUE_DIR / info['file']

        table_row_indices = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row_idx, row_values in enumerate(reader):
                global_id = f'{table_name}:row_{row_idx}'
                global_idx = len(row_ids)

                row_ids.append(global_id)
                row_to_idx[global_id] = global_idx
                table_row_indices.append(global_idx)

                # Hash row values to fixed-dim feature vector
                feat = hash_row(row_values)
                # Pad/truncate to FEAT_DIM
                feat = (feat + [0.0] * FEAT_DIM)[:FEAT_DIM]
                row_features.append(feat)

        table_to_rows[table_name] = table_row_indices

    n_rows = len(row_ids)
    print(f'  Rows: {n_rows:,} total across {len(table_to_rows)} tables')

    # ──────────────────────────────────────────────────────────
    # Build HeteroData
    # ──────────────────────────────────────────────────────────
    data = HeteroData()

    # Copy table-level data from original schema graph
    data['table'].x = schema_graph['table'].x
    data['table'].table_names = table_names
    data['table'].num_nodes = n_tables

    # Copy existing table-level edges
    if ('table', 'fk_to', 'table') in schema_graph.edge_types:
        data['table', 'fk_to', 'table'].edge_index = \
            schema_graph['table', 'fk_to', 'table'].edge_index
    if ('table', 'derived_from', 'table') in schema_graph.edge_types:
        data['table', 'derived_from', 'table'].edge_index = \
            schema_graph['table', 'derived_from', 'table'].edge_index

    # Row node features
    data['row'].x = torch.tensor(row_features, dtype=torch.float32)
    data['row'].row_ids = row_ids
    data['row'].num_nodes = n_rows

    # ──────────────────────────────────────────────────────────
    # Edge 1: (row, belongs_to, table) — membership
    # ──────────────────────────────────────────────────────────
    table_name_to_idx = {name: i for i, name in enumerate(table_names)}
    bt_src, bt_dst = [], []

    for table_name, row_indices in table_to_rows.items():
        if table_name not in table_name_to_idx:
            continue
        table_idx = table_name_to_idx[table_name]
        for row_idx in row_indices:
            bt_src.append(row_idx)
            bt_dst.append(table_idx)

    data['row', 'belongs_to', 'table'].edge_index = torch.tensor(
        [bt_src, bt_dst], dtype=torch.long)
    print(f'  belongs_to edges: {len(bt_src):,}')

    # ──────────────────────────────────────────────────────────
    # Edge 2: (row, fk_ref, row) — row-level FK references
    # ──────────────────────────────────────────────────────────
    # For fact tables with dim_*_key columns, link fact rows to dim rows
    fk_src, fk_dst = [], []

    for table_name, info in manifest.items():
        if not table_name.startswith('fact_'):
            continue
        csv_path = VALUE_DIR / info['file']
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            # Find FK columns (dim_*_key)
            fk_cols = [(i, col) for i, col in enumerate(header)
                       if col.endswith('_key') and col != 'fact_key']
            if not fk_cols:
                continue

            for row_idx, row_values in enumerate(reader):
                fact_global = row_to_idx.get(f'{table_name}:row_{row_idx}')
                if fact_global is None:
                    continue
                for col_idx, col_name in fk_cols:
                    dim_name = col_name.replace('_key', '')
                    try:
                        dim_row_idx = int(row_values[col_idx])
                    except (ValueError, IndexError):
                        continue
                    dim_global = row_to_idx.get(f'{dim_name}:row_{dim_row_idx}')
                    if dim_global is not None:
                        fk_src.append(fact_global)
                        fk_dst.append(dim_global)

    if fk_src:
        data['row', 'fk_ref', 'row'].edge_index = torch.tensor(
            [fk_src, fk_dst], dtype=torch.long)
    print(f'  fk_ref edges: {len(fk_src):,}')

    # ──────────────────────────────────────────────────────────
    # Edge 3: (row, derived_from_row, row) — row lineage
    # ──────────────────────────────────────────────────────────
    lin_src, lin_dst = [], []

    for target_table, rows_map in lineage_map.items():
        for row_key, row_info in rows_map.items():
            target_global = row_to_idx.get(f'{target_table}:{row_key}')
            if target_global is None:
                continue
            for src_entry in row_info['sources']:
                src_table = src_entry['table']
                for src_row_idx in src_entry['rows']:
                    src_global = row_to_idx.get(f'{src_table}:row_{src_row_idx}')
                    if src_global is not None:
                        lin_src.append(target_global)
                        lin_dst.append(src_global)

    if lin_src:
        data['row', 'derived_from_row', 'row'].edge_index = torch.tensor(
            [lin_src, lin_dst], dtype=torch.long)
    print(f'  derived_from_row edges: {len(lin_src):,}')

    # ──────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────
    out_path = DS_DIR / 'value_schema_graph.pt'
    torch.save(data, out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f'\n  Saved: {out_path} ({size_mb:.1f} MB)')
    print(f'  Node types: {data.node_types}')
    print(f'  Edge types: {data.edge_types}')

    return data


def validate():
    """Validate the value schema graph."""
    graph_path = DS_DIR / 'value_schema_graph.pt'
    if not graph_path.exists():
        print('❌ No value_schema_graph.pt found.')
        return False

    data = torch.load(graph_path, weights_only=False)

    errors = []

    # Check node types
    if 'table' not in data.node_types:
        errors.append('Missing table node type')
    if 'row' not in data.node_types:
        errors.append('Missing row node type')

    # Check edge types
    if ('row', 'belongs_to', 'table') not in data.edge_types:
        errors.append('Missing belongs_to edge type')
    if ('row', 'derived_from_row', 'row') not in data.edge_types:
        errors.append('Missing derived_from_row edge type')

    # Check table-level edges preserved
    if ('table', 'fk_to', 'table') not in data.edge_types:
        errors.append('Missing table fk_to edges')
    if ('table', 'derived_from', 'table') not in data.edge_types:
        errors.append('Missing table derived_from edges')

    # Check row features shape
    n_rows = data['row'].num_nodes
    feat_shape = data['row'].x.shape
    if feat_shape[0] != n_rows:
        errors.append(f'Row features shape {feat_shape} != num_nodes {n_rows}')

    # Check belongs_to covers all rows
    bt_edges = data['row', 'belongs_to', 'table'].edge_index
    unique_rows = bt_edges[0].unique().numel()
    if unique_rows != n_rows:
        errors.append(f'belongs_to covers {unique_rows} rows, expected {n_rows}')

    if errors:
        print(f'❌ {len(errors)} errors:')
        for e in errors:
            print(f'  {e}')
        return False

    print(f'✅ Value graph validated')
    print(f'   Tables: {data["table"].num_nodes}')
    print(f'   Rows: {n_rows:,}')
    print(f'   belongs_to edges: {bt_edges.shape[1]:,}')
    if ('row', 'fk_ref', 'row') in data.edge_types:
        print(f'   fk_ref edges: {data["row", "fk_ref", "row"].edge_index.shape[1]:,}')
    print(f'   derived_from_row edges: '
          f'{data["row", "derived_from_row", "row"].edge_index.shape[1]:,}')
    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build Syn-Logistics value graph')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()

    if args.validate:
        ok = validate()
        exit(0 if ok else 1)

    print('Building value schema graph...')
    build_value_graph()
    print('\nValidating...')
    validate()
    print('\n✅ Done!')
