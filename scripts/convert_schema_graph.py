"""Convert data warehouse schemas into PyG HeteroData Schema Track graphs.

Each table becomes a node. Each FK relationship becomes a directed edge.
Node features include structural properties (degree, betweenness centrality)
computed via networkx.

Supported datasets:
    - adventureworks (OLTP + DW combined)
    - tpc-ds (generated from DDL)
    - tpc-di (from ETL DAG metadata)

Usage:
    python convert_schema_graph.py --dataset adventureworks
    python convert_schema_graph.py --dataset adventureworks --data-dir ../datasets
"""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from torch_geometric.data import HeteroData


def load_adventureworks(data_dir: Path) -> HeteroData:
    """Convert AdventureWorks CSVs into a Schema Track HeteroData.

    Combines both OLTP (AdventureWorks) and DW (AdventureWorksDW) schemas
    into a single graph with two database subgraphs.

    Node type: 'table'
    Edge types: ('table', 'fk_to', 'table'), ('table', 'derived_from', 'table')
    """
    import pandas as pd

    # ── Load tables ──────────────────────────────────────────────────
    oltp_tables = pd.read_csv(data_dir / 'AdventureWorks_Tables.csv')
    dw_tables = pd.read_csv(data_dir / 'AdventureWorksDW_Tables.csv')

    # Filter to base tables only (exclude views)
    oltp_base = oltp_tables[
        oltp_tables['TABLE_TYPE'] == 'BASE TABLE'
    ].copy()
    dw_base = dw_tables[
        dw_tables['TABLE_TYPE'] == 'BASE TABLE'
    ].copy()

    # Prefix table names with database to avoid collisions
    oltp_base['full_name'] = 'OLTP.' + oltp_base['TABLE_SCHEMA'] + '.' + \
        oltp_base['TABLE_NAME']
    dw_base['full_name'] = 'DW.dbo.' + dw_base['TABLE_NAME']

    all_tables = pd.concat([
        oltp_base[['full_name', 'TABLE_NAME', 'TABLE_SCHEMA']],
        dw_base[['full_name', 'TABLE_NAME']].assign(TABLE_SCHEMA='dbo'),
    ], ignore_index=True)

    table_names = all_tables['full_name'].tolist()
    name_to_idx = {name: i for i, name in enumerate(table_names)}
    num_tables = len(table_names)

    # ── Load FK relationships ────────────────────────────────────────
    oltp_rels = pd.read_csv(data_dir / 'AdventureWorks_Relationships.csv')
    dw_rels = pd.read_csv(data_dir / 'AdventureWorksDW_Relationships.csv')

    # Build networkx graph for structural features
    G = nx.DiGraph()
    G.add_nodes_from(range(num_tables))

    fk_src, fk_dst = [], []

    # Process OLTP FK edges
    oltp_table_lookup = dict(zip(
        oltp_base['TABLE_NAME'],
        oltp_base['full_name'],
    ))
    for _, row in oltp_rels.iterrows():
        child = oltp_table_lookup.get(row['ChildTable'])
        parent = oltp_table_lookup.get(row['ParentTable'])
        if child and parent and child in name_to_idx and parent in name_to_idx:
            src_idx = name_to_idx[child]
            dst_idx = name_to_idx[parent]
            fk_src.append(src_idx)
            fk_dst.append(dst_idx)
            G.add_edge(src_idx, dst_idx)

    # Process DW FK edges
    dw_table_lookup = {
        row['TABLE_NAME']: f"DW.dbo.{row['TABLE_NAME']}"
        for _, row in dw_base.iterrows()
    }
    for _, row in dw_rels.iterrows():
        child = dw_table_lookup.get(row['ChildTable'])
        parent = dw_table_lookup.get(row['ParentTable'])
        if child and parent and child in name_to_idx and parent in name_to_idx:
            src_idx = name_to_idx[child]
            dst_idx = name_to_idx[parent]
            fk_src.append(src_idx)
            fk_dst.append(dst_idx)
            G.add_edge(src_idx, dst_idx)

    # ── Load ETL lineage edges (DERIVED_FROM) ────────────────────────
    lineage_src, lineage_dst = [], []
    lineage_file = data_dir / 'lineage_edges.json'
    if lineage_file.exists():
        lineage_edges = json.load(open(lineage_file, 'r', encoding='utf-8'))
        for edge in lineage_edges:
            src = edge.get('oltp_source', '')  # OLTP source table
            dst = edge.get('dw_table', '')     # DW target table
            if src in name_to_idx and dst in name_to_idx:
                lineage_src.append(name_to_idx[src])
                lineage_dst.append(name_to_idx[dst])
        print(f"  Loaded {len(lineage_src)} lineage edges from {lineage_file}")
    else:
        print(f"  ⚠️  No lineage_edges.json found — run build_lineage.py first")

    # ── Compute structural node features ─────────────────────────────
    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)

    # Compute pagerank on undirected version for additional signal
    G_undir = G.to_undirected()
    pagerank = nx.pagerank(G_undir) if len(G_undir.edges) > 0 else \
        {i: 1.0 / num_tables for i in range(num_tables)}

    # Connected component (silo) ID
    components = list(nx.connected_components(G_undir))
    silo_id = {}
    for comp_idx, comp in enumerate(components):
        for node in comp:
            silo_id[node] = comp_idx

    features = []
    for i in range(num_tables):
        features.append([
            G.in_degree(i),
            G.out_degree(i),
            degree_centrality.get(i, 0.0),
            betweenness.get(i, 0.0),
            pagerank.get(i, 0.0),
            float(silo_id.get(i, 0)),
        ])

    # ── Build HeteroData ─────────────────────────────────────────────
    data = HeteroData()

    data['table'].x = torch.tensor(features, dtype=torch.float32)
    data['table'].num_nodes = num_tables
    data['table'].table_names = table_names

    if fk_src:
        data['table', 'fk_to', 'table'].edge_index = torch.tensor(
            [fk_src, fk_dst], dtype=torch.long,
        )

    if lineage_src:
        data['table', 'derived_from', 'table'].edge_index = torch.tensor(
            [lineage_src, lineage_dst], dtype=torch.long,
        )

    # Store metadata for downstream use
    data.dataset_name = 'adventureworks'
    data.num_silos = len(components)
    data.has_lineage = len(lineage_src) > 0

    return data


def load_tpc_ds(data_dir: Path) -> HeteroData:
    """Convert TPC-DS schema into a Schema Track HeteroData.

    TPC-DS snowflake schema: 7 fact tables + 17 dimension tables.
    FK relationships are derived from the TPC-DS specification.
    """
    # TPC-DS table definitions (from TPC-DS v4.0.0 spec)
    fact_tables = [
        'store_sales', 'store_returns', 'catalog_sales',
        'catalog_returns', 'web_sales', 'web_returns', 'inventory',
    ]
    dim_tables = [
        'date_dim', 'time_dim', 'item', 'customer', 'customer_address',
        'customer_demographics', 'household_demographics', 'store',
        'promotion', 'warehouse', 'ship_mode', 'reason', 'income_band',
        'call_center', 'catalog_page', 'web_site', 'web_page',
    ]

    all_tables = fact_tables + dim_tables
    name_to_idx = {name: i for i, name in enumerate(all_tables)}
    num_tables = len(all_tables)

    # TPC-DS FK relationships — COMPLETE list from TPC-DS v4.0.0 spec.
    # Every _sk column in a fact table maps to the PK of a dimension.
    # Verified against: github.com/gregrahn/tpcds-kit/blob/master/tools/tpcds.sql
    fk_edges = [
        # store_sales FKs (9 edges)
        ('store_sales', 'date_dim'), ('store_sales', 'time_dim'),
        ('store_sales', 'item'), ('store_sales', 'customer'),
        ('store_sales', 'customer_demographics'),
        ('store_sales', 'household_demographics'),
        ('store_sales', 'customer_address'), ('store_sales', 'store'),
        ('store_sales', 'promotion'),
        # store_returns FKs (9 edges)
        ('store_returns', 'date_dim'), ('store_returns', 'time_dim'),
        ('store_returns', 'item'), ('store_returns', 'customer'),
        ('store_returns', 'customer_demographics'),
        ('store_returns', 'household_demographics'),
        ('store_returns', 'customer_address'),
        ('store_returns', 'store'), ('store_returns', 'reason'),
        # catalog_sales FKs (12 edges)
        ('catalog_sales', 'date_dim'), ('catalog_sales', 'time_dim'),
        ('catalog_sales', 'item'), ('catalog_sales', 'customer'),
        ('catalog_sales', 'customer_demographics'),
        ('catalog_sales', 'household_demographics'),
        ('catalog_sales', 'customer_address'),
        ('catalog_sales', 'call_center'),
        ('catalog_sales', 'catalog_page'), ('catalog_sales', 'ship_mode'),
        ('catalog_sales', 'warehouse'), ('catalog_sales', 'promotion'),
        # catalog_returns FKs (12 edges)
        ('catalog_returns', 'date_dim'), ('catalog_returns', 'time_dim'),
        ('catalog_returns', 'item'), ('catalog_returns', 'customer'),
        ('catalog_returns', 'customer_demographics'),
        ('catalog_returns', 'household_demographics'),
        ('catalog_returns', 'customer_address'),
        ('catalog_returns', 'reason'), ('catalog_returns', 'call_center'),
        ('catalog_returns', 'catalog_page'),
        ('catalog_returns', 'ship_mode'), ('catalog_returns', 'warehouse'),
        # web_sales FKs (12 edges)
        ('web_sales', 'date_dim'), ('web_sales', 'time_dim'),
        ('web_sales', 'item'), ('web_sales', 'customer'),
        ('web_sales', 'customer_demographics'),
        ('web_sales', 'household_demographics'),
        ('web_sales', 'customer_address'), ('web_sales', 'web_site'),
        ('web_sales', 'web_page'), ('web_sales', 'ship_mode'),
        ('web_sales', 'warehouse'), ('web_sales', 'promotion'),
        # web_returns FKs (9 edges)
        ('web_returns', 'date_dim'), ('web_returns', 'time_dim'),
        ('web_returns', 'item'), ('web_returns', 'customer'),
        ('web_returns', 'customer_demographics'),
        ('web_returns', 'household_demographics'),
        ('web_returns', 'customer_address'),
        ('web_returns', 'reason'), ('web_returns', 'web_page'),
        # inventory FKs (3 edges)
        ('inventory', 'date_dim'), ('inventory', 'item'),
        ('inventory', 'warehouse'),
        # dim → dim FKs (4 edges)
        ('customer', 'customer_address'),
        ('customer', 'customer_demographics'),
        ('customer', 'household_demographics'),
        ('household_demographics', 'income_band'),
    ]

    G = nx.DiGraph()
    G.add_nodes_from(range(num_tables))
    fk_src, fk_dst = [], []
    for child, parent in fk_edges:
        if child in name_to_idx and parent in name_to_idx:
            s, d = name_to_idx[child], name_to_idx[parent]
            fk_src.append(s)
            fk_dst.append(d)
            G.add_edge(s, d)

    # Structural features
    degree_c = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    G_undir = G.to_undirected()
    pagerank = nx.pagerank(G_undir) if len(G_undir.edges) > 0 else \
        {i: 1.0 / num_tables for i in range(num_tables)}
    components = list(nx.connected_components(G_undir))
    silo_id = {}
    for ci, comp in enumerate(components):
        for n in comp:
            silo_id[n] = ci

    features = []
    for i in range(num_tables):
        features.append([
            G.in_degree(i), G.out_degree(i),
            degree_c.get(i, 0.0), betweenness.get(i, 0.0),
            pagerank.get(i, 0.0), float(silo_id.get(i, 0)),
        ])

    data = HeteroData()
    data['table'].x = torch.tensor(features, dtype=torch.float32)
    data['table'].num_nodes = num_tables
    data['table'].table_names = all_tables

    if fk_src:
        data['table', 'fk_to', 'table'].edge_index = torch.tensor(
            [fk_src, fk_dst], dtype=torch.long,
        )

    data.dataset_name = 'tpc-ds'
    data.num_silos = len(components)
    data.has_lineage = False

    return data


def load_tpc_di(data_dir: Path) -> HeteroData:
    """Convert TPC-DI CSVs + lineage JSON into a Schema Track HeteroData.

    TPC-DI is a brokerage ETL benchmark with explicit source→DW lineage.
    Source files (SRC.*) and DW tables are combined into one graph.
    """
    import pandas as pd

    # ── Load tables ──────────────────────────────────────────────────
    tables_df = pd.read_csv(data_dir / 'TPCDI_Tables.csv')
    dw_table_names = tables_df['TABLE_NAME'].tolist()

    # Add source file nodes (from lineage edges)
    lineage_file = data_dir / 'lineage_edges.json'
    source_nodes = set()
    lineage_edges_raw = []
    if lineage_file.exists():
        lineage_edges_raw = json.load(open(lineage_file, 'r', encoding='utf-8'))
        for edge in lineage_edges_raw:
            src = edge.get('source', '')
            if src.startswith('SRC.'):
                source_nodes.add(src)

    all_tables = list(source_nodes) + dw_table_names
    name_to_idx = {name: i for i, name in enumerate(all_tables)}
    num_tables = len(all_tables)

    # ── Build FK edges ───────────────────────────────────────────────
    rels_df = pd.read_csv(data_dir / 'TPCDI_Relationships.csv')
    G = nx.DiGraph()
    G.add_nodes_from(range(num_tables))
    fk_src, fk_dst = [], []
    for _, row in rels_df.iterrows():
        child = row['ChildTable']
        parent = row['ParentTable']
        if child in name_to_idx and parent in name_to_idx:
            s, d = name_to_idx[child], name_to_idx[parent]
            fk_src.append(s)
            fk_dst.append(d)
            G.add_edge(s, d)

    # ── Load ETL lineage edges ───────────────────────────────────────
    lineage_src, lineage_dst = [], []
    for edge in lineage_edges_raw:
        src = edge.get('source', '')
        tgt = edge.get('target', '')
        if src in name_to_idx and tgt in name_to_idx:
            lineage_src.append(name_to_idx[src])
            lineage_dst.append(name_to_idx[tgt])
            G.add_edge(name_to_idx[src], name_to_idx[tgt])
    print(f"  Loaded {len(lineage_src)} lineage edges")

    # ── Structural features ──────────────────────────────────────────
    degree_c = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    G_undir = G.to_undirected()
    pagerank = nx.pagerank(G_undir) if len(G_undir.edges) > 0 else \
        {i: 1.0 / num_tables for i in range(num_tables)}
    components = list(nx.connected_components(G_undir))
    silo_id = {}
    for ci, comp in enumerate(components):
        for n in comp:
            silo_id[n] = ci

    features = []
    for i in range(num_tables):
        features.append([
            G.in_degree(i), G.out_degree(i),
            degree_c.get(i, 0.0), betweenness.get(i, 0.0),
            pagerank.get(i, 0.0), float(silo_id.get(i, 0)),
        ])

    data = HeteroData()
    data['table'].x = torch.tensor(features, dtype=torch.float32)
    data['table'].num_nodes = num_tables
    data['table'].table_names = all_tables

    if fk_src:
        data['table', 'fk_to', 'table'].edge_index = torch.tensor(
            [fk_src, fk_dst], dtype=torch.long,
        )
    if lineage_src:
        data['table', 'derived_from', 'table'].edge_index = torch.tensor(
            [lineage_src, lineage_dst], dtype=torch.long,
        )

    data.dataset_name = 'tpc-di'
    data.num_silos = len(components)
    data.has_lineage = len(lineage_src) > 0

    return data


def print_graph_summary(data: HeteroData) -> None:
    """Print a summary of the converted Schema Track graph."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: {data.dataset_name}")
    print(f"{'=' * 60}")
    print(f"Tables (nodes):    {data['table'].num_nodes}")
    print(f"Node features:     {data['table'].x.shape[1]} "
          f"(in_deg, out_deg, degree_c, betweenness, pagerank, silo_id)")

    for edge_type in data.edge_types:
        ei = data[edge_type].edge_index
        print(f"Edge type '{edge_type[1]}': {ei.shape[1]} edges")

    print(f"Connected components (silos): {data.num_silos}")
    print(f"Has lineage edges: {data.has_lineage}")

    # Show top-5 most connected tables
    x = data['table'].x
    in_deg = x[:, 0]
    total_deg = x[:, 0] + x[:, 1]
    top5 = torch.argsort(total_deg, descending=True)[:5]
    print(f"\nTop 5 most connected tables:")
    for idx in top5:
        name = data['table'].table_names[idx]
        print(f"  {name}: in={int(in_deg[idx])}, "
              f"total={int(total_deg[idx])}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Convert DW schemas to Schema Track HeteroData',
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['adventureworks', 'tpc-ds', 'tpc-di'],
        help='Dataset to convert',
    )
    parser.add_argument(
        '--data-dir', type=str, default=None,
        help='Path to dataset directory',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output .pt file path',
    )
    args = parser.parse_args()

    # Resolve data directory
    repo_root = Path(__file__).parent.parent
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.dataset == 'adventureworks':
        data_dir = repo_root / 'datasets' / 'adventureworks'
    else:
        data_dir = repo_root / 'datasets' / args.dataset

    # Convert
    if args.dataset == 'adventureworks':
        data = load_adventureworks(data_dir)
    elif args.dataset == 'tpc-ds':
        data = load_tpc_ds(data_dir)
    elif args.dataset == 'tpc-di':
        data = load_tpc_di(data_dir)

    print_graph_summary(data)

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = repo_root / 'datasets' / args.dataset / \
            'schema_graph.pt'

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    print(f"Saved to: {out_path}")


if __name__ == '__main__':
    main()
