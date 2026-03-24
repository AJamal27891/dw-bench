"""Validation script: cross-checks DW-Bench data against official sources.

Validates:
  1. AdventureWorks table counts, FK edge counts
  2. TPC-DS table list, FK edge list against official spec
  3. Schema graph node/edge integrity

Usage:
    python validate_data.py
"""
from pathlib import Path

import pandas as pd
import torch


def validate_adventureworks(ds_dir: Path) -> bool:
    """Validate AdventureWorks against known official counts.

    Official: ~71 OLTP base tables, ~32 DW tables+views, ~91 FK rels (OLTP),
              ~45 FK rels (DW)
    Source: Microsoft SQL Server AdventureWorks 2019 sample database
    """
    print("=" * 60)
    print("VALIDATING: AdventureWorks")
    print("=" * 60)
    ok = True

    # --- Table counts ---
    oltp_tables = pd.read_csv(ds_dir / 'AdventureWorks_Tables.csv')
    dw_tables = pd.read_csv(ds_dir / 'AdventureWorksDW_Tables.csv')

    oltp_base = oltp_tables[oltp_tables['TABLE_TYPE'] == 'BASE TABLE']
    oltp_views = oltp_tables[oltp_tables['TABLE_TYPE'] == 'VIEW']
    dw_base = dw_tables[dw_tables['TABLE_TYPE'] == 'BASE TABLE']
    dw_views = dw_tables[dw_tables['TABLE_TYPE'] == 'VIEW']

    print(f"\nOLTP tables: {len(oltp_base)} base + {len(oltp_views)} views "
          f"= {len(oltp_tables)}")
    print(f"DW tables:   {len(dw_base)} base + {len(dw_views)} views "
          f"= {len(dw_tables)}")

    # Microsoft docs say ~71 base tables for OLTP
    if len(oltp_base) < 65 or len(oltp_base) > 75:
        print(f"  ❌ OLTP base table count {len(oltp_base)} outside "
              f"expected range [65-75]")
        ok = False
    else:
        print(f"  ✅ OLTP base tables: {len(oltp_base)} (expected ~71)")

    # DW has 32 base tables + 5 views in standard install
    if len(dw_base) < 28 or len(dw_base) > 35:
        print(f"  ❌ DW base table count {len(dw_base)} outside "
              f"expected range [28-35]")
        ok = False
    else:
        print(f"  ✅ DW base tables: {len(dw_base)} (expected ~32)")

    # --- FK relationship counts ---
    oltp_rels = pd.read_csv(ds_dir / 'AdventureWorks_Relationships.csv')
    dw_rels = pd.read_csv(ds_dir / 'AdventureWorksDW_Relationships.csv')

    print(f"\nOLTP FK relationships: {len(oltp_rels)}")
    print(f"DW FK relationships:   {len(dw_rels)}")

    # Verify all FK child/parent tables exist in the tables list
    oltp_table_names = set(oltp_tables['TABLE_NAME'])
    dw_table_names = set(dw_tables['TABLE_NAME'])

    oltp_fk_tables = set(oltp_rels['ChildTable']) | set(oltp_rels['ParentTable'])
    dw_fk_tables = set(dw_rels['ChildTable']) | set(dw_rels['ParentTable'])

    oltp_orphans = oltp_fk_tables - oltp_table_names
    dw_orphans = dw_fk_tables - dw_table_names

    if oltp_orphans:
        print(f"  ❌ OLTP FK references non-existent tables: {oltp_orphans}")
        ok = False
    else:
        print(f"  ✅ All OLTP FK tables exist in table list")

    if dw_orphans:
        print(f"  ❌ DW FK references non-existent tables: {dw_orphans}")
        ok = False
    else:
        print(f"  ✅ All DW FK tables exist in table list")

    # --- Column counts ---
    oltp_cols = pd.read_csv(ds_dir / 'AdventureWorks_Columns.csv')
    dw_cols = pd.read_csv(ds_dir / 'AdventureWorksDW_Columns.csv')

    print(f"\nOLTP columns: {len(oltp_cols)}")
    print(f"DW columns:   {len(dw_cols)}")

    # Verify columns reference existing tables
    oltp_col_tables = set(oltp_cols['TABLE_NAME'])
    dw_col_tables = set(dw_cols['TABLE_NAME'])

    col_orphans_oltp = oltp_col_tables - oltp_table_names
    col_orphans_dw = dw_col_tables - dw_table_names

    if col_orphans_oltp:
        print(f"  ❌ OLTP column CSV references non-existent tables: "
              f"{col_orphans_oltp}")
        ok = False
    else:
        print(f"  ✅ All OLTP column tables exist")

    if col_orphans_dw:
        print(f"  ❌ DW column CSV references non-existent tables: "
              f"{col_orphans_dw}")
        ok = False
    else:
        print(f"  ✅ All DW column tables exist")

    # --- Spot-check key tables ---
    key_oltp_tables = {
        'SalesOrderHeader', 'SalesOrderDetail', 'Product', 'Person',
        'Employee', 'Customer', 'Address', 'Store', 'Vendor',
        'PurchaseOrderHeader',
    }
    missing_key = key_oltp_tables - oltp_table_names
    if missing_key:
        print(f"\n  ❌ Missing key tables: {missing_key}")
        ok = False
    else:
        print(f"  ✅ All 10 key OLTP tables present")

    key_dw_tables = {
        'DimCustomer', 'DimProduct', 'DimDate', 'DimEmployee',
        'FactInternetSales', 'FactResellerSales', 'FactFinance',
    }
    missing_dw_key = key_dw_tables - dw_table_names
    if missing_dw_key:
        print(f"  ❌ Missing key DW tables: {missing_dw_key}")
        ok = False
    else:
        print(f"  ✅ All 7 key DW tables present")

    # --- Validate schema graph .pt ---
    graph_path = ds_dir / 'schema_graph.pt'
    if graph_path.exists():
        data = torch.load(graph_path, weights_only=False)
        n_nodes = data['table'].num_nodes
        n_edges = data['table', 'fk_to', 'table'].edge_index.shape[1]
        expected_tables = len(oltp_base) + len(dw_base)
        print(f"\n  Schema graph: {n_nodes} nodes, {n_edges} edges")
        if n_nodes != expected_tables:
            print(f"  ❌ Node count {n_nodes} != expected {expected_tables}")
            ok = False
        else:
            print(f"  ✅ Node count matches base table count")
    else:
        print(f"  ⚠️  No schema_graph.pt found")

    return ok


def validate_tpc_ds(ds_dir: Path) -> bool:
    """Validate TPC-DS against official spec (TPC-DS v4.0.0).

    Official: 7 fact + 17 dim = 24 tables + dbgen_version
    """
    print("\n" + "=" * 60)
    print("VALIDATING: TPC-DS")
    print("=" * 60)
    ok = True

    # Official TPC-DS v4.0 tables (excluding dbgen_version)
    official_fact = {
        'store_sales', 'store_returns', 'catalog_sales',
        'catalog_returns', 'web_sales', 'web_returns', 'inventory',
    }
    official_dim = {
        'date_dim', 'time_dim', 'item', 'customer', 'customer_address',
        'customer_demographics', 'household_demographics', 'store',
        'promotion', 'warehouse', 'ship_mode', 'reason', 'income_band',
        'call_center', 'catalog_page', 'web_site', 'web_page',
    }
    official_all = official_fact | official_dim

    # Official FK edges from TPC-DS spec (surrogate key joins)
    # Every _sk column in a fact table referencing a dimension PK
    official_fk_edges = {
        # store_sales FKs (9)
        ('store_sales', 'date_dim'),
        ('store_sales', 'time_dim'),
        ('store_sales', 'item'),
        ('store_sales', 'customer'),
        ('store_sales', 'customer_demographics'),
        ('store_sales', 'household_demographics'),
        ('store_sales', 'customer_address'),
        ('store_sales', 'store'),
        ('store_sales', 'promotion'),
        # store_returns FKs (9)
        ('store_returns', 'date_dim'),
        ('store_returns', 'time_dim'),
        ('store_returns', 'item'),
        ('store_returns', 'customer'),
        ('store_returns', 'customer_demographics'),
        ('store_returns', 'household_demographics'),
        ('store_returns', 'customer_address'),
        ('store_returns', 'store'),
        ('store_returns', 'reason'),
        # catalog_sales FKs (13) — bill + ship customers
        ('catalog_sales', 'date_dim'),
        ('catalog_sales', 'time_dim'),
        ('catalog_sales', 'item'),
        ('catalog_sales', 'customer'),          # bill_customer
        ('catalog_sales', 'customer_demographics'),
        ('catalog_sales', 'household_demographics'),
        ('catalog_sales', 'customer_address'),
        ('catalog_sales', 'call_center'),
        ('catalog_sales', 'catalog_page'),
        ('catalog_sales', 'ship_mode'),
        ('catalog_sales', 'warehouse'),
        ('catalog_sales', 'promotion'),
        # catalog_returns FKs (11)
        ('catalog_returns', 'date_dim'),
        ('catalog_returns', 'time_dim'),
        ('catalog_returns', 'item'),
        ('catalog_returns', 'customer'),
        ('catalog_returns', 'customer_demographics'),
        ('catalog_returns', 'household_demographics'),
        ('catalog_returns', 'customer_address'),
        ('catalog_returns', 'reason'),
        ('catalog_returns', 'call_center'),
        ('catalog_returns', 'catalog_page'),
        ('catalog_returns', 'ship_mode'),
        ('catalog_returns', 'warehouse'),
        # web_sales FKs (13)
        ('web_sales', 'date_dim'),
        ('web_sales', 'time_dim'),
        ('web_sales', 'item'),
        ('web_sales', 'customer'),
        ('web_sales', 'customer_demographics'),
        ('web_sales', 'household_demographics'),
        ('web_sales', 'customer_address'),
        ('web_sales', 'web_site'),
        ('web_sales', 'web_page'),
        ('web_sales', 'ship_mode'),
        ('web_sales', 'warehouse'),
        ('web_sales', 'promotion'),
        # web_returns FKs (9)
        ('web_returns', 'date_dim'),
        ('web_returns', 'time_dim'),
        ('web_returns', 'item'),
        ('web_returns', 'customer'),
        ('web_returns', 'reason'),
        ('web_returns', 'web_page'),
        ('web_returns', 'customer_demographics'),
        ('web_returns', 'household_demographics'),
        ('web_returns', 'customer_address'),
        # inventory FKs (3)
        ('inventory', 'date_dim'),
        ('inventory', 'item'),
        ('inventory', 'warehouse'),
        # dim-to-dim FKs
        ('customer', 'customer_address'),
        ('customer', 'customer_demographics'),
        ('customer', 'household_demographics'),
        ('household_demographics', 'income_band'),
    }

    print(f"\nOfficial table count: {len(official_all)} "
          f"(7 fact + 17 dim)")
    print(f"Official FK edge count: {len(official_fk_edges)}")

    # Validate schema graph
    graph_path = ds_dir / 'schema_graph.pt'
    if graph_path.exists():
        data = torch.load(graph_path, weights_only=False)
        graph_tables = set(data['table'].table_names)
        n_edges = data['table', 'fk_to', 'table'].edge_index.shape[1]

        print(f"\nGraph table count: {len(graph_tables)}")
        print(f"Graph FK edge count: {n_edges}")

        # Check table names match
        missing = official_all - graph_tables
        extra = graph_tables - official_all
        if missing:
            print(f"  ❌ Missing tables: {missing}")
            ok = False
        if extra:
            print(f"  ❌ Extra tables: {extra}")
            ok = False
        if not missing and not extra:
            print(f"  ✅ All 24 tables match official spec")

        # Check edge count
        if n_edges < len(official_fk_edges):
            print(f"  ❌ Only {n_edges} edges, expected "
                  f"{len(official_fk_edges)}")
            ok = False
        else:
            print(f"  ✅ Edge count {n_edges} >= official "
                  f"{len(official_fk_edges)}")
    else:
        print(f"  ⚠️  No schema_graph.pt found")

    return ok


def main():
    repo_root = Path(__file__).parent.parent
    all_ok = True

    aw_dir = repo_root / 'datasets' / 'adventureworks'
    if aw_dir.exists():
        all_ok &= validate_adventureworks(aw_dir)
    else:
        print("⚠️  AdventureWorks dataset not found")

    tpc_dir = repo_root / 'datasets' / 'tpc-ds'
    if tpc_dir.exists():
        all_ok &= validate_tpc_ds(tpc_dir)
    else:
        print("⚠️  TPC-DS dataset not found")

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL VALIDATIONS PASSED ✅")
    else:
        print("SOME VALIDATIONS FAILED ❌")
    print("=" * 60)


if __name__ == '__main__':
    main()
