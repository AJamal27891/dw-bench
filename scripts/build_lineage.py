"""Build ground-truth ETL lineage (DERIVED_FROM edges) for AdventureWorks.

Sources of truth (in priority order):
  1. complete_etl_sql_metadata.json — staging views with explicit source_tables
  2. Microsoft documentation — well-known DimCustomer, DimProduct, etc. mappings
  3. Column name matching — DW columns that clearly reference OLTP table patterns

Output: datasets/adventureworks/lineage_edges.json
  Format: [{"dw_table": "DW.dbo.X", "oltp_source": "OLTP.Schema.Y", "evidence": "..."}]

Usage:
    python build_lineage.py
"""
import json
from pathlib import Path


def build_lineage() -> list:
    """Build the complete OLTP -> DW lineage edge list.

    Each edge means: DW table is DERIVED_FROM the OLTP source table.
    """
    edges = []

    # ─────────────────────────────────────────────────────────────────
    # Source 1: ETL metadata from staging views
    # From: enrichment_files/complete_etl_sql_metadata.json
    # These are the actual SQL views used in the SSIS ETL packages.
    # ─────────────────────────────────────────────────────────────────
    etl_lineage = {
        # FactResellerSales staging view joins these OLTP tables
        'FactResellerSales': {
            'sources': [
                'Sales.SalesOrderHeader',
                'Sales.SalesOrderDetail',
                'Sales.Customer',
                'Production.Product',
            ],
            'evidence': 'complete_etl_sql_metadata.json: staging view [dbo].[FactResellerSales]',
        },
        # FactInternetSales/FactOnlineSales staging view
        'FactInternetSales': {
            'sources': [
                'Sales.SalesOrderHeader',
                'Sales.SalesOrderDetail',
                'Sales.Customer',
                'Production.Product',
            ],
            'evidence': 'complete_etl_sql_metadata.json: staging view [dbo].[FactOnlineSales]',
        },
        # FactPurchaseOrder staging view (maps to DW FactFinance/purchasing)
        'FactProductInventory': {
            'sources': [
                'Purchasing.PurchaseOrderDetail',
                'Purchasing.PurchaseOrderHeader',
                'Purchasing.Vendor',
            ],
            'evidence': 'complete_etl_sql_metadata.json: staging view [dbo].[FactPurchaseOrder]',
        },
        # DimReseller staging view
        'DimReseller': {
            'sources': [
                'Sales.Customer',
                'Sales.Store',
            ],
            'evidence': 'complete_etl_sql_metadata.json: staging view [dbo].[vw_DimReseller]',
        },
        # DimWorkOrder (custom dim added by ETL project)
        # Note: not in standard AdventureWorksDW but exists in repo ETL
    }

    # ─────────────────────────────────────────────────────────────────
    # Source 2: Microsoft documentation — well-known DW dimension lineage
    # Verified against: learn.microsoft.com, dataedo.com AdventureWorks docs
    # ─────────────────────────────────────────────────────────────────
    doc_lineage = {
        'DimCustomer': {
            'sources': [
                'Person.Person',
                'Sales.Customer',
                'Person.EmailAddress',
                'Person.BusinessEntityAddress',
                'Person.Address',
            ],
            'evidence': 'Microsoft docs: DimCustomer derives from Person + Sales.Customer + contact info tables',
        },
        'DimProduct': {
            'sources': [
                'Production.Product',
                'Production.ProductSubcategory',
                'Production.ProductCategory',
                'Production.ProductModel',
            ],
            'evidence': 'Microsoft docs: DimProduct joins Product + Subcategory + Category + Model',
        },
        'DimEmployee': {
            'sources': [
                'HumanResources.Employee',
                'Person.Person',
                'Sales.SalesPerson',
            ],
            'evidence': 'Microsoft docs: DimEmployee joins Employee + Person + SalesPerson',
        },
        'DimGeography': {
            'sources': [
                'Person.Address',
                'Person.StateProvince',
                'Person.CountryRegion',
            ],
            'evidence': 'Microsoft docs: DimGeography from Address + StateProvince + CountryRegion',
        },
        'DimSalesTerritory': {
            'sources': [
                'Sales.SalesTerritory',
            ],
            'evidence': 'Microsoft docs: direct mapping from Sales.SalesTerritory',
        },
        'DimCurrency': {
            'sources': [
                'Sales.Currency',
            ],
            'evidence': 'Microsoft docs: direct mapping from Sales.Currency',
        },
        'DimPromotion': {
            'sources': [
                'Sales.SpecialOffer',
            ],
            'evidence': 'Microsoft docs: DimPromotion from Sales.SpecialOffer',
        },
    }

    # ─────────────────────────────────────────────────────────────────
    # Source 3: Column name matching + Dataedo/online verification
    # Verified against: dataedo.com AdventureWorksDW2017 docs
    # ─────────────────────────────────────────────────────────────────
    column_match_lineage = {
        'FactInternetSalesReason': {
            'sources': [
                'Sales.SalesOrderHeaderSalesReason',
                'Sales.SalesReason',
            ],
            'evidence': 'Column match: SalesReasonKey -> Sales.SalesReason.SalesReasonID',
        },
        'DimSalesReason': {
            'sources': [
                'Sales.SalesReason',
            ],
            'evidence': 'Dataedo: DimSalesReason direct mapping from Sales.SalesReason',
        },
        'FactCurrencyRate': {
            'sources': [
                'Sales.CurrencyRate',
            ],
            'evidence': 'Dataedo: FactCurrencyRate derives from Sales.CurrencyRate',
        },
        'FactSalesQuota': {
            'sources': [
                'Sales.SalesPersonQuotaHistory',
            ],
            'evidence': 'Dataedo: FactSalesQuota derives from Sales.SalesPersonQuotaHistory',
        },
        'DimDepartmentGroup': {
            'sources': [
                'HumanResources.Department',
            ],
            'evidence': 'Column match: DepartmentGroupName <-> Department hierarchy',
        },
        'DimProductCategory': {
            'sources': [
                'Production.ProductCategory',
            ],
            'evidence': 'Column match: direct mapping from Production.ProductCategory',
        },
        'DimProductSubcategory': {
            'sources': [
                'Production.ProductSubcategory',
            ],
            'evidence': 'Column match: direct mapping from Production.ProductSubcategory',
        },
        # ── Tables with NO OLTP source (verified online) ──────────
        # DimDate: generated calendar dimension (no OLTP source)
        # DimAccount: finance GL account hierarchy (no OLTP source)
        # DimOrganization: organizational hierarchy for finance (no OLTP source)
        # DimScenario: budget/forecast scenario labels (no OLTP source)
        # FactCallCenter: loaded from external call center data
        # FactSurveyResponse: loaded from external survey data
        # FactFinance: loaded from finance/GL system
        # FactAdditionalInternationalProductDescription: catalog metadata
        # ProspectiveBuyer: marketing import data
        # System tables: sysdiagrams, DatabaseLog, AdventureWorksDWBuildVersion
        # NewFactCurrencyRate: staging/duplicate of FactCurrencyRate
    }

    # Merge all lineage sources
    all_lineage = {}
    all_lineage.update(etl_lineage)
    all_lineage.update(doc_lineage)
    all_lineage.update(column_match_lineage)

    # Convert to edges
    for dw_table, info in all_lineage.items():
        for source in info['sources']:
            schema_table = source.split('.')
            if len(schema_table) == 2:
                oltp_name = f"OLTP.{schema_table[0]}.{schema_table[1]}"
            else:
                oltp_name = f"OLTP.dbo.{source}"

            edges.append({
                'dw_table': f"DW.dbo.{dw_table}",
                'oltp_source': oltp_name,
                'evidence': info['evidence'],
            })

    return edges


def validate_lineage(edges: list, data_dir: Path) -> dict:
    """Validate lineage edges against actual table lists."""
    import pandas as pd

    oltp_tables = pd.read_csv(data_dir / 'AdventureWorks_Tables.csv')
    dw_tables = pd.read_csv(data_dir / 'AdventureWorksDW_Tables.csv')

    oltp_names = set()
    for _, row in oltp_tables.iterrows():
        oltp_names.add(f"OLTP.{row['TABLE_SCHEMA']}.{row['TABLE_NAME']}")

    dw_names = set()
    for _, row in dw_tables.iterrows():
        dw_names.add(f"DW.dbo.{row['TABLE_NAME']}")

    valid = []
    invalid_dw = []
    invalid_oltp = []

    for edge in edges:
        dw_ok = edge['dw_table'] in dw_names
        oltp_ok = edge['oltp_source'] in oltp_names
        if dw_ok and oltp_ok:
            valid.append(edge)
        elif not dw_ok:
            invalid_dw.append(edge)
        else:
            invalid_oltp.append(edge)

    return {
        'valid': valid,
        'invalid_dw': invalid_dw,
        'invalid_oltp': invalid_oltp,
    }


def main():
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / 'datasets' / 'adventureworks'

    print("Building ground-truth lineage edges...")
    edges = build_lineage()
    print(f"  Total candidate edges: {len(edges)}")

    # Validate against actual table lists
    print("\nValidating against table CSVs...")
    results = validate_lineage(edges, data_dir)

    print(f"  ✅ Valid edges: {len(results['valid'])}")
    if results['invalid_dw']:
        print(f"  ❌ Invalid DW tables ({len(results['invalid_dw'])}):")
        for e in results['invalid_dw']:
            print(f"      {e['dw_table']} (not found)")
    if results['invalid_oltp']:
        print(f"  ❌ Invalid OLTP tables ({len(results['invalid_oltp'])}):")
        for e in results['invalid_oltp']:
            print(f"      {e['oltp_source']} (not found)")

    # Print valid lineage
    print(f"\n{'=' * 60}")
    print("VALIDATED LINEAGE (DERIVED_FROM edges)")
    print(f"{'=' * 60}")
    dw_groups = {}
    for e in results['valid']:
        dw_groups.setdefault(e['dw_table'], []).append(e['oltp_source'])

    for dw_table in sorted(dw_groups.keys()):
        sources = dw_groups[dw_table]
        print(f"\n  {dw_table}:")
        for s in sorted(sources):
            print(f"    ← {s}")

    # Save validated edges
    out_path = data_dir / 'lineage_edges.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results['valid'], f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results['valid'])} validated edges to: {out_path}")


if __name__ == '__main__':
    main()
