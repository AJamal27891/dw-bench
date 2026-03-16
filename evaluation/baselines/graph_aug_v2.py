"""Graph-Augmented V2 Baseline for DW-Bench Tier 2.

Extends graph_aug.py by providing value-level context:
  - Loads value_data CSVs and lineage_map
  - For each question, extracts relevant value data +
    lineage chain context to feed to the LLM
  - The LLM receives structural graph context PLUS
    value-level data for relevant tables

Does NOT modify graph_aug.py — standalone Tier 2 baseline.

Usage:
    Called by evaluate.py with --baseline graph_aug_v2 --tier 2
"""
import csv
import json
import time
from collections import deque
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# SILOS definition for silo awareness
# ============================================================
SILOS = {
    'logistics': ['raw_purchase_orders', 'raw_shipment_tracking',
                   'raw_supplier_catalog', 'raw_warehouse_inventory',
                   'raw_freight_invoices', 'stg_orders_cleaned',
                   'stg_shipments_validated', 'stg_suppliers_deduped',
                   'stg_inventory_snapshot', 'stg_freight_reconciled',
                   'dim_supplier', 'dim_product', 'dim_warehouse',
                   'dim_carrier', 'fact_shipment', 'fact_purchase_order',
                   'mart_delivery_performance', 'mart_procurement_cost',
                   'mart_supplier_scorecard', 'mart_inventory_turnover'],
    'hr': ['raw_employee_records', 'raw_department_hierarchy',
            'raw_compensation_data', 'stg_employees_cleaned',
            'stg_departments_mapped', 'stg_compensation_normalized',
            'dim_employee', 'dim_department', 'fact_payroll',
            'mart_headcount_report', 'mart_attrition_analysis'],
    'healthcare': ['raw_patient_intake', 'raw_lab_results',
                    'raw_prescriptions', 'stg_patients_validated',
                    'stg_labs_standardized', 'stg_prescriptions_coded',
                    'dim_patient', 'dim_provider', 'fact_encounter',
                    'mart_readmission_risk', 'mart_treatment_outcomes'],
    'ecommerce': ['raw_cart_sessions', 'raw_product_catalog',
                   'raw_payment_events', 'stg_sessions_enriched',
                   'stg_products_normalized', 'stg_payments_validated',
                   'dim_customer', 'dim_product_listing',
                   'fact_transaction', 'mart_conversion_funnel',
                   'mart_revenue_dashboard'],
    'finance': ['raw_ledger_entries', 'raw_bank_feeds', 'raw_tax_filings',
                 'stg_ledger_reconciled', 'stg_bank_matched',
                 'stg_tax_validated', 'dim_account', 'dim_cost_center',
                 'fact_journal_entry', 'mart_trial_balance',
                 'mart_cash_flow_statement'],
}

TABLE_TO_SILO = {}
for silo, tables in SILOS.items():
    for t in tables:
        TABLE_TO_SILO[t] = silo


def load_value_context(dataset_dir: Path):
    """Load value data, lineage map, and build lineage summaries."""
    value_dir = dataset_dir / 'value_data'
    with open(value_dir / '_manifest.json', 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    with open(dataset_dir / 'lineage_map.json', 'r', encoding='utf-8') as f:
        lineage_map = json.load(f)
    return manifest, lineage_map, value_dir


def get_table_sample(table_name: str, value_dir: Path, n_rows: int = 3):
    """Get first n rows from a table CSV."""
    csv_path = value_dir / f'{table_name}.csv'
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n_rows:
                break
            rows.append(dict(row))
    return rows


def build_lineage_summary(table_name: str, lineage_map: dict,
                           manifest: dict, depth: int = 2):
    """Build a textual summary of lineage for a table."""
    lines = []

    # Backward lineage
    if table_name in lineage_map:
        rows_map = lineage_map[table_name]
        # Get unique source tables
        source_tables = set()
        for row_info in rows_map.values():
            for src in row_info['sources']:
                source_tables.add(src['table'])

        if source_tables:
            lines.append(f'  Sources → {table_name}: {", ".join(sorted(source_tables))}')

            # Go one level deeper
            if depth > 1:
                for src_table in sorted(source_tables):
                    if src_table in lineage_map:
                        src_sources = set()
                        for row_info in lineage_map[src_table].values():
                            for s in row_info['sources']:
                                src_sources.add(s['table'])
                        if src_sources:
                            lines.append(f'    Sources → {src_table}: '
                                         f'{", ".join(sorted(src_sources))}')

    return '\n'.join(lines)


def build_value_context(question: str, manifest: dict,
                        lineage_map: dict, value_dir: Path,
                        table_names: list):
    """Build value-enhanced context for a Tier 2 question."""
    lines = []

    # Find tables mentioned in question
    q_lower = question.lower()
    mentioned = []
    for t in sorted(table_names, key=len, reverse=True):
        if t.lower() in q_lower:
            mentioned.append(t)
            q_lower = q_lower.replace(t.lower(), '___', 1)

    lines.append(f"SCHEMA: {len(table_names)} tables across 5 silos "
                 f"(logistics, hr, healthcare, ecommerce, finance)")
    lines.append(f"Layers: raw → staging → core (dim/fact) → mart")
    lines.append("")

    # Silo overview
    lines.append("SILOS:")
    for silo_name, silo_tables in SILOS.items():
        lines.append(f"  {silo_name}: {', '.join(silo_tables)}")
    lines.append("")

    # Lineage context for mentioned tables
    if mentioned:
        lines.append("LINEAGE CHAINS:")
        for t in mentioned:
            chain = build_lineage_summary(t, lineage_map, manifest, depth=3)
            if chain:
                lines.append(chain)
        lines.append("")

        # Value sample for mentioned tables
        lines.append("VALUE SAMPLES:")
        for t in mentioned:
            if t in manifest:
                sample = get_table_sample(t, value_dir, n_rows=3)
                if sample:
                    lines.append(f"  {t} ({manifest[t]['rows']} rows, "
                                 f"silo={TABLE_TO_SILO.get(t, '?')}):")
                    for row in sample:
                        lines.append(f"    {row}")
    else:
        # No specific tables - show full lineage overview
        lines.append("LINEAGE OVERVIEW (all mart tables):")
        for t in table_names:
            if t.startswith('mart_'):
                chain = build_lineage_summary(t, lineage_map, manifest, depth=3)
                if chain:
                    lines.append(chain)

    lines.append("")

    # Row-level lineage detail for specific rows mentioned
    import re
    row_match = re.search(r'(row_\d+)', question)
    if row_match and mentioned:
        row_key = row_match.group(1)
        target_table = mentioned[0]
        if target_table in lineage_map:
            row_info = lineage_map[target_table].get(row_key, {})
            sources = row_info.get('sources', [])
            if sources:
                lines.append(f"ROW LINEAGE for {target_table} {row_key}:")
                for src in sources:
                    lines.append(f"  ← {src['table']}: rows "
                                 f"{src['rows'][:10]}{'...' if len(src['rows']) > 10 else ''}")
                lines.append("")

    return '\n'.join(lines)


def run_graph_aug_v2(dataset_dir: Path, api_key: str = "",
                      api_base: str = "https://api.groq.com/openai/v1",
                      model: str = "llama-3.3-70b-versatile",
                      qa_file: str = None) -> list:
    """Run Graph-Aug V2 baseline on Tier 2 QA pairs."""
    from flat_text import call_llm, SYSTEM_PROMPT, extract_answer

    if qa_file is None:
        qa_file = 'qa_pairs_tier2.json'

    with open(dataset_dir / qa_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # Load schema graph for table names
    import torch
    data = torch.load(dataset_dir / 'schema_graph.pt', weights_only=False)
    table_names = data['table'].table_names

    # Load value data
    manifest, lineage_map, value_dir = load_value_context(dataset_dir)
    is_local = '127.0.0.1' in api_base or 'localhost' in api_base

    print(f"\n  Graph-Aug V2 baseline: {len(questions)} Tier 2 questions")

    results = []
    for i, q in enumerate(questions):
        context = build_value_context(
            q['question'], manifest, lineage_map, value_dir, table_names)

        user_prompt = "CONTEXT:\n" + context + "\n\nQUESTION: " + q['question']

        raw_response = call_llm(SYSTEM_PROMPT, user_prompt, api_key,
                                api_base=api_base, model=model)
        predicted = extract_answer(raw_response, q['answer_type'])
        api_failure = not raw_response.get('_raw_text')

        results.append({
            'id': q['id'],
            'question': q['question'],
            'gold_answer': q['answer'],
            'predicted_answer': predicted,
            'answer_type': q['answer_type'],
            'type': q.get('type', ''),
            'subtype': q['subtype'],
            'difficulty': q['difficulty'],
            'reasoning': raw_response.get('reasoning', ''),
            'api_failure': api_failure,
            'raw_response': raw_response,
        })

        if (i + 1) % 10 == 0:
            failures = sum(1 for r in results if r['api_failure'])
            print(f"    Processed {i+1}/{len(questions)} "
                  f"(API failures: {failures})")

        if not is_local:
            time.sleep(4)

    return results
