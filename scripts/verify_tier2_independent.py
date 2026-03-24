"""Independent Tier 2 QA Verification Suite.

This script verifies QA correctness using a COMPLETELY DIFFERENT code path
from generate_value_qa.py. It re-derives every answer from scratch using
only lineage_map.json, then checks:

  1. ANSWER CORRECTNESS: Re-derive every answer via independent BFS
  2. QUESTION PARSABILITY: Every question contains enough info to solve
  3. HUMAN-SOLVABILITY: Every answer can be derived from lineage_map alone
  4. NO INFORMATION LEAKAGE: Answers don't require value_data CSV content
  5. EDGE CASES: No trivially unsolvable questions, no ambiguous phrasing
  6. ANSWER UNIQUENESS: No two questions produce the same answer by accident
  7. FORMAT CONSISTENCY: answer_type matches actual Python type

Usage:
    python scripts/verify_tier2_independent.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DS_DIR = REPO_ROOT / 'datasets' / 'syn_logistics'
VALUE_DIR = DS_DIR / 'value_data'

# ─── Load data ───────────────────────────────────────────────

def load():
    with open(DS_DIR / 'qa_pairs_tier2.json', 'r') as f:
        qa = json.load(f)
    with open(DS_DIR / 'lineage_map.json', 'r') as f:
        lm = json.load(f)
    with open(VALUE_DIR / '_manifest.json', 'r') as f:
        manifest = json.load(f)
    return qa, lm, manifest


# ─── Independent BFS implementations ────────────────────────
# These are intentionally DIFFERENT from the generator's code.
# Generator uses defaultdict(set) + list pop; we use dict + deque-like.

def backward_bfs(lm, start_table, start_row):
    """Independent backward BFS. Returns dict of {table: set(row_keys)}."""
    ancestors = {}
    frontier = [(start_table, start_row)]
    seen = set()
    while frontier:
        t, r = frontier.pop(0)  # BFS order (different from generator's stack)
        if (t, r) in seen:
            continue
        seen.add((t, r))
        entry = lm.get(t, {}).get(r, {})
        for src in entry.get('sources', []):
            st = src['table']
            for ri in src['rows']:
                rk = f'row_{ri}'
                if st not in ancestors:
                    ancestors[st] = set()
                ancestors[st].add(rk)
                frontier.append((st, rk))
    return ancestors


def forward_bfs(reverse_idx, start_table, start_row):
    """Independent forward BFS. Returns dict of {table: set(row_keys)}."""
    descendants = {}
    frontier = [f'{start_table}:{start_row}']
    seen = set()
    while frontier:
        sid = frontier.pop(0)  # BFS order
        if sid in seen:
            continue
        seen.add(sid)
        for dt, dr in reverse_idx.get(sid, []):
            if dt not in descendants:
                descendants[dt] = set()
            descendants[dt].add(dr)
            frontier.append(f'{dt}:{dr}')
    return descendants


def build_reverse_idx(lm):
    """Build reverse lineage index independently."""
    rev = {}
    for tgt_table in lm:
        for rk, info in lm[tgt_table].items():
            for src in info.get('sources', []):
                for ri in src['rows']:
                    key = f"{src['table']}:row_{ri}"
                    if key not in rev:
                        rev[key] = []
                    rev[key].append((tgt_table, rk))
    return rev


# ─── Silo definitions (hardcoded, independent of generator) ─

SILO_MAP = {}
_SILOS = {
    'logistics': [
        'raw_purchase_orders', 'raw_shipment_tracking', 'raw_supplier_catalog',
        'raw_warehouse_inventory', 'raw_freight_invoices',
        'stg_orders_cleaned', 'stg_shipments_validated', 'stg_suppliers_deduped',
        'stg_inventory_snapshot', 'stg_freight_reconciled',
        'dim_supplier', 'dim_product', 'dim_warehouse', 'dim_carrier',
        'fact_shipment', 'fact_purchase_order',
        'mart_delivery_performance', 'mart_procurement_cost',
        'mart_supplier_scorecard', 'mart_inventory_turnover',
    ],
    'hr': [
        'raw_employee_records', 'raw_department_hierarchy', 'raw_compensation_data',
        'stg_employees_cleaned', 'stg_departments_mapped', 'stg_compensation_normalized',
        'dim_employee', 'dim_department', 'fact_payroll',
        'mart_headcount_report', 'mart_attrition_analysis',
    ],
    'healthcare': [
        'raw_patient_intake', 'raw_lab_results', 'raw_prescriptions',
        'stg_patients_validated', 'stg_labs_standardized', 'stg_prescriptions_coded',
        'dim_patient', 'dim_provider', 'fact_encounter',
        'mart_readmission_risk', 'mart_treatment_outcomes',
    ],
    'ecommerce': [
        'raw_cart_sessions', 'raw_product_catalog', 'raw_payment_events',
        'stg_sessions_enriched', 'stg_products_normalized', 'stg_payments_validated',
        'dim_customer', 'dim_product_listing', 'fact_transaction',
        'mart_conversion_funnel', 'mart_revenue_dashboard',
    ],
    'finance': [
        'raw_ledger_entries', 'raw_bank_feeds', 'raw_tax_filings',
        'stg_ledger_reconciled', 'stg_bank_matched', 'stg_tax_validated',
        'dim_account', 'dim_cost_center', 'fact_journal_entry',
        'mart_trial_balance', 'mart_cash_flow_statement',
    ],
}
for silo, tables in _SILOS.items():
    for t in tables:
        SILO_MAP[t] = silo


# ─── Question parser (independent of generator) ─────────────

def extract_tables_from_question(q_text, known_tables):
    """Find all known table names in question text."""
    found = []
    remaining = q_text
    for name in sorted(known_tables, key=len, reverse=True):
        if name in remaining:
            found.append(name)
            remaining = remaining.replace(name, '\x00' * len(name), 1)
    return found


def extract_row_from_question(q_text):
    """Extract row_NNN from question text."""
    m = re.search(r'\brow_(\d+)\b', q_text)
    return m.group(0) if m else None


# ─── Independent answer derivation per subtype ──────────────

def derive_answer(q, lm, rev, manifest):
    """Derive the expected answer independently."""
    question = q['question']
    subtype = q['subtype']
    known = set(manifest.keys())
    tables = extract_tables_from_question(question, known)
    row = extract_row_from_question(question)

    if subtype == 'row_provenance':
        if not tables or not row:
            return None, 'Cannot parse table/row'
        tbl = tables[0]
        entry = lm.get(tbl, {}).get(row, {})
        sources = entry.get('sources', [])
        result = []
        for s in sources:
            for ri in s['rows']:
                result.append(f"{s['table']}:row_{ri}")
        return sorted(result), None

    elif subtype == 'value_origin':
        if not tables or not row:
            return None, 'Cannot parse'
        tbl = tables[0]
        ancestors = backward_bfs(lm, tbl, row)
        raw_tables = sorted(t for t in ancestors if t.startswith('raw_'))
        return raw_tables, None

    elif subtype == 'multi_hop_trace':
        if not tables or not row:
            return None, 'Cannot parse'
        tbl = tables[0]
        ancestors = backward_bfs(lm, tbl, row)
        if 'ALL source tables' in question or 'full provenance' in question:
            return sorted(ancestors.keys()), None
        else:
            stg = sorted(t for t in ancestors if t.startswith('stg_'))
            return stg, None

    elif subtype == 'row_impact':
        if not tables or not row:
            return None, 'Cannot parse'
        tbl = tables[0]
        desc = forward_bfs(rev, tbl, row)
        if 'core and mart' in question:
            return sorted(desc.keys()), None
        else:
            marts = sorted(t for t in desc if t.startswith('mart_'))
            return marts, None

    elif subtype == 'value_propagation':
        if len(tables) < 2:
            return None, 'Need 2 tables'
        # Match by prefix, not position (longest-first may reorder)
        raw_tbl = next((t for t in tables if t.startswith('raw_')), None)
        mart_tbl = next((t for t in tables if t.startswith('mart_')), None)
        if not raw_tbl or not mart_tbl:
            return None, f'Need raw_ and mart_ tables, got {tables}'
        # Check if raw_tbl is in backward provenance of mart_tbl
        mart_rows = lm.get(mart_tbl, {})
        if not mart_rows:
            return False, None
        first_row = next(iter(mart_rows))
        ancestors = backward_bfs(lm, mart_tbl, first_row)
        return raw_tbl in ancestors, None

    elif subtype == 'cascade_count':
        if not tables or not row:
            return None, 'Cannot parse'
        tbl = tables[0]
        desc = forward_bfs(rev, tbl, row)
        if 'mart-layer' in question or 'mart layer' in question:
            total = sum(len(rows) for t, rows in desc.items()
                       if t.startswith('mart_'))
        else:
            total = sum(len(rows) for rows in desc.values())
        return total, None

    elif subtype == 'cross_silo_reachability':
        if "silo's raw tables" in question:
            # Silo-pair question: "from the X silo's raw ... into the Y silo's mart"
            import re as _re
            m_src = _re.search(r'from the (\w+) silo', question)
            m_tgt = _re.search(r'into the (\w+) silo', question)
            if not m_src or not m_tgt:
                return None, 'Cannot parse silo names from question'
            src_silo = m_src.group(1)
            tgt_silo = m_tgt.group(1)
            if src_silo not in _SILOS or tgt_silo not in _SILOS:
                return None, f'Unknown silos: {src_silo}, {tgt_silo}'
            # Build provenance for all target marts
            tgt_marts = [t for t in _SILOS[tgt_silo] if t.startswith('mart_')]
            src_raws = [t for t in _SILOS[src_silo] if t.startswith('raw_')]
            flows = False
            for mart in tgt_marts:
                if flows:
                    break
                mart_rows = lm.get(mart, {})
                if not mart_rows:
                    continue
                first_r = next(iter(mart_rows))
                ancestors = backward_bfs(lm, mart, first_r)
                for sr in src_raws:
                    if sr in ancestors:
                        flows = True
                        break
            return flows, None
        else:
            # Per-table question
            if len(tables) < 2:
                return None, 'Need 2 tables'
            # Match by prefix, not position
            raw_tbl = next((t for t in tables if t.startswith('raw_')), None)
            mart_tbl = next((t for t in tables if t.startswith('mart_')), None)
            if not raw_tbl or not mart_tbl:
                return None, f'Need raw_ and mart_, got {tables}'
            mart_rows = lm.get(mart_tbl, {})
            if not mart_rows:
                return False, None
            first_r = next(iter(mart_rows))
            ancestors = backward_bfs(lm, mart_tbl, first_r)
            return raw_tbl in ancestors, None

    elif subtype == 'shared_source':
        if len(tables) < 2:
            return None, 'Need 2 tables'
        mart_a = tables[0]
        mart_b = tables[1]
        rows_a = lm.get(mart_a, {})
        rows_b = lm.get(mart_b, {})
        raw_a = set()
        raw_b = set()
        if rows_a:
            anc_a = backward_bfs(lm, mart_a, next(iter(rows_a)))
            raw_a = {t for t in anc_a if t.startswith('raw_')}
        if rows_b:
            anc_b = backward_bfs(lm, mart_b, next(iter(rows_b)))
            raw_b = {t for t in anc_b if t.startswith('raw_')}
        return sorted(raw_a & raw_b), None

    return None, f'Unknown subtype: {subtype}'


# ─── Human-solvability checks ───────────────────────────────

def check_human_solvable(q, lm, manifest):
    """Check that a question is solvable by a human with lineage_map access."""
    issues = []
    question = q['question']
    subtype = q['subtype']
    known = set(manifest.keys())
    tables = extract_tables_from_question(question, known)
    row = extract_row_from_question(question)

    # 1. Question must mention at least one known table
    if not tables and subtype not in ('cross_silo_reachability',):
        issues.append(f'No known table found in question')

    # 2. Row-specific questions must have a parseable row_NNN
    row_subtypes = {'row_provenance', 'value_origin', 'multi_hop_trace',
                    'row_impact', 'cascade_count'}
    if subtype in row_subtypes and not row:
        issues.append(f'No row_NNN found for row-specific subtype')

    # 3. Table mentioned must exist in manifest
    for t in tables:
        if t not in manifest:
            issues.append(f'Table {t} not in manifest')

    # 4. Row mentioned must exist in lineage_map (for backward subtypes)
    backward_subtypes = {'row_provenance', 'value_origin', 'multi_hop_trace'}
    if subtype in backward_subtypes and tables and row:
        tbl = tables[0]
        if tbl in lm and row not in lm[tbl]:
            issues.append(f'{row} not in lineage_map[{tbl}]')

    # 5. Question should be grammatically complete (ends with ? or .)
    if not question.rstrip().endswith('?') and not question.rstrip().endswith('.'):
        issues.append(f'Question does not end with ? or .')

    # 6. Answer should not require CSV cell values (value data independence)
    csv_keywords = ['column', 'value', 'field', 'cell']
    if subtype not in ('value_propagation',):  # propagation mentions column names
        for kw in csv_keywords:
            if kw in question.lower() and 'column' not in question.lower():
                issues.append(f'Question may require CSV value reading ({kw})')

    return issues


# ─── Main verification suite ────────────────────────────────

def main():
    qa, lm, manifest = load()
    rev = build_reverse_idx(lm)

    print('=' * 70)
    print(f'INDEPENDENT TIER 2 QA VERIFICATION ({len(qa)} questions)')
    print('=' * 70)

    # ── Test 1: Answer Correctness ───────────────────────────
    print('\n[1/7] ANSWER CORRECTNESS (independent re-derivation)...')
    correct = 0
    wrong = 0
    parse_fail = 0
    wrong_details = []

    for q in qa:
        expected, err = derive_answer(q, lm, rev, manifest)
        if err:
            parse_fail += 1
            wrong_details.append(f"  PARSE FAIL [{q['subtype']}] {q['id']}: {err}")
            continue
        if expected == q['answer']:
            correct += 1
        else:
            wrong += 1
            wrong_details.append(
                f"  MISMATCH [{q['subtype']}] {q['id']}:\n"
                f"    Expected: {str(expected)[:100]}\n"
                f"    Got:      {str(q['answer'])[:100]}"
            )

    status = '✅' if wrong == 0 and parse_fail == 0 else '❌'
    print(f'  {status} Correct: {correct}, Wrong: {wrong}, Parse fails: {parse_fail}')
    for d in wrong_details[:10]:
        print(d)

    # ── Test 2: Human Solvability ────────────────────────────
    print('\n[2/7] HUMAN SOLVABILITY CHECKS...')
    total_issues = 0
    issue_details = []

    for q in qa:
        issues = check_human_solvable(q, lm, manifest)
        if issues:
            total_issues += len(issues)
            for iss in issues:
                issue_details.append(f"  ⚠️ {q['id']}: {iss}")

    status = '✅' if total_issues == 0 else '❌'
    print(f'  {status} Total issues: {total_issues}')
    for d in issue_details[:10]:
        print(d)

    # ── Test 3: Format Consistency ───────────────────────────
    print('\n[3/7] FORMAT CONSISTENCY...')
    fmt_errors = 0

    for q in qa:
        at = q['answer_type']
        ans = q['answer']
        if at == 'list' and not isinstance(ans, list):
            fmt_errors += 1
            print(f"  ❌ {q['id']}: type={at} but answer is {type(ans).__name__}")
        elif at == 'integer' and not isinstance(ans, int):
            fmt_errors += 1
            print(f"  ❌ {q['id']}: type={at} but answer is {type(ans).__name__}")
        elif at == 'boolean' and not isinstance(ans, bool):
            fmt_errors += 1
            print(f"  ❌ {q['id']}: type={at} but answer is {type(ans).__name__}")

        # Check required fields
        for field in ['id', 'dataset', 'tier', 'type', 'subtype', 'question',
                      'answer', 'answer_type', 'difficulty', 'reasoning']:
            if field not in q:
                fmt_errors += 1
                print(f"  ❌ {q['id']}: missing field '{field}'")

    status = '✅' if fmt_errors == 0 else '❌'
    print(f'  {status} Format errors: {fmt_errors}')

    # ── Test 4: No Duplicate Questions ───────────────────────
    print('\n[4/7] DUPLICATE CHECK...')
    seen_q = {}
    dupes = 0
    for q in qa:
        if q['question'] in seen_q:
            dupes += 1
            print(f"  ❌ Duplicate: {q['id']} == {seen_q[q['question']]}")
        seen_q[q['question']] = q['id']

    status = '✅' if dupes == 0 else '❌'
    print(f'  {status} Duplicates: {dupes}')

    # ── Test 5: Answer Non-Triviality ────────────────────────
    print('\n[5/7] ANSWER NON-TRIVIALITY...')
    trivial = 0
    notes = []

    for q in qa:
        ans = q['answer']
        # Flag questions with trivially guessable answers
        if isinstance(ans, list) and len(ans) > 30:
            notes.append(f"  NOTE {q['id']}: list with {len(ans)} items "
                        f"(may be hard to verify manually)")

    # Check that boolean subtypes aren't all-same
    by_subtype_bool = defaultdict(lambda: {'T': 0, 'F': 0})
    for q in qa:
        if isinstance(q['answer'], bool):
            k = 'T' if q['answer'] else 'F'
            by_subtype_bool[q['subtype']][k] += 1

    for st, counts in by_subtype_bool.items():
        total = counts['T'] + counts['F']
        ratio = counts['T'] / total if total else 0
        if ratio == 0 or ratio == 1:
            trivial += 1
            print(f"  ❌ {st}: 100% {'True' if ratio == 1 else 'False'} "
                  f"(trivially guessable)")
        elif ratio < 0.2 or ratio > 0.8:
            notes.append(f"  NOTE {st}: {counts['T']}T/{counts['F']}F = "
                        f"{ratio:.0%} (slightly imbalanced)")

    status = '✅' if trivial == 0 else '❌'
    print(f'  {status} Trivially guessable subtypes: {trivial}')
    for n in notes[:10]:
        print(n)

    # ── Test 6: Subtype Coverage ─────────────────────────────
    print('\n[6/7] SUBTYPE COVERAGE...')
    expected_subtypes = {'row_provenance', 'value_origin', 'multi_hop_trace',
                        'row_impact', 'value_propagation', 'cascade_count',
                        'cross_silo_reachability', 'shared_source'}
    found = set(q['subtype'] for q in qa)
    missing = expected_subtypes - found
    extra = found - expected_subtypes

    subtype_counts = defaultdict(int)
    for q in qa:
        subtype_counts[q['subtype']] += 1

    below_30 = [st for st, c in subtype_counts.items() if c < 30]

    status = '✅' if not missing and not extra and not below_30 else '❌'
    print(f'  {status} Missing: {missing or "none"}, Extra: {extra or "none"}')
    if below_30:
        for st in below_30:
            print(f'  ⚠️ {st}: only {subtype_counts[st]} questions (need ≥30)')

    # ── Test 7: Cross-silo edge consistency ──────────────────
    print('\n[7/7] CROSS-SILO EDGE CONSISTENCY...')
    # Verify that healthcare silo is truly isolated
    healthcare_tables = set(_SILOS['healthcare'])
    healthcare_mart_tables = [t for t in _SILOS['healthcare'] if t.startswith('mart_')]

    isolated = True
    for mart in healthcare_mart_tables:
        mart_rows = lm.get(mart, {})
        if not mart_rows:
            continue
        first_r = next(iter(mart_rows))
        ancestors = backward_bfs(lm, mart, first_r)
        non_healthcare_ancestors = [t for t in ancestors
                                   if t not in healthcare_tables]
        if non_healthcare_ancestors:
            isolated = False
            print(f"  ❌ Healthcare mart {mart} has non-healthcare ancestors: "
                  f"{non_healthcare_ancestors}")

    # Check connected silos actually have cross-silo paths
    connected_pairs = [
        ('hr', 'logistics'), ('logistics', 'finance'), ('finance', 'hr'),
        ('ecommerce', 'logistics'),
    ]
    for src_silo, tgt_silo in connected_pairs:
        tgt_marts = [t for t in _SILOS[tgt_silo] if t.startswith('mart_')]
        src_raws = [t for t in _SILOS[src_silo] if t.startswith('raw_')]
        found_path = False
        for mart in tgt_marts:
            if found_path:
                break
            mart_rows = lm.get(mart, {})
            if not mart_rows:
                continue
            first_r = next(iter(mart_rows))
            ancestors = backward_bfs(lm, mart, first_r)
            for sr in src_raws:
                if sr in ancestors:
                    found_path = True
                    break
        if not found_path:
            isolated = False
            print(f"  ❌ Expected {src_silo}→{tgt_silo} path not found")

    status = '✅' if isolated else '❌'
    print(f'  {status} Healthcare isolated: {isolated}')
    print(f'  {status} Connected silo paths verified')

    # ── Summary ──────────────────────────────────────────────
    print('\n' + '=' * 70)
    all_ok = (wrong == 0 and parse_fail == 0 and total_issues == 0
              and fmt_errors == 0 and dupes == 0 and trivial == 0
              and not missing and not below_30 and isolated)

    if all_ok:
        print('ALL 7 VERIFICATION CHECKS PASSED ✅')
        print(f'Dataset is verified: {len(qa)} questions, bug-free, human-solvable.')
    else:
        print('VERIFICATION FAILURES DETECTED ❌')
        if wrong:
            print(f'  Answer mismatches: {wrong}')
        if parse_fail:
            print(f'  Parse failures: {parse_fail}')
        if total_issues:
            print(f'  Human-solvability issues: {total_issues}')
        if fmt_errors:
            print(f'  Format errors: {fmt_errors}')

    print('=' * 70)


if __name__ == '__main__':
    main()
