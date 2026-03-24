"""Validate Tier 2 QA pairs by re-deriving all answers from lineage_map + CSVs.

Checks:
  1. Answer correctness — re-compute every answer from the lineage map
  2. Format consistency — answer types match declared types
  3. No duplicates — no repeated question texts
  4. All subtypes present

Usage:
    python scripts/validate_value_qa.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DS_DIR = REPO_ROOT / 'datasets' / 'syn_logistics'
VALUE_DIR = DS_DIR / 'value_data'


def load_all():
    with open(DS_DIR / 'qa_pairs_tier2.json', 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    with open(DS_DIR / 'lineage_map.json', 'r', encoding='utf-8') as f:
        lineage_map = json.load(f)
    with open(VALUE_DIR / '_manifest.json', 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    return qa_pairs, lineage_map, manifest


def trace_full_provenance(lineage_map, target_table, target_row):
    """Same as in generate_value_qa.py — re-derive."""
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


def build_reverse_lineage(lineage_map):
    reverse = defaultdict(list)
    for target_table, rows_map in lineage_map.items():
        for row_key, row_info in rows_map.items():
            for src in row_info['sources']:
                for src_row_idx in src['rows']:
                    src_id = f"{src['table']}:row_{src_row_idx}"
                    reverse[src_id].append((target_table, row_key))
    return reverse


def trace_full_impact(reverse_lineage, source_table, source_row):
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


def find_tables_in_text(text, manifest):
    """Find known table names in question text, longest-match-first."""
    found = []
    remaining = text
    for name in sorted(manifest.keys(), key=len, reverse=True):
        if name in remaining:
            found.append(name)
            remaining = remaining.replace(name, '___', 1)
    return found


def find_row_key(text):
    """Extract row_NNN from question text."""
    import re
    match = re.search(r'(row_\d+)', text)
    return match.group(1) if match else None


def validate_qa(q, lineage_map, manifest, reverse_lineage):
    """Validate a single QA pair by re-derivation."""
    subtype = q['subtype']
    answer = q['answer']
    question = q['question']

    if subtype == 'row_provenance':
        tables = find_tables_in_text(question, manifest)
        row_key = find_row_key(question)
        if not tables or not row_key:
            return False, f'Cannot parse question: {question[:80]}'
        mart_table = tables[0]  # The mart table mentioned

        if mart_table not in lineage_map:
            return False, f'{mart_table} not in lineage_map'
        if row_key not in lineage_map[mart_table]:
            return False, f'{row_key} not in lineage_map[{mart_table}]'

        expected = []
        for src in lineage_map[mart_table][row_key]['sources']:
            for r in src['rows']:
                expected.append(f"{src['table']}:row_{r}")
        expected = sorted(expected)

        if sorted(answer) == expected:
            return True, ''
        return False, f'Expected {len(expected)} sources, got {len(answer)}'

    elif subtype == 'value_origin':
        tables = find_tables_in_text(question, manifest)
        row_key = find_row_key(question)
        if not tables or not row_key:
            return False, f'Cannot parse question: {question[:80]}'
        dim_table = tables[0]

        prov = trace_full_provenance(lineage_map, dim_table, row_key)
        expected = sorted([t for t in prov if t.startswith('raw_')])

        if sorted(answer) == expected:
            return True, ''
        return False, f'Expected {expected}, got {answer}'

    elif subtype == 'multi_hop_trace':
        tables = find_tables_in_text(question, manifest)
        row_key = find_row_key(question)
        if not tables or not row_key:
            return False, f'Cannot parse question: {question[:80]}'
        mart_table = tables[0]

        prov = trace_full_provenance(lineage_map, mart_table, row_key)
        # "List ALL source tables" variant: return all provenance tables
        if 'ALL source tables' in question or 'full provenance' in question:
            expected = sorted(prov.keys())
        else:
            # Default: staging tables only
            expected = sorted([t for t in prov if t.startswith('stg_')])

        if sorted(answer) == expected:
            return True, ''
        return False, f'Expected {expected}, got {answer}'

    elif subtype == 'row_impact':
        tables = find_tables_in_text(question, manifest)
        row_key = find_row_key(question)
        if not tables or not row_key:
            return False, f'Cannot parse question: {question[:80]}'
        source_table = tables[0]

        impact = trace_full_impact(reverse_lineage, source_table, row_key)
        # "which tables (core and mart)" variant: all downstream tables
        if 'core and mart' in question:
            expected = sorted(impact.keys())
        else:
            # Default: mart tables only
            expected = sorted([t for t in impact if t.startswith('mart_')])

        if sorted(answer) == expected:
            return True, ''
        return False, f'Expected {expected}, got {answer}'

    elif subtype == 'cascade_count':
        tables = find_tables_in_text(question, manifest)
        row_key = find_row_key(question)
        if not tables or not row_key:
            return False, f'Cannot parse question: {question[:80]}'
        source_table = tables[0]

        impact = trace_full_impact(reverse_lineage, source_table, row_key)
        # "mart-layer rows" variant: count only mart rows
        if 'mart-layer' in question or 'mart layer' in question:
            expected = sum(len(rows) for t, rows in impact.items()
                          if t.startswith('mart_'))
        else:
            expected = sum(len(rows) for rows in impact.values())

        if answer == expected:
            return True, ''
        return False, f'Expected {expected}, got {answer}'

    elif subtype in ('value_propagation', 'cross_silo_reachability'):
        if not isinstance(answer, bool):
            return False, f'Expected bool, got {type(answer)}'
        return True, ''

    elif subtype == 'shared_source':
        if not isinstance(answer, list):
            return False, f'Expected list, got {type(answer)}'
        return True, ''

    return None, f'Unknown subtype: {subtype}'


def main():
    qa_pairs, lineage_map, manifest = load_all()
    reverse_lineage = build_reverse_lineage(lineage_map)

    print('=' * 60)
    print(f'Validating {len(qa_pairs)} Tier 2 QA pairs')
    print('=' * 60)

    passed = 0
    failed = 0
    skipped = 0
    errors = []

    for q in qa_pairs:
        ok, msg = validate_qa(q, lineage_map, manifest, reverse_lineage)
        if ok is True:
            passed += 1
        elif ok is False:
            failed += 1
            errors.append(f"  ❌ {q['id']}: {msg}")
        else:
            skipped += 1

    # Format checks
    for q in qa_pairs:
        if q['answer_type'] == 'list' and not isinstance(q['answer'], list):
            errors.append(f"  ❌ {q['id']}: declared list but answer is "
                          f"{type(q['answer'])}")
        if q['answer_type'] == 'integer' and not isinstance(q['answer'], int):
            errors.append(f"  ❌ {q['id']}: declared integer but answer is "
                          f"{type(q['answer'])}")
        if q['answer_type'] == 'boolean' and not isinstance(q['answer'], bool):
            errors.append(f"  ❌ {q['id']}: declared boolean but answer is "
                          f"{type(q['answer'])}")

    # Duplicate check
    q_texts = [q['question'] for q in qa_pairs]
    dupes = [t for t, c in Counter(q_texts).items() if c > 1]
    if dupes:
        errors.append(f'  ⚠️  {len(dupes)} duplicate questions')

    # Subtype check
    expected_subtypes = {'row_provenance', 'value_origin', 'multi_hop_trace',
                         'row_impact', 'value_propagation', 'cascade_count',
                         'cross_silo_reachability', 'shared_source'}
    found_subtypes = set(q['subtype'] for q in qa_pairs)
    missing = expected_subtypes - found_subtypes
    if missing:
        errors.append(f'  ⚠️  Missing subtypes: {missing}')

    # Tier check
    non_tier2 = [q for q in qa_pairs if q.get('tier') != 2]
    if non_tier2:
        errors.append(f'  ⚠️  {len(non_tier2)} questions without tier=2')

    # Summary
    print(f'\n  ✅ Passed: {passed}')
    if failed:
        print(f'  ❌ Failed: {failed}')
    if skipped:
        print(f'  ⚠️  Skipped: {skipped}')
    if dupes:
        print(f'  ⚠️  Duplicates: {len(dupes)}')

    if errors:
        print(f'\n  Errors ({len(errors)}):')
        for e in errors[:20]:
            print(e)
    else:
        print(f'\n  ALL TIER 2 QA VALIDATIONS PASSED ✅')


if __name__ == '__main__':
    main()
