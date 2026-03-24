"""Validate EXTENDED Q&A pairs (hard/medium) by re-deriving answers from graphs.

Two checks:
  1. Gold answer correctness — re-compute every answer from the schema graph
  2. Result file consistency — gold_answer in result files matches QA source

Covers all 4 datasets × 3 baselines (12 result files).

Usage:
    python scripts/validate_extended_qa.py
"""
import json
from collections import Counter, deque
from pathlib import Path

import torch


# ── Graph loading & algorithms (same as validate_qa.py / generate_extended_qa.py) ──

def load_graph(graph_path):
    """Load graph and build adjacency structures."""
    data = torch.load(graph_path, weights_only=False)
    names = data['table'].table_names
    n = len(names)
    name_to_idx = {name: i for i, name in enumerate(names)}

    fk_adj = {i: set() for i in range(n)}
    fk_adj_rev = {i: set() for i in range(n)}
    lineage_adj = {i: set() for i in range(n)}
    lineage_adj_rev = {i: set() for i in range(n)}

    if ('table', 'fk_to', 'table') in data.edge_types:
        ei = data['table', 'fk_to', 'table'].edge_index
        for j in range(ei.shape[1]):
            s, d = ei[0, j].item(), ei[1, j].item()
            fk_adj[s].add(d)
            fk_adj_rev[d].add(s)

    if ('table', 'derived_from', 'table') in data.edge_types:
        ei = data['table', 'derived_from', 'table'].edge_index
        for j in range(ei.shape[1]):
            s, d = ei[0, j].item(), ei[1, j].item()
            lineage_adj[s].add(d)
            lineage_adj_rev[d].add(s)

    # Connected components (undirected view)
    all_adj = {i: set() for i in range(n)}
    for i in range(n):
        all_adj[i] = fk_adj[i] | fk_adj_rev[i] | lineage_adj[i] | lineage_adj_rev[i]
    components = []
    visited = set()
    for i in range(n):
        if i not in visited:
            comp = set()
            queue = deque([i])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                comp.add(node)
                for nb in all_adj[node]:
                    if nb not in visited:
                        queue.append(nb)
            components.append(comp)

    node_to_comp = {}
    for ci, comp in enumerate(components):
        for node in comp:
            node_to_comp[node] = ci

    return {
        'names': names, 'name_to_idx': name_to_idx, 'n': n,
        'fk_adj': fk_adj, 'fk_adj_rev': fk_adj_rev,
        'lineage_adj': lineage_adj, 'lineage_adj_rev': lineage_adj_rev,
        'components': components, 'node_to_comp': node_to_comp,
    }


def trace_forward(adj, source, visited=None):
    if visited is None:
        visited = set()
    if source in visited:
        return set()
    visited.add(source)
    reachable = set()
    for tgt in adj.get(source, set()):
        if tgt not in visited:
            reachable.add(tgt)
            reachable |= trace_forward(adj, tgt, visited)
    return reachable


def bfs_shortest_path(adj, start, end, n):
    if start == end:
        return [start]
    visited = {start}
    queue = deque([(start, [start])])
    undir = {i: set() for i in range(n)}
    for i, neighbors in adj.items():
        for j in neighbors:
            undir[i].add(j)
            undir[j].add(i)
    while queue:
        node, path = queue.popleft()
        for nb in undir[node]:
            if nb == end:
                return path + [end]
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return []


def find_table_names(text, names, count=1):
    """Extract table name(s) from question text using longest-match-first."""
    found = []
    remaining = text
    for name in sorted(names, key=len, reverse=True):
        if name in remaining:
            found.append(name)
            remaining = remaining.replace(name, '___MATCHED___', 1)
            if len(found) >= count:
                break
    return found


# ── Per-subtype validators ──

def validate_membership(q, g):
    """membership (hard): All other tables in same component."""
    names = g['names']
    found = find_table_names(q['question'], names, 1)
    if not found:
        return False, "Could not identify table in question"
    idx = g['name_to_idx'][found[0]]
    comp_id = g['node_to_comp'][idx]
    comp = g['components'][comp_id]
    expected = sorted([names[i] for i in comp if i != idx])
    actual = sorted(q['answer'])
    if expected == actual:
        return True, ""
    return False, f"Expected {len(expected)} members, got {len(actual)}"


def validate_full_enumeration(q, g):
    """full_enumeration (hard): All tables in the component containing the named table."""
    names = g['names']
    found = find_table_names(q['question'], names, 1)
    if not found:
        return False, "Could not identify table in question"
    idx = g['name_to_idx'][found[0]]
    comp_id = g['node_to_comp'][idx]
    comp = g['components'][comp_id]
    expected = sorted([names[i] for i in comp])
    actual = sorted(q['answer'])
    if expected == actual:
        return True, ""
    return False, f"Expected {len(expected)} tables, got {len(actual)}"


def validate_connected(q, g):
    """connected (medium): Are two tables in the same component?"""
    names = g['names']
    found = find_table_names(q['question'], names, 2)
    if len(found) < 2:
        return False, f"Could not identify 2 tables, found {len(found)}"
    a_comp = g['node_to_comp'][g['name_to_idx'][found[0]]]
    b_comp = g['node_to_comp'][g['name_to_idx'][found[1]]]
    expected = (a_comp == b_comp)
    if q['answer'] == expected:
        return True, ""
    return False, f"Expected {expected}, got {q['answer']}"


def validate_isolation(q, g):
    """isolation (medium): Are two tables in different silos?"""
    names = g['names']
    found = find_table_names(q['question'], names, 2)
    if len(found) < 2:
        return False, f"Could not identify 2 tables, found {len(found)}"
    a_comp = g['node_to_comp'][g['name_to_idx'][found[0]]]
    b_comp = g['node_to_comp'][g['name_to_idx'][found[1]]]
    expected = (a_comp != b_comp)
    if q['answer'] == expected:
        return True, ""
    return False, f"Expected {expected}, got {q['answer']}"


def validate_transitive(q, g):
    """transitive (hard): All transitively affected tables via lineage."""
    names = g['names']
    found = find_table_names(q['question'], names, 1)
    if not found:
        return False, "Could not identify source table"
    idx = g['name_to_idx'][found[0]]
    reachable = trace_forward(g['lineage_adj'], idx)
    expected = sorted([names[t] for t in reachable])
    actual = sorted(q['answer'])
    if expected == actual:
        return True, ""
    return False, f"Expected {expected}, got {actual}"


def validate_multi_source(q, g):
    """multi_source (medium): All source tables feeding into X."""
    names = g['names']
    found = find_table_names(q['question'], names, 1)
    if not found:
        return False, "Could not identify target table"
    idx = g['name_to_idx'][found[0]]
    expected = sorted([names[s] for s in g['lineage_adj_rev'][idx]])
    actual = sorted(q['answer'])
    if expected == actual:
        return True, ""
    return False, f"Expected {expected}, got {actual}"


def validate_combined_impact(q, g):
    """combined_impact (hard): Lineage + FK cascade."""
    names = g['names']
    found = find_table_names(q['question'], names, 1)
    if not found:
        return False, "Could not identify source table"
    src_idx = g['name_to_idx'][found[0]]
    lineage_reached = trace_forward(g['lineage_adj'], src_idx)
    fk_affected = set()
    for lr in lineage_reached:
        for child in g.get('fk_adj_rev', {}).get(lr, set()):
            if child not in lineage_reached and child != src_idx:
                fk_affected.add(child)
    expected = sorted([names[t] for t in (lineage_reached | fk_affected)])
    actual = sorted(q['answer'])
    if expected == actual:
        return True, ""
    return False, f"Expected {len(expected)} tables, got {len(actual)}"


def validate_join_path(q, g):
    """join_path (hard): Shortest FK path between two tables."""
    answer = q['answer']
    if not isinstance(answer, list) or len(answer) < 2:
        return False, f"Answer is not a valid path: {answer}"
    start_name, end_name = answer[0], answer[-1]
    if start_name not in g['name_to_idx'] or end_name not in g['name_to_idx']:
        return False, "Path endpoints not in graph"
    start_idx = g['name_to_idx'][start_name]
    end_idx = g['name_to_idx'][end_name]
    expected_path = bfs_shortest_path(g['fk_adj'], start_idx, end_idx, g['n'])
    if len(expected_path) != len(answer):
        expected_names = [g['names'][i] for i in expected_path] if expected_path else []
        return False, (f"Path length mismatch: expected {len(expected_path)} "
                       f"({expected_names}), got {len(answer)} ({answer})")
    # Also verify each hop is valid
    expected_names = [g['names'][i] for i in expected_path]
    if expected_names != answer:
        return False, f"Path mismatch: expected {expected_names}, got {answer}"
    return True, ""


def validate_question(q, g):
    """Route validation to the correct checker."""
    st = q['subtype']
    validators = {
        'membership': validate_membership,
        'full_enumeration': validate_full_enumeration,
        'connected': validate_connected,
        'isolation': validate_isolation,
        'transitive': validate_transitive,
        'multi_source': validate_multi_source,
        'combined_impact': validate_combined_impact,
        'join_path': validate_join_path,
    }
    validator = validators.get(st)
    if validator is None:
        return None, f"Unknown subtype: {st}"
    return validator(q, g)


# ── Main ──

def main():
    repo_root = Path(__file__).parent.parent
    datasets = ['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm']
    baselines = ['flat_text', 'vector_rag', 'graph_aug']

    print("=" * 70)
    print("EXTENDED Q&A VALIDATION (Hard + Medium)")
    print("=" * 70)

    grand_pass = 0
    grand_fail = 0
    grand_skip = 0
    all_errors = []

    # ── Phase 1: Gold answer correctness ──
    print("\n" + "─" * 70)
    print("PHASE 1: Re-derive gold answers from schema graphs")
    print("─" * 70)

    for ds in datasets:
        ds_dir = repo_root / 'datasets' / ds
        graph_path = ds_dir / 'schema_graph.pt'
        qa_path = ds_dir / 'qa_pairs_extended.json'

        if not graph_path.exists():
            print(f"\n  ⚠️  Skipping {ds} (no schema_graph.pt)")
            continue
        if not qa_path.exists():
            print(f"\n  ⚠️  Skipping {ds} (no qa_pairs_extended.json)")
            continue

        g = load_graph(graph_path)
        with open(qa_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        # Filter to hard + medium only
        hm_questions = [q for q in questions
                        if q.get('difficulty') in ('hard', 'medium')]

        print(f"\n  Dataset: {ds}")
        print(f"    Total extended Qs: {len(questions)}")
        print(f"    Hard+Medium: {len(hm_questions)}")

        ds_pass = 0
        ds_fail = 0
        ds_skip = 0
        ds_errors = []

        # Validate each question
        for q in hm_questions:
            ok, msg = validate_question(q, g)
            if ok is True:
                ds_pass += 1
            elif ok is False:
                ds_fail += 1
                ds_errors.append(
                    f"    ❌ {q['id']} [{q['subtype']}/{q['difficulty']}]: {msg}")
            else:
                ds_skip += 1
                ds_errors.append(
                    f"    ⚠️  {q['id']}: {msg}")

        # Duplicate check
        q_texts = [q['question'] for q in hm_questions]
        dupes = [t for t, c in Counter(q_texts).items() if c > 1]

        # Format check
        for q in hm_questions:
            atype = q.get('answer_type', '')
            ans = q['answer']
            if atype in ('list', 'ordered_list') and not isinstance(ans, list):
                ds_errors.append(
                    f"    ❌ {q['id']}: declared {atype} but answer is "
                    f"{type(ans).__name__}")
            if atype == 'integer' and not isinstance(ans, int):
                ds_errors.append(
                    f"    ❌ {q['id']}: declared integer but answer is "
                    f"{type(ans).__name__}")
            if atype == 'boolean' and not isinstance(ans, bool):
                ds_errors.append(
                    f"    ❌ {q['id']}: declared boolean but answer is "
                    f"{type(ans).__name__}")

        # Per-subtype breakdown
        by_sub = Counter()
        pass_by_sub = Counter()
        fail_by_sub = Counter()
        for q in hm_questions:
            key = f"{q['subtype']}({q['difficulty']})"
            by_sub[key] += 1
            ok, _ = validate_question(q, g)
            if ok is True:
                pass_by_sub[key] += 1
            elif ok is False:
                fail_by_sub[key] += 1

        print(f"    ✅ Passed: {ds_pass}")
        if ds_fail:
            print(f"    ❌ Failed: {ds_fail}")
        if ds_skip:
            print(f"    ⚠️  Skipped: {ds_skip}")
        if dupes:
            print(f"    ⚠️  Duplicate questions: {len(dupes)}")

        print(f"    Per-subtype breakdown:")
        for key in sorted(by_sub.keys()):
            p = pass_by_sub.get(key, 0)
            f_count = fail_by_sub.get(key, 0)
            status = "✅" if f_count == 0 else "❌"
            print(f"      {status} {key}: {p}/{by_sub[key]} pass"
                  + (f", {f_count} FAIL" if f_count else ""))

        for err in ds_errors[:15]:
            print(err)
        if len(ds_errors) > 15:
            print(f"    ... and {len(ds_errors) - 15} more errors")

        grand_pass += ds_pass
        grand_fail += ds_fail
        grand_skip += ds_skip
        all_errors.extend(ds_errors)

    # ── Phase 2: Result file cross-check ──
    print("\n" + "─" * 70)
    print("PHASE 2: Cross-check gold answers in result files")
    print("─" * 70)

    results_dir = repo_root / 'evaluation' / 'results'
    result_mismatches = 0
    result_files_checked = 0

    for ds in datasets:
        qa_path = repo_root / 'datasets' / ds / 'qa_pairs_extended.json'
        if not qa_path.exists():
            continue
        with open(qa_path, 'r', encoding='utf-8') as f:
            source_qs = json.load(f)
        source_by_id = {q['id']: q for q in source_qs}

        for baseline in baselines:
            # Try common model tag patterns
            result_files = list(results_dir.glob(
                f"{baseline}_extended_{ds}_*.json"))
            for result_file in result_files:
                result_files_checked += 1
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)

                results = result_data.get('results', [])
                file_mismatches = 0

                for r in results:
                    qid = r.get('id', '')
                    gold_in_result = r.get('gold_answer')
                    source_q = source_by_id.get(qid)

                    if source_q is None:
                        print(f"    ⚠️  {result_file.name}: Q {qid} not found "
                              f"in source QA")
                        continue

                    source_gold = source_q['answer']

                    # Normalize for comparison
                    def normalize(val):
                        if isinstance(val, list):
                            return sorted([str(v) for v in val])
                        return str(val)

                    if normalize(gold_in_result) != normalize(source_gold):
                        file_mismatches += 1
                        result_mismatches += 1
                        if file_mismatches <= 3:
                            print(f"    ❌ {result_file.name} | Q {qid}:")
                            print(f"       Result gold:  {gold_in_result}")
                            print(f"       Source gold:  {source_gold}")

                status = "✅" if file_mismatches == 0 else "❌"
                mismatch_str = (f" ({file_mismatches} mismatches)"
                                if file_mismatches else "")
                print(f"  {status} {result_file.name}: "
                      f"{len(results)} Qs checked{mismatch_str}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Phase 1 — Gold answer re-derivation:")
    print(f"    Total passed:  {grand_pass}")
    print(f"    Total failed:  {grand_fail}")
    print(f"    Total skipped: {grand_skip}")
    print(f"  Phase 2 — Result file cross-check:")
    print(f"    Files checked:  {result_files_checked}")
    print(f"    Gold mismatches: {result_mismatches}")

    if grand_fail == 0 and result_mismatches == 0:
        print(f"\n  ALL VALIDATIONS PASSED ✅")
    else:
        print(f"\n  ❌ VALIDATION FAILURES DETECTED")
        if grand_fail:
            print(f"     {grand_fail} gold answers do not match graph")
        if result_mismatches:
            print(f"     {result_mismatches} result file golds differ from source")


if __name__ == '__main__':
    main()
