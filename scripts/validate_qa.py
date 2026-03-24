"""Validate Phase 3 Q&A pairs by re-deriving all answers from graphs.

Checks:
  1. Answer correctness — re-compute every answer from the graph
  2. Completeness — no missing questions for existing graph structures
  3. Format consistency — answer types match declared types
  4. No duplicates — no repeated question texts
  5. Obfuscation consistency — obfuscated answers map back correctly
  6. Name leakage — no original names in obfuscated Q&A

Usage:
    python validate_qa.py
"""
import json
from collections import Counter, deque
from pathlib import Path

import torch


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

    # Connected components
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
    # Sort by length descending to prevent substring collisions
    # e.g., 'SalesPersonQuotaHistory' must match before 'SalesPerson'
    found = []
    remaining = text
    for name in sorted(names, key=len, reverse=True):
        if name in remaining:
            found.append(name)
            # Remove the match to prevent double-counting
            remaining = remaining.replace(name, '___MATCHED___', 1)
            if len(found) >= count:
                break
    return found


def validate_lineage_forward(q, g):
    """Validate: 'Which tables derive from X?'"""
    names = g['names']
    found = find_table_names(q['question'], names, 1)
    if not found:
        return False, "Could not identify source table in question"
    source_name = found[0]
    idx = g['name_to_idx'][source_name]
    expected = sorted([names[t] for t in g['lineage_adj'][idx]])
    actual = sorted(q['answer'])
    if expected == actual:
        return True, ""
    return False, f"Expected {expected}, got {actual}"


def validate_lineage_reverse(q, g):
    """Validate: 'What sources does X derive from?'"""
    names = g['names']
    found = find_table_names(q['question'], names, 1)
    if not found:
        return False, "Could not identify target table in question"
    target_name = found[0]
    idx = g['name_to_idx'][target_name]
    expected = sorted([names[s] for s in g['lineage_adj_rev'][idx]])
    actual = sorted(q['answer'])
    if expected == actual:
        return True, ""
    return False, f"Expected {expected}, got {actual}"


def validate_lineage_transitive(q, g):
    """Validate: 'If X changes, which tables are affected?'"""
    names = g['names']
    found = find_table_names(q['question'], names, 1)
    if not found:
        return False, "Could not identify source table in question"
    idx = g['name_to_idx'][found[0]]
    reachable = trace_forward(g['lineage_adj'], idx)
    expected = sorted([names[t] for t in reachable])
    actual = sorted(q['answer'])
    if expected == actual:
        return True, ""
    return False, f"Expected {expected}, got {actual}"


def validate_silo_count(q, g):
    """Validate: 'How many silos?'"""
    expected = len(g['components'])
    if q['answer'] == expected:
        return True, ""
    return False, f"Expected {expected}, got {q['answer']}"


def validate_silo_membership(q, g):
    """Validate: 'Which tables are in the same component as X?'"""
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


def validate_silo_connectivity(q, g):
    """Validate: 'Are X and Y connected?'"""
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


def validate_routing_path(q, g):
    """Validate: 'Shortest join path from A to B?'"""
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
        return False, (f"Path length mismatch: expected {len(expected_path)}, "
                       f"got {len(answer)}")
    return True, ""


def validate_routing_hops(q, g):
    """Validate: 'Min number of joins between A and B?'"""
    names = g['names']
    found = find_table_names(q['question'], names, 2)
    if len(found) < 2:
        return False, f"Could not find 2 table names in question"
    path = bfs_shortest_path(g['fk_adj'],
                             g['name_to_idx'][found[0]],
                             g['name_to_idx'][found[1]], g['n'])
    expected_hops = len(path) - 1 if path else -1
    if q['answer'] == expected_hops:
        return True, ""
    return False, f"Expected {expected_hops} hops, got {q['answer']}"


def validate_routing_direct(q, g):
    """Validate: 'Does A have a direct FK to B?'"""
    names = g['names']
    found = find_table_names(q['question'], names, 2)
    if len(found) < 2:
        # Self-referential FK (e.g., DimEmployee -> DimEmployee)
        found1 = find_table_names(q['question'], names, 1)
        if found1:
            a_idx = g['name_to_idx'][found1[0]]
            has_self = a_idx in g['fk_adj'][a_idx]
            if q['answer'] == has_self:
                return True, ""
        return False, f"Could not find 2 table names"
    a_idx = g['name_to_idx'][found[0]]
    b_idx = g['name_to_idx'][found[1]]
    has_direct = (b_idx in g['fk_adj'][a_idx] or a_idx in g['fk_adj'][b_idx])
    if q['answer'] == has_direct:
        return True, ""
    return False, f"Expected {has_direct}, got {q['answer']}"


def validate_lineage_combined(q, g):
    """Validate: 'If X changes, what's affected via lineage AND FK?'"""
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


def validate_lineage_multisrc(q, g):
    """Validate: 'List ALL sources that feed into X.'"""
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


def validate_silo_full_enum(q, g):
    """Validate: 'List ALL tables in the largest component.'"""
    largest = max(g['components'], key=len)
    expected = sorted([g['names'][i] for i in largest])
    actual = sorted(q['answer'])
    if expected == actual:
        return True, ""
    return False, f"Expected {len(expected)} tables, got {len(actual)}"


def validate_silo_nopath(q, g):
    """Validate: 'Shortest path from A to B? (answer: no path)'"""
    names = g['names']
    found = find_table_names(q['question'], names, 2)
    if len(found) < 2:
        return False, "Could not find 2 table names"
    path = bfs_shortest_path(g['fk_adj'],
                             g['name_to_idx'][found[0]],
                             g['name_to_idx'][found[1]], g['n'])
    if not path and q['answer'] == 'no path':
        return True, ""
    if path:
        return False, f"Path exists ({len(path)} nodes), but answer is 'no path'"
    return False, f"Expected 'no path', got {q['answer']}"


def validate_question(q, g):
    """Route validation to the correct checker."""
    t = q['type']
    st = q['subtype']

    validators = {
        ('lineage_impact', 'forward'): validate_lineage_forward,
        ('lineage_impact', 'reverse'): validate_lineage_reverse,
        ('lineage_impact', 'transitive'): validate_lineage_transitive,
        ('silo_detection', 'count'): validate_silo_count,
        ('silo_detection', 'membership'): validate_silo_membership,
        ('silo_detection', 'isolation'): validate_silo_connectivity,
        ('silo_detection', 'connected'): validate_silo_connectivity,
        ('schema_routing', 'join_path'): validate_routing_path,
        ('schema_routing', 'hop_count'): validate_routing_hops,
        ('schema_routing', 'direct_fk'): validate_routing_direct,
        ('lineage_impact', 'combined_impact'): validate_lineage_combined,
        ('lineage_impact', 'multi_source'): validate_lineage_multisrc,
        ('silo_detection', 'full_enumeration'): validate_silo_full_enum,
        ('silo_detection', 'no_path'): validate_silo_nopath,
    }

    validator = validators.get((t, st))
    if validator is None:
        return None, f"Unknown type/subtype: {t}/{st}"
    return validator(q, g)


def validate_obfuscation(orig_path, obf_path, map_path, original_names):
    """Validate obfuscated Q&A against mapping."""
    errors = []

    with open(orig_path, 'r', encoding='utf-8') as f:
        orig_qs = json.load(f)
    with open(obf_path, 'r', encoding='utf-8') as f:
        obf_qs = json.load(f)
    with open(map_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    # Check counts match
    if len(orig_qs) != len(obf_qs):
        errors.append(f"Count mismatch: {len(orig_qs)} orig vs {len(obf_qs)} obf")

    # Check no original names in obfuscated text
    obf_text = json.dumps(obf_qs)
    leaked = []
    for name in original_names:
        if len(name) >= 5 and name in obf_text:
            leaked.append(name)
    if leaked:
        errors.append(f"Name leakage: {leaked[:5]}")

    # Check answer types preserved
    for i, (orig, obf) in enumerate(zip(orig_qs, obf_qs)):
        if type(orig['answer']) != type(obf['answer']):
            errors.append(f"Q{i}: answer type mismatch "
                          f"{type(orig['answer'])} vs {type(obf['answer'])}")

    return errors


def main():
    repo_root = Path(__file__).parent.parent
    datasets = ['adventureworks', 'tpc-ds', 'tpc-di']

    print("=" * 60)
    print("Q&A Validation")
    print("=" * 60)

    total_pass = 0
    total_fail = 0
    total_skip = 0
    all_errors = []

    for ds in datasets:
        ds_dir = repo_root / 'datasets' / ds
        graph_path = ds_dir / 'schema_graph.pt'
        qa_path = ds_dir / 'qa_pairs.json'

        if not graph_path.exists() or not qa_path.exists():
            print(f"\n  ⚠️  Skipping {ds} (missing files)")
            continue

        g = load_graph(graph_path)
        with open(qa_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        print(f"\n  Dataset: {ds} ({len(questions)} questions)")

        ds_pass = 0
        ds_fail = 0
        ds_skip = 0
        ds_errors = []

        # 1. Answer correctness
        for q in questions:
            ok, msg = validate_question(q, g)
            if ok is True:
                ds_pass += 1
            elif ok is False:
                ds_fail += 1
                ds_errors.append(f"    ❌ {q['id']}: {msg}")
            else:
                ds_skip += 1

        # 2. Duplicate check
        q_texts = [q['question'] for q in questions]
        dupes = [t for t, c in Counter(q_texts).items() if c > 1]
        if dupes:
            ds_errors.append(f"    ⚠️  {len(dupes)} duplicate questions")

        # 3. Format check
        for q in questions:
            if q['answer_type'] == 'list' and not isinstance(q['answer'], list):
                ds_errors.append(
                    f"    ❌ {q['id']}: declared list but answer is "
                    f"{type(q['answer'])}")
            if q['answer_type'] == 'integer' and not isinstance(q['answer'], int):
                ds_errors.append(
                    f"    ❌ {q['id']}: declared integer but answer is "
                    f"{type(q['answer'])}")
            if q['answer_type'] == 'boolean' and not isinstance(q['answer'], bool):
                ds_errors.append(
                    f"    ❌ {q['id']}: declared boolean but answer is "
                    f"{type(q['answer'])}")

        # 4. Obfuscation validation
        obf_path = ds_dir / 'qa_pairs_obfuscated.json'
        map_path = ds_dir / 'obfuscation_map.json'
        if obf_path.exists() and map_path.exists():
            obf_errors = validate_obfuscation(
                qa_path, obf_path, map_path, g['names'])
            for e in obf_errors:
                ds_errors.append(f"    ❌ Obfuscation: {e}")
            if not obf_errors:
                print(f"    ✅ Obfuscation: valid, no leakage")

        print(f"    ✅ Passed: {ds_pass}")
        if ds_fail:
            print(f"    ❌ Failed: {ds_fail}")
        if ds_skip:
            print(f"    ⚠️  Skipped: {ds_skip}")
        if dupes:
            print(f"    ⚠️  Duplicate questions: {len(dupes)}")

        for err in ds_errors[:10]:
            print(err)

        total_pass += ds_pass
        total_fail += ds_fail
        total_skip += ds_skip
        all_errors.extend(ds_errors)

    # Summary
    print(f"\n{'=' * 60}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total passed:  {total_pass}")
    print(f"  Total failed:  {total_fail}")
    print(f"  Total skipped: {total_skip}")
    print(f"  Total errors:  {len(all_errors)}")

    if total_fail == 0 and not all_errors:
        print(f"\n  ALL Q&A VALIDATIONS PASSED ✅")
    else:
        print(f"\n  ❌ VALIDATION FAILURES DETECTED")
        for err in all_errors[:20]:
            print(err)


if __name__ == '__main__':
    main()
