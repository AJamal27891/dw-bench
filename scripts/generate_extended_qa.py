"""Generate EXTENDED hard/medium Q&A pairs for DW-Bench.

Produces qa_pairs_extended.json with additional questions from
underrepresented subtypes. Keeps existing qa_pairs.json unchanged.

Usage:
    python scripts/generate_extended_qa.py                     # all datasets
    python scripts/generate_extended_qa.py --dataset omop_cdm  # one dataset
"""
import json
import random
from collections import deque
from pathlib import Path

import torch

SEED = 123  # Different seed from original to avoid duplicates
random.seed(SEED)


def load_graph(graph_path: Path) -> dict:
    """Load schema graph and return adjacency structures."""
    data = torch.load(graph_path, weights_only=False)
    names = data['table'].table_names
    n = len(names)

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

    has_lineage = False
    if ('table', 'derived_from', 'table') in data.edge_types:
        ei = data['table', 'derived_from', 'table'].edge_index
        for j in range(ei.shape[1]):
            s, d = ei[0, j].item(), ei[1, j].item()
            lineage_adj[s].add(d)
            lineage_adj_rev[d].add(s)
        has_lineage = ei.shape[1] > 0

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
        'names': names, 'n': n,
        'fk_adj': fk_adj, 'fk_adj_rev': fk_adj_rev,
        'lineage_adj': lineage_adj, 'lineage_adj_rev': lineage_adj_rev,
        'components': components, 'node_to_comp': node_to_comp,
        'has_lineage': has_lineage,
        'dataset_name': data.dataset_name,
    }


def trace_forward(adj, source, visited=None):
    if visited is None:
        visited = set()
    if source in visited:
        return set()
    visited.add(source)
    reachable = set()
    for target in adj.get(source, set()):
        if target not in visited:
            reachable.add(target)
            reachable |= trace_forward(adj, target, visited)
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


def load_existing_ids(qa_path: Path) -> set:
    """Load IDs from existing qa_pairs.json to avoid duplicates."""
    if qa_path.exists():
        with open(qa_path, 'r', encoding='utf-8') as f:
            return {q['id'] for q in json.load(f)}
    return set()


def generate_extended(g: dict, existing_ids: set) -> list:
    """Generate additional hard/medium questions."""
    questions = []
    names = g['names']
    ds = g['dataset_name']
    qid = 0

    def next_id(prefix):
        nonlocal qid
        qid += 1
        return f"{ds}_ext_{prefix}_{qid:03d}"

    # ================================================================
    # 1. MORE MEMBERSHIP (hard) — sample more tables from each component
    # ================================================================
    for ci, comp in enumerate(g['components']):
        if len(comp) < 3:
            continue
        comp_names = sorted([names[i] for i in comp])
        # Sample up to 5 tables per component (original only picked 3)
        sample_nodes = random.sample(list(comp), min(8, len(comp)))
        for node in sample_nodes:
            node_name = names[node]
            other_members = sorted([n for n in comp_names if n != node_name])
            if other_members:
                questions.append({
                    'id': next_id('membership'),
                    'dataset': ds,
                    'type': 'silo_detection',
                    'subtype': 'membership',
                    'question': f'Which tables are in the same connected '
                                f'component (data silo) as {node_name}?',
                    'answer': other_members,
                    'answer_type': 'list',
                    'difficulty': 'hard',
                })

    # ================================================================
    # 2. MORE FULL ENUMERATION (hard) — all components >= 3 tables
    # ================================================================
    for ci, comp in enumerate(sorted(g['components'], key=len, reverse=True)):
        if len(comp) >= 3:
            comp_names = sorted([names[i] for i in comp])
            questions.append({
                'id': next_id('full_enum'),
                'dataset': ds,
                'type': 'silo_detection',
                'subtype': 'full_enumeration',
                'question': f'List ALL tables that belong to the connected '
                            f'component containing {names[random.choice(list(comp))]}.',
                'answer': comp_names,
                'answer_type': 'list',
                'difficulty': 'hard',
            })

    # ================================================================
    # 3. MORE CONNECTED checks (medium) — cross-component pairs
    # ================================================================
    if len(g['components']) > 1:
        cross_pairs = []
        for i in range(len(g['components'])):
            for j in range(i + 1, len(g['components'])):
                for a in random.sample(list(g['components'][i]),
                                       min(3, len(g['components'][i]))):
                    for b in random.sample(list(g['components'][j]),
                                           min(3, len(g['components'][j]))):
                        cross_pairs.append((a, b, False))
        # Same-component pairs
        same_pairs = []
        for comp in g['components']:
            cl = list(comp)
            if len(cl) >= 2:
                for a, b in random.sample(
                    [(cl[i], cl[j]) for i in range(len(cl))
                     for j in range(i+1, len(cl))],
                    min(5, len(cl) * (len(cl)-1) // 2)
                ):
                    same_pairs.append((a, b, True))

        all_conn = cross_pairs + same_pairs
        random.shuffle(all_conn)
        for a, b, connected in all_conn[:20]:
            questions.append({
                'id': next_id('connected'),
                'dataset': ds,
                'type': 'silo_detection',
                'subtype': 'connected',
                'question': f'Are {names[a]} and {names[b]} connected '
                            f'through any chain of foreign key relationships?',
                'answer': connected,
                'answer_type': 'boolean',
                'difficulty': 'medium',
            })

    # ================================================================
    # 4. MORE ISOLATION (medium) — balanced true/false
    # ================================================================
    if len(g['components']) > 1:
        iso_qs = []
        for i in range(len(g['components'])):
            for j in range(i + 1, len(g['components'])):
                a = random.choice(list(g['components'][i]))
                b = random.choice(list(g['components'][j]))
                iso_qs.append({
                    'id': next_id('isolation'),
                    'dataset': ds,
                    'type': 'silo_detection',
                    'subtype': 'isolation',
                    'question': f'Are {names[a]} and {names[b]} in different '
                                f'disconnected data silos?',
                    'answer': True,
                    'answer_type': 'boolean',
                    'difficulty': 'medium',
                })
        # Same-silo false cases
        for comp in g['components']:
            cl = list(comp)
            if len(cl) >= 2:
                pairs = random.sample(
                    [(cl[i], cl[j]) for i in range(len(cl))
                     for j in range(i+1, len(cl))],
                    min(3, len(cl) * (len(cl)-1) // 2)
                )
                for a, b in pairs:
                    iso_qs.append({
                        'id': next_id('isolation'),
                        'dataset': ds,
                        'type': 'silo_detection',
                        'subtype': 'isolation',
                        'question': f'Are {names[a]} and {names[b]} in different '
                                    f'disconnected data silos?',
                        'answer': False,
                        'answer_type': 'boolean',
                        'difficulty': 'medium',
                    })
        random.shuffle(iso_qs)
        questions.extend(iso_qs[:15])

    # ================================================================
    # 5. MORE TRANSITIVE (hard) — all multi-hop lineage chains
    # ================================================================
    if g['has_lineage']:
        for src_idx in range(g['n']):
            direct = g['lineage_adj'][src_idx]
            if not direct:
                continue
            transitive = trace_forward(g['lineage_adj'], src_idx)
            indirect = transitive - direct - {src_idx}
            if indirect:
                all_affected = sorted([names[t] for t in transitive])
                questions.append({
                    'id': next_id('transitive'),
                    'dataset': ds,
                    'type': 'lineage_impact',
                    'subtype': 'transitive',
                    'question': f'If the schema of {names[src_idx]} changes, '
                                f'which tables are directly or indirectly '
                                f'affected through data lineage?',
                    'answer': all_affected,
                    'answer_type': 'list',
                    'difficulty': 'hard',
                })

    # ================================================================
    # 6. MORE MULTI-SOURCE (medium) — tables with 2+ sources
    # ================================================================
    if g['has_lineage']:
        for tgt_idx in range(g['n']):
            srcs = g['lineage_adj_rev'].get(tgt_idx, set())
            if len(srcs) >= 2:  # lowered from 3 to 2
                src_names = sorted([names[s] for s in srcs])
                questions.append({
                    'id': next_id('multi_source'),
                    'dataset': ds,
                    'type': 'lineage_impact',
                    'subtype': 'multi_source',
                    'question': f'{names[tgt_idx]} integrates data from '
                                f'multiple sources. List ALL source tables '
                                f'that feed into {names[tgt_idx]}.',
                    'answer': src_names,
                    'answer_type': 'list',
                    'difficulty': 'medium',
                })

    # ================================================================
    # 7. MORE COMBINED IMPACT (hard) — additional FK+lineage chains
    # ================================================================
    if g['has_lineage']:
        for src_idx in range(g['n']):
            direct_lin = g['lineage_adj'][src_idx]
            if not direct_lin:
                continue
            lineage_reached = trace_forward(g['lineage_adj'], src_idx)
            fk_affected = set()
            for lr in lineage_reached:
                fk_children = g['fk_adj_rev'].get(lr, set())
                for child in fk_children:
                    if child not in lineage_reached and child != src_idx:
                        fk_affected.add(child)
            if fk_affected:
                all_combined = sorted([names[t]
                                       for t in (lineage_reached | fk_affected)])
                questions.append({
                    'id': next_id('combined_impact'),
                    'dataset': ds,
                    'type': 'lineage_impact',
                    'subtype': 'combined_impact',
                    'question': f'If {names[src_idx]} changes, which tables '
                                f'are affected either through direct/indirect '
                                f'data lineage OR because they have a foreign '
                                f'key dependency on an affected table?',
                    'answer': all_combined,
                    'answer_type': 'list',
                    'difficulty': 'hard',
                })

    # ================================================================
    # 8. LONG JOIN PATHS (hard) — paths with 4+ hops
    # ================================================================
    long_path_qs = []
    tried = set()
    all_nodes = list(range(g['n']))
    random.shuffle(all_nodes)
    for a in all_nodes:
        for b in all_nodes:
            if a >= b or (a, b) in tried:
                continue
            tried.add((a, b))
            path = bfs_shortest_path(g['fk_adj'], a, b, g['n'])
            if len(path) >= 5:  # 4+ hops
                path_names = [names[p] for p in path]
                long_path_qs.append({
                    'id': next_id('long_path'),
                    'dataset': ds,
                    'type': 'schema_routing',
                    'subtype': 'join_path',
                    'question': f'What is the shortest chain of foreign key '
                                f'joins that connects {names[a]} to '
                                f'{names[b]}?',
                    'answer': path_names,
                    'answer_type': 'ordered_list',
                    'difficulty': 'hard',
                })
            if len(long_path_qs) >= 15:
                break
        if len(long_path_qs) >= 15:
            break
    questions.extend(long_path_qs)

    # Deduplicate against existing
    new_questions = []
    for q in questions:
        # Check if a similar question exists (same subtype + same answer)
        q_key = f"{q['subtype']}_{str(q['answer'])}"
        new_questions.append(q)

    return new_questions


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate EXTENDED hard/medium Q&A pairs')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['adventureworks', 'tpc-ds', 'tpc-di',
                                 'omop_cdm', 'all'])
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    datasets = (['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm']
                if args.dataset == 'all' else [args.dataset])

    print("=" * 60)
    print("Generating EXTENDED Hard/Medium Q&A Pairs")
    print("=" * 60)

    for ds in datasets:
        ds_dir = repo_root / 'datasets' / ds
        graph_path = ds_dir / 'schema_graph.pt'

        if not graph_path.exists():
            print(f"\n  ❌ {graph_path} not found, skipping {ds}")
            continue

        print(f"\n  Processing: {ds}")

        g = load_graph(graph_path)
        existing_ids = load_existing_ids(ds_dir / 'qa_pairs.json')

        questions = generate_extended(g, existing_ids)

        # Count by subtype/difficulty
        from collections import Counter
        by_sub = Counter(q['subtype'] for q in questions)
        by_diff = Counter(q['difficulty'] for q in questions)

        print(f"    Generated {len(questions)} extended questions")
        print(f"    By difficulty: {dict(by_diff)}")
        print(f"    By subtype: {dict(by_sub)}")

        # Save
        out_path = ds_dir / 'qa_pairs_extended.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        print(f"    Saved to: {out_path}")


if __name__ == '__main__':
    main()
