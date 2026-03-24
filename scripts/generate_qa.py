"""Phase 3: Generate Q&A pairs from DW-Bench schema graphs.

Produces three question types:
  1. Lineage Impact  — "Which DW tables are affected if X changes?"
  2. Silo Detection  — "Which tables are in the same/different component?"
  3. Schema Routing  — "What is the join path from A to B?"

All answers are derived programmatically from graph traversal,
guaranteeing provable correctness.

Usage:
    python generate_qa.py                    # all datasets
    python generate_qa.py --dataset tpc-di   # single dataset
"""
import argparse
import json
import random
from collections import deque
from pathlib import Path

import torch

# Reproducibility
SEED = 42
random.seed(SEED)


# ──────────────────────────────────────────────────────────────────────
# Graph utilities
# ──────────────────────────────────────────────────────────────────────

def load_graph(graph_path: Path) -> dict:
    """Load schema graph and return adjacency structures."""
    data = torch.load(graph_path, weights_only=False)
    names = data['table'].table_names
    n = len(names)
    name_to_idx = {name: i for i, name in enumerate(names)}

    # Build adjacency lists for each edge type
    fk_adj = {i: set() for i in range(n)}      # forward FK
    fk_adj_rev = {i: set() for i in range(n)}   # reverse FK
    lineage_adj = {i: set() for i in range(n)}   # source → target (derived_from)
    lineage_adj_rev = {i: set() for i in range(n)}  # target → source

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
        'names': names,
        'name_to_idx': name_to_idx,
        'n': n,
        'fk_adj': fk_adj,
        'fk_adj_rev': fk_adj_rev,
        'lineage_adj': lineage_adj,
        'lineage_adj_rev': lineage_adj_rev,
        'components': components,
        'node_to_comp': node_to_comp,
        'has_lineage': has_lineage,
        'dataset_name': data.dataset_name,
    }


def trace_forward(adj: dict, source: int, visited: set = None) -> set:
    """Trace all reachable nodes from source via forward edges (with cycle guard)."""
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


def bfs_shortest_path(adj: dict, start: int, end: int, n: int) -> list:
    """BFS shortest path on undirected FK graph. Returns path as list of node indices."""
    if start == end:
        return [start]
    visited = {start}
    queue = deque([(start, [start])])
    # Build undirected adjacency
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
    return []  # No path


# ──────────────────────────────────────────────────────────────────────
# Question generators
# ──────────────────────────────────────────────────────────────────────

def generate_lineage_questions(g: dict) -> list:
    """Generate lineage impact questions (1 per lineage edge)."""
    questions = []
    names = g['names']
    dataset = g['dataset_name']

    # Type 1a: Forward — "Which DW tables derive from source X?"
    # Group by source
    source_to_targets = {}
    for src_idx, targets in g['lineage_adj'].items():
        if targets:  # has outgoing lineage edges
            source_to_targets[src_idx] = sorted(targets)

    for src_idx, target_idxs in source_to_targets.items():
        src_name = names[src_idx]
        target_names = sorted([names[t] for t in target_idxs])
        questions.append({
            'id': f'{dataset}_lineage_fwd_{len(questions):03d}',
            'dataset': dataset,
            'type': 'lineage_impact',
            'subtype': 'forward',
            'question': f'Which tables in the data warehouse are directly '
                        f'derived from {src_name}?',
            'answer': target_names,
            'answer_type': 'list',
            'difficulty': 'easy',
        })

    # Type 1b: Reverse — "What are the source tables for DW table X?"
    target_to_sources = {}
    for tgt_idx, sources in g['lineage_adj_rev'].items():
        if sources:
            target_to_sources[tgt_idx] = sorted(sources)

    for tgt_idx, src_idxs in target_to_sources.items():
        tgt_name = names[tgt_idx]
        src_names = sorted([names[s] for s in src_idxs])
        questions.append({
            'id': f'{dataset}_lineage_rev_{len(questions):03d}',
            'dataset': dataset,
            'type': 'lineage_impact',
            'subtype': 'reverse',
            'question': f'What are all the source tables that {tgt_name} '
                        f'is derived from?',
            'answer': src_names,
            'answer_type': 'list',
            'difficulty': 'easy',
        })

    # Type 1c: Transitive — "If X changes, which tables are transitively affected?"
    # Only for sources that have multi-hop reachability
    for src_idx in source_to_targets:
        direct = g['lineage_adj'][src_idx]
        # Check if any direct target also has outgoing lineage
        transitive = trace_forward(g['lineage_adj'], src_idx)
        indirect = transitive - direct - {src_idx}
        if indirect:
            src_name = names[src_idx]
            all_affected = sorted([names[t] for t in transitive])
            questions.append({
                'id': f'{dataset}_lineage_trans_{len(questions):03d}',
                'dataset': dataset,
                'type': 'lineage_impact',
                'subtype': 'transitive',
                'question': f'If the schema of {src_name} changes, which '
                            f'tables are directly or indirectly affected '
                            f'through data lineage?',
                'answer': all_affected,
                'answer_type': 'list',
                'difficulty': 'hard',
            })

    # Type 1d: Combined lineage + FK impact (HARD)
    # "If source X changes, which tables are affected through lineage AND FK chains?"
    # Traces lineage forward, then for each reached DW table, finds all FK dependents
    for src_idx in source_to_targets:
        lineage_reached = trace_forward(g['lineage_adj'], src_idx)
        # From the source AND each lineage-reached node, follow FK edges
        # (children that FK into it). The source itself is also "affected"
        # so its FK dependents should be included.
        fk_affected = set()
        nodes_to_check = lineage_reached | {src_idx}
        for node in nodes_to_check:
            fk_children = g['fk_adj_rev'].get(node, set())
            for child in fk_children:
                if child not in nodes_to_check:
                    fk_affected.add(child)
        if fk_affected:
            src_name = names[src_idx]
            all_combined = sorted([names[t] for t in (lineage_reached | fk_affected)])
            questions.append({
                'id': f'{dataset}_lineage_combined_{len(questions):03d}',
                'dataset': dataset,
                'type': 'lineage_impact',
                'subtype': 'combined_impact',
                'question': f'If {src_name} changes, which tables are affected '
                            f'either through direct/indirect data lineage OR '
                            f'because they have a foreign key dependency on '
                            f'an affected table?',
                'answer': all_combined,
                'answer_type': 'list',
                'difficulty': 'hard',
            })

    # Type 1e: Multi-source reverse (HARD) — tables with 3+ sources
    for tgt_idx, src_idxs in target_to_sources.items():
        if len(src_idxs) >= 3:
            tgt_name = names[tgt_idx]
            src_names = sorted([names[s] for s in src_idxs])
            questions.append({
                'id': f'{dataset}_lineage_multisrc_{len(questions):03d}',
                'dataset': dataset,
                'type': 'lineage_impact',
                'subtype': 'multi_source',
                'question': f'{tgt_name} integrates data from multiple sources. '
                            f'List ALL source tables that feed into {tgt_name}.',
                'answer': src_names,
                'answer_type': 'list',
                'difficulty': 'hard',
            })

    return questions


def generate_silo_questions(g: dict) -> list:
    """Generate silo detection questions from connected components."""
    questions = []
    names = g['names']
    dataset = g['dataset_name']
    components = g['components']

    # Type 2a: Component count
    questions.append({
        'id': f'{dataset}_silo_count_000',
        'dataset': dataset,
        'type': 'silo_detection',
        'subtype': 'count',
        'question': f'How many disconnected data silos (connected components) '
                    f'exist in this schema?',
        'answer': len(components),
        'answer_type': 'integer',
        'difficulty': 'easy',
    })

    # Type 2b: Component membership — "List all tables in the same component as X"
    # Pick representative tables from larger components
    for ci, comp in enumerate(components):
        if len(comp) < 2:
            continue  # Skip singleton components
        comp_names = sorted([names[i] for i in comp])
        # Pick a random member to ask about
        sample_nodes = random.sample(list(comp), min(3, len(comp)))
        for node in sample_nodes:
            node_name = names[node]
            other_members = sorted([n for n in comp_names if n != node_name])
            if other_members:
                questions.append({
                    'id': f'{dataset}_silo_member_{len(questions):03d}',
                    'dataset': dataset,
                    'type': 'silo_detection',
                    'subtype': 'membership',
                    'question': f'Which tables are in the same connected '
                                f'component (data silo) as {node_name}?',
                    'answer': other_members,
                    'answer_type': 'list',
                    'difficulty': 'medium',
                })

    # Type 2c: Isolation — "Is table X connected to table Y?"
    # BALANCED: generate both True (same-component) and False (cross-component)
    isolation_qs = []
    if len(components) > 1:
        # False cases: cross-component pairs
        cross_pairs = []
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                ci_sample = random.sample(list(components[i]),
                                          min(2, len(components[i])))
                cj_sample = random.sample(list(components[j]),
                                          min(2, len(components[j])))
                for a in ci_sample:
                    for b in cj_sample:
                        cross_pairs.append((a, b))

        for a, b in random.sample(cross_pairs, min(5, len(cross_pairs))):
            isolation_qs.append({
                'id': f'{dataset}_silo_isolated_{len(questions) + len(isolation_qs):03d}',
                'dataset': dataset,
                'type': 'silo_detection',
                'subtype': 'isolation',
                'question': f'Are {names[a]} and {names[b]} connected through '
                            f'any chain of foreign key relationships?',
                'answer': False,
                'answer_type': 'boolean',
                'difficulty': 'medium',
            })

    # True cases: same-component pairs (balanced with False cases)
    target_true = max(len(isolation_qs), 5)
    same_pairs = []
    for comp in components:
        comp_list = list(comp)
        if len(comp_list) >= 2:
            for a, b in [(comp_list[i], comp_list[j])
                         for i in range(len(comp_list))
                         for j in range(i+1, len(comp_list))]:
                same_pairs.append((a, b))
    if same_pairs:
        for a, b in random.sample(same_pairs,
                                  min(target_true, len(same_pairs))):
            isolation_qs.append({
                'id': f'{dataset}_silo_isolated_{len(questions) + len(isolation_qs):03d}',
                'dataset': dataset,
                'type': 'silo_detection',
                'subtype': 'isolation',
                'question': f'Are {names[a]} and {names[b]} connected through '
                            f'any chain of foreign key relationships?',
                'answer': True,
                'answer_type': 'boolean',
                'difficulty': 'medium',
            })
    random.shuffle(isolation_qs)
    questions.extend(isolation_qs)

    # Type 2d: Same-component connectivity confirmation
    for ci, comp in enumerate(components):
        if len(comp) >= 3:
            sample = random.sample(list(comp), min(2, len(comp)))
            if len(sample) == 2:
                a, b = sample
                questions.append({
                    'id': f'{dataset}_silo_connected_{len(questions):03d}',
                    'dataset': dataset,
                    'type': 'silo_detection',
                    'subtype': 'connected',
                    'question': f'Are {names[a]} and {names[b]} connected '
                                f'through any chain of foreign key '
                                f'relationships?',
                    'answer': True,
                    'answer_type': 'boolean',
                    'difficulty': 'easy',
                })

    # Type 2e: Full large silo enumeration (HARD) — list ALL tables in a large silo
    for ci, comp in enumerate(components):
        if len(comp) >= 10:
            comp_names = sorted([names[i] for i in comp])
            questions.append({
                'id': f'{dataset}_silo_enumerate_{len(questions):03d}',
                'dataset': dataset,
                'type': 'silo_detection',
                'subtype': 'full_enumeration',
                'question': f'List ALL tables that belong to the largest '
                            f'connected component in this schema.',
                'answer': comp_names,
                'answer_type': 'list',
                'difficulty': 'hard',
            })
            break  # only the largest

    # Type 2f: Cross-silo path non-existence (HARD)
    # Ask about specific paths that don't exist across silos
    if len(components) > 1:
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                if len(components[i]) >= 3 and len(components[j]) >= 3:
                    a = random.choice(list(components[i]))
                    b = random.choice(list(components[j]))
                    path = bfs_shortest_path(g['fk_adj'], a, b, g['n'])
                    questions.append({
                        'id': f'{dataset}_silo_nopath_{len(questions):03d}',
                        'dataset': dataset,
                        'type': 'silo_detection',
                        'subtype': 'no_path',
                        'question': f'What is the shortest chain of foreign key '
                                    f'joins connecting {names[a]} to '
                                    f'{names[b]}? If no path exists, answer '
                                    f'"no path".',
                        'answer': 'no path',
                        'answer_type': 'string',
                        'difficulty': 'hard',
                    })

    return questions


def generate_routing_questions(g: dict, max_questions: int = 80) -> list:
    """Generate schema routing (join path) questions via BFS."""
    questions = []
    names = g['names']
    dataset = g['dataset_name']
    n = g['n']

    # Build undirected FK adjacency for path finding
    fk_undir = {i: set() for i in range(n)}
    for i, neighbors in g['fk_adj'].items():
        for j in neighbors:
            fk_undir[i].add(j)
            fk_undir[j].add(i)

    # Only consider tables with at least one FK connection
    connected_tables = [i for i in range(n) if fk_undir[i]]
    if len(connected_tables) < 2:
        return questions

    # Generate random pairs and find paths
    pairs_tried = set()
    candidates = []

    for _ in range(max_questions * 5):  # oversample then filter
        a, b = random.sample(connected_tables, 2)
        pair_key = (min(a, b), max(a, b))
        if pair_key in pairs_tried:
            continue
        pairs_tried.add(pair_key)

        path = bfs_shortest_path(g['fk_adj'], a, b, n)
        if path and len(path) >= 2:
            candidates.append((a, b, path))

    # Sort by path length for diversity (mix of short and long paths)
    candidates.sort(key=lambda x: len(x[2]))

    # Sample diverse mix: 1/3 short (2 hops), 1/3 medium (3-4), 1/3 long (5+)
    short = [c for c in candidates if len(c[2]) == 2]
    medium = [c for c in candidates if 3 <= len(c[2]) <= 4]
    long = [c for c in candidates if len(c[2]) >= 5]

    target_per_bucket = max_questions // 3
    selected = (
        random.sample(short, min(target_per_bucket, len(short))) +
        random.sample(medium, min(target_per_bucket, len(medium))) +
        random.sample(long, min(max_questions - 2 * target_per_bucket,
                                len(long)))
    )

    # Fill remaining quota
    remaining = max_questions - len(selected)
    if remaining > 0:
        unused = [c for c in candidates if c not in selected]
        selected += random.sample(unused, min(remaining, len(unused)))

    for a, b, path in selected[:max_questions]:
        path_names = [names[i] for i in path]
        hop_count = len(path) - 1
        difficulty = 'easy' if hop_count <= 2 else (
            'medium' if hop_count <= 3 else 'hard')

        # Type 3a: Path question
        questions.append({
            'id': f'{dataset}_route_path_{len(questions):03d}',
            'dataset': dataset,
            'type': 'schema_routing',
            'subtype': 'join_path',
            'question': f'What is the shortest chain of foreign key joins '
                        f'that connects {names[a]} to {names[b]}?',
            'answer': path_names,
            'answer_type': 'ordered_list',
            'difficulty': difficulty,
        })

    # Type 3b: Hop count questions
    hop_candidates = random.sample(selected,
                                   min(max_questions // 4, len(selected)))
    for a, b, path in hop_candidates:
        hop_count = len(path) - 1
        questions.append({
            'id': f'{dataset}_route_hops_{len(questions):03d}',
            'dataset': dataset,
            'type': 'schema_routing',
            'subtype': 'hop_count',
            'question': f'What is the minimum number of foreign key joins '
                        f'needed to connect {names[a]} to {names[b]}?',
            'answer': hop_count,
            'answer_type': 'integer',
            'difficulty': 'medium',
        })

    # Type 3c: Direct FK yes/no questions (BALANCED True + False)
    direct_pairs = []
    for i in connected_tables:
        for j in g['fk_adj'][i]:
            direct_pairs.append((i, j))

    fk_set = set((i, j) for i in range(g['n'])
                 for j in g['fk_adj'].get(i, []))

    direct_qs = []
    if direct_pairs:
        # True cases: actual FK pairs
        true_count = min(max_questions // 10, len(direct_pairs))
        for a, b in random.sample(direct_pairs, true_count):
            direct_qs.append({
                'id': f'{dataset}_route_direct_{len(questions) + len(direct_qs):03d}',
                'dataset': dataset,
                'type': 'schema_routing',
                'subtype': 'direct_fk',
                'question': f'Does {names[a]} have a direct foreign key '
                            f'relationship to {names[b]}?',
                'answer': True,
                'answer_type': 'boolean',
                'difficulty': 'easy',
            })

        # False cases: table pairs with NO direct FK
        non_fk_pairs = []
        for a in connected_tables:
            for b in connected_tables:
                if a != b and (a, b) not in fk_set:
                    non_fk_pairs.append((a, b))
        if non_fk_pairs:
            false_count = min(true_count, len(non_fk_pairs))
            for a, b in random.sample(non_fk_pairs, false_count):
                direct_qs.append({
                    'id': f'{dataset}_route_direct_{len(questions) + len(direct_qs):03d}',
                    'dataset': dataset,
                    'type': 'schema_routing',
                    'subtype': 'direct_fk',
                    'question': f'Does {names[a]} have a direct foreign key '
                                f'relationship to {names[b]}?',
                    'answer': False,
                    'answer_type': 'boolean',
                    'difficulty': 'easy',
                })
    random.shuffle(direct_qs)
    questions.extend(direct_qs)

    return questions


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def generate_for_dataset(graph_path: Path, output_dir: Path) -> list:
    """Generate all Q&A pairs for a single dataset."""
    g = load_graph(graph_path)
    dataset = g['dataset_name']

    print(f"\n  Generating Q&A for: {dataset}")
    print(f"    Nodes: {g['n']}, Silos: {len(g['components'])}, "
          f"Has lineage: {g['has_lineage']}")

    all_qs = []

    # Lineage questions (only for datasets with lineage)
    if g['has_lineage']:
        lineage_qs = generate_lineage_questions(g)
        print(f"    Lineage Impact: {len(lineage_qs)} questions")
        all_qs.extend(lineage_qs)

    # Silo questions
    silo_qs = generate_silo_questions(g)
    print(f"    Silo Detection: {len(silo_qs)} questions")
    all_qs.extend(silo_qs)

    # Routing questions
    routing_qs = generate_routing_questions(g)
    print(f"    Schema Routing: {len(routing_qs)} questions")
    all_qs.extend(routing_qs)

    # Save per-dataset
    out_path = output_dir / 'qa_pairs.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_qs, f, indent=2, ensure_ascii=False)
    print(f"    Saved {len(all_qs)} Q&A pairs to: {out_path}")

    return all_qs


def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: Generate Q&A pairs from schema graphs',
    )
    parser.add_argument(
        '--dataset', type=str, default='all',
        choices=['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm', 'all'],
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    datasets = (
        ['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm']
        if args.dataset == 'all'
        else [args.dataset]
    )

    print("=" * 60)
    print("Phase 3: Q&A Generation")
    print("=" * 60)

    all_questions = []
    for ds in datasets:
        ds_dir = repo_root / 'datasets' / ds
        graph_path = ds_dir / 'schema_graph.pt'
        if not graph_path.exists():
            print(f"\n  ❌ {graph_path} not found, skipping {ds}")
            continue
        qs = generate_for_dataset(graph_path, ds_dir)
        all_questions.extend(qs)

    # Save combined Q&A
    combined_path = repo_root / 'datasets' / 'qa_pairs_all.json'
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'=' * 60}")
    print("Q&A Generation Summary")
    print(f"{'=' * 60}")
    by_type = {}
    by_dataset = {}
    by_difficulty = {}
    for q in all_questions:
        by_type[q['type']] = by_type.get(q['type'], 0) + 1
        by_dataset[q['dataset']] = by_dataset.get(q['dataset'], 0) + 1
        by_difficulty[q['difficulty']] = by_difficulty.get(q['difficulty'], 0) + 1

    print(f"\n  By type:")
    for t, c in sorted(by_type.items()):
        print(f"    {t:25s} {c:4d}")
    print(f"\n  By dataset:")
    for d, c in sorted(by_dataset.items()):
        print(f"    {d:25s} {c:4d}")
    print(f"\n  By difficulty:")
    for d, c in sorted(by_difficulty.items()):
        print(f"    {d:25s} {c:4d}")
    print(f"\n  TOTAL: {len(all_questions)} questions")
    print(f"  Saved to: {combined_path}")


if __name__ == '__main__':
    main()
