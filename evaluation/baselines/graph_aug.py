"""Graph-Augmented LLM baseline for DW-Bench.

Uses explicit graph algorithms (NOT a trained GNN) to provide
structural context to the LLM:

1. Loads the pre-computed schema graph
2. For each question, identifies relevant tables via BFS traversal
3. Computes graph-derived context: connected components, shortest paths,
   node degrees, centrality scores
4. Provides this structural context + relevant table info to the LLM

NOTE: This baseline uses classical graph algorithms (BFS, connected
components, PageRank) to extract topology — it does NOT use a learned
Graph Neural Network. A GNN-based baseline (e.g. GRetriever) would
be a natural extension for future work.
"""
import json
import time
import numpy as np
from pathlib import Path
from collections import deque


def load_graph(data_dir: Path, obfuscated: bool = False):
    """Load the schema graph and build adjacency structures."""
    import torch

    graph_file = ('obfuscated_schema_graph.pt' if obfuscated
                  else 'schema_graph.pt')
    data = torch.load(data_dir / graph_file, weights_only=False)
    table_names = data['table'].table_names
    struct_features = data['table'].x  # [N, 6] structural features

    # Build UNDIRECTED adjacency lists (for BFS traversal)
    fk_adj = {t: set() for t in table_names}
    lineage_adj = {t: set() for t in table_names}
    # Build DIRECTED edge lists (for display — preserves original direction)
    fk_directed = []       # list of (source, target) tuples
    lineage_directed = []  # list of (source, target) tuples

    if ('table', 'fk_to', 'table') in data.edge_types:
        ei = data['table', 'fk_to', 'table'].edge_index
        for j in range(ei.shape[1]):
            s, d = table_names[ei[0, j].item()], table_names[ei[1, j].item()]
            fk_adj[s].add(d)
            fk_adj[d].add(s)  # undirected for traversal
            fk_directed.append((s, d))

    if ('table', 'derived_from', 'table') in data.edge_types:
        ei = data['table', 'derived_from', 'table'].edge_index
        for j in range(ei.shape[1]):
            s, d = table_names[ei[0, j].item()], table_names[ei[1, j].item()]
            lineage_adj[s].add(d)
            lineage_adj[d].add(s)
            lineage_directed.append((s, d))

    # Feature names
    feat_names = ['out_degree', 'in_degree', 'norm_degree',
                  'lineage_degree', 'betweenness', 'pagerank']

    return {
        'table_names': table_names,
        'struct_features': struct_features,
        'feat_names': feat_names,
        'fk_adj': fk_adj,
        'lineage_adj': lineage_adj,
        'fk_directed': fk_directed,
        'lineage_directed': lineage_directed,
        'data': data,
    }


def bfs_subgraph(start_tables: list, adj: dict, max_hops: int = 3):
    """BFS from start tables, return tables within max_hops."""
    visited = set(start_tables)
    queue = deque([(t, 0) for t in start_tables])
    layers = {0: set(start_tables)}

    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
                if depth + 1 not in layers:
                    layers[depth + 1] = set()
                layers[depth + 1].add(neighbor)

    return visited, layers


def find_shortest_path_bfs(start: str, end: str, adj: dict):
    """BFS shortest path between two tables."""
    if start == end:
        return [start]
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        node, path = queue.popleft()
        for neighbor in adj.get(node, []):
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None  # No path


def find_connected_components(table_names: list, adj: dict):
    """Find connected components using BFS."""
    visited = set()
    components = []
    for t in table_names:
        if t not in visited:
            component = set()
            queue = deque([t])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                for neighbor in adj.get(node, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(sorted(component))
    return components


def build_graph_context(question: str, graph_data: dict):
    """Build graph-enhanced context for a question.
    
    Extracts table names mentioned in the question,
    then provides structural context around those tables.
    """
    table_names = graph_data['table_names']
    fk_adj = graph_data['fk_adj']
    lineage_adj = graph_data['lineage_adj']
    features = graph_data['struct_features']
    feat_names = graph_data['feat_names']

    # Find tables mentioned in question
    q_lower = question.lower()
    mentioned = [t for t in table_names if t.lower() in q_lower]

    # Combined adjacency for traversal
    combined_adj = {t: set() for t in table_names}
    for t in table_names:
        combined_adj[t] = fk_adj[t] | lineage_adj[t]

    # Get subgraph around mentioned tables (3 hops)
    if mentioned:
        relevant, layers = bfs_subgraph(mentioned, combined_adj, max_hops=3)
    else:
        relevant = set(table_names)
        layers = {0: set(table_names)}

    # Build connected components
    components = find_connected_components(table_names, combined_adj)

    # Build structural context
    lines = []
    lines.append(f"GRAPH STRUCTURE ({len(table_names)} tables, "
                 f"{len(components)} connected components)")
    lines.append("")

    # Component membership
    lines.append("Connected Components (Silos):")
    for i, comp in enumerate(components):
        lines.append(f"  Component {i+1} ({len(comp)} tables): "
                     f"{', '.join(comp)}")
    lines.append("")

    # Structural features for relevant tables
    lines.append("Table Structural Features:")
    name_to_idx = {n: i for i, n in enumerate(table_names)}
    for t in sorted(relevant):
        idx = name_to_idx[t]
        feats = features[idx].tolist()
        feat_str = ", ".join(f"{fn}={fv:.3f}" for fn, fv in
                           zip(feat_names, feats))
        lines.append(f"  {t}: {feat_str}")
    lines.append("")

    # FK relationships — use DIRECTED edges with correct direction
    fk_directed = graph_data['fk_directed']
    lines.append("FK Relationships (relevant tables):")
    for src, dst in fk_directed:
        if src in relevant or dst in relevant:
            lines.append(f"  {src} --FK--> {dst}")
    lines.append("")

    # Lineage relationships — use DIRECTED edges with correct direction
    lineage_directed = graph_data['lineage_directed']
    if lineage_directed:
        lines.append("Lineage Relationships (derived_from):")
        for src, dst in lineage_directed:
            if src in relevant or dst in relevant:
                lines.append(f"  {src} --DERIVED_FROM--> {dst}")
        lines.append("")

    # If two specific tables are mentioned, provide shortest path
    if len(mentioned) >= 2:
        lines.append("Shortest Paths Between Mentioned Tables:")
        for i in range(len(mentioned)):
            for j in range(i + 1, len(mentioned)):
                path = find_shortest_path_bfs(
                    mentioned[i], mentioned[j], combined_adj)
                if path:
                    lines.append(
                        f"  {mentioned[i]} → {mentioned[j]}: "
                        f"{' → '.join(path)} ({len(path)-1} hops)")
                else:
                    lines.append(
                        f"  {mentioned[i]} → {mentioned[j]}: no path exists")
        lines.append("")

    return "\n".join(lines)


def run_gnn_llm(dataset_dir: Path, api_key: str = "",
                api_base: str = "https://api.groq.com/openai/v1",
                model: str = "llama-3.3-70b-versatile",
                obfuscated: bool = False,
                qa_file: str = None) -> list:
    """Run GNN+LLM baseline on all Q&A pairs."""
    from baselines.flat_text import call_llm, SYSTEM_PROMPT, extract_answer

    # Load questions
    if qa_file is None:
        qa_file = ('qa_pairs_obfuscated.json' if obfuscated
                   else 'qa_pairs.json')
    with open(dataset_dir / qa_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # Load graph
    print(f"    Loading graph...")
    graph_data = load_graph(dataset_dir, obfuscated)
    print(f"    Graph: {len(graph_data['table_names'])} tables")

    is_local = '127.0.0.1' in api_base or 'localhost' in api_base

    results = []
    for i, q in enumerate(questions):
        # Build graph-enhanced context
        context = build_graph_context(q['question'], graph_data)

        user_prompt = "CONTEXT:\n" + context + "\n\nQUESTION: " + q['question']

        raw_response = call_llm(SYSTEM_PROMPT, user_prompt, api_key,
                                api_base=api_base, model=model)
        predicted = extract_answer(raw_response, q['answer_type'])
        api_failure = (not raw_response.get('_raw_text'))

        results.append({
            'id': q['id'],
            'question': q['question'],
            'gold_answer': q['answer'],
            'predicted_answer': predicted,
            'answer_type': q['answer_type'],
            'type': q['type'],
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

        # No delay for local, 4s for cloud
        if not is_local:
            time.sleep(4)

    return results
