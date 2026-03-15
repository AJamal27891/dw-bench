"""Tool-Use Baseline for DW-Bench.

Gives the LLM access to graph algorithm tools (shortest_path, connected_components,
bfs_forward, lineage_chain, etc.) and lets it call them to answer questions.

This tests whether the gap is retrieval vs reasoning: if the LLM can call the
right tool, the question becomes trivial. If hop_count goes from 0% to ~95%,
it proves the gap is purely retrieval.
"""
import json
import time
import networkx as nx
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from flat_text import call_llm, extract_answer, _parse_json_response


# ============================================================
# Graph Tool Definitions
# ============================================================

def build_graph(dataset_dir: Path, obfuscated: bool = False) -> tuple:
    """Load PyG graph and build NetworkX graph for tool execution."""
    graph_file = 'obfuscated_schema_graph.pt' if obfuscated else 'schema_graph.pt'
    data = torch.load(dataset_dir / graph_file, weights_only=False)
    table_names = data['table'].table_names

    # Build undirected graph for FK queries
    G_fk = nx.Graph()
    G_directed = nx.DiGraph()  # For lineage queries
    
    for t in table_names:
        G_fk.add_node(t)
        G_directed.add_node(t)

    if ('table', 'fk_to', 'table') in data.edge_types:
        ei = data['table', 'fk_to', 'table'].edge_index
        for j in range(ei.shape[1]):
            s, d = ei[0, j].item(), ei[1, j].item()
            G_fk.add_edge(table_names[s], table_names[d], type='fk')
            G_directed.add_edge(table_names[s], table_names[d], type='fk')

    if ('table', 'derived_from', 'table') in data.edge_types:
        ei = data['table', 'derived_from', 'table'].edge_index
        for j in range(ei.shape[1]):
            s, d = ei[0, j].item(), ei[1, j].item()
            G_fk.add_edge(table_names[s], table_names[d], type='lineage')
            G_directed.add_edge(table_names[s], table_names[d], type='lineage')

    return table_names, G_fk, G_directed


def execute_tool(tool_name: str, args: dict, G_fk: nx.Graph, G_dir: nx.DiGraph, table_names: list) -> str:
    """Execute a graph tool and return the result as a string."""
    try:
        if tool_name == 'shortest_path':
            src = args.get('source', '')
            dst = args.get('target', '')
            if src not in G_fk or dst not in G_fk:
                return f"Error: Table '{src}' or '{dst}' not found in graph."
            try:
                path = nx.shortest_path(G_fk, src, dst)
                return json.dumps({"path": path, "length": len(path) - 1})
            except nx.NetworkXNoPath:
                return json.dumps({"path": None, "length": -1, "message": "No path exists"})

        elif tool_name == 'connected_components':
            components = list(nx.connected_components(G_fk))
            result = {
                "count": len(components),
                "components": [sorted(list(c)) for c in components]
            }
            return json.dumps(result)

        elif tool_name == 'check_fk_adjacency':
            src = args.get('source', '')
            dst = args.get('target', '')
            if src not in G_fk or dst not in G_fk:
                return f"Error: Table not found."
            adjacent = G_fk.has_edge(src, dst)
            return json.dumps({"adjacent": adjacent})

        elif tool_name == 'get_lineage_forward':
            src = args.get('source', '')
            if src not in G_dir:
                return f"Error: Table '{src}' not found."
            # Find all tables derived FROM this source (successors in lineage)
            targets = []
            for _, dst, d in G_dir.edges(src, data=True):
                if d.get('type') == 'lineage':
                    targets.append(dst)
            return json.dumps({"source": src, "derived_tables": targets})

        elif tool_name == 'get_lineage_reverse':
            dst = args.get('target', '')
            if dst not in G_dir:
                return f"Error: Table '{dst}' not found."
            # Find all tables that this table is derived FROM (predecessors)
            sources = []
            for src, _, d in G_dir.in_edges(dst, data=True):
                if d.get('type') == 'lineage':
                    sources.append(src)
            return json.dumps({"target": dst, "source_tables": sources})

        elif tool_name == 'transitive_lineage':
            src = args.get('source', '')
            direction = args.get('direction', 'forward')
            if src not in G_dir:
                return f"Error: Table '{src}' not found."
            
            # Build lineage-only subgraph
            lineage_edges = [(u, v) for u, v, d in G_dir.edges(data=True) if d.get('type') == 'lineage']
            G_lin = nx.DiGraph(lineage_edges)
            
            if direction == 'forward':
                reachable = nx.descendants(G_lin, src) if src in G_lin else set()
            else:
                reachable = nx.ancestors(G_lin, src) if src in G_lin else set()
            return json.dumps({"source": src, "direction": direction, "reachable": sorted(list(reachable))})

        elif tool_name == 'get_component_of':
            table = args.get('table', '')
            if table not in G_fk:
                return f"Error: Table '{table}' not found."
            for comp in nx.connected_components(G_fk):
                if table in comp:
                    return json.dumps({"table": table, "component": sorted(list(comp))})
            return json.dumps({"table": table, "component": [table]})

        elif tool_name == 'get_fk_neighbors':
            table = args.get('table', '')
            depth = args.get('depth', 1)
            if table not in G_fk:
                return f"Error: Table '{table}' not found."
            # BFS to given depth
            visited = set()
            queue = [(table, 0)]
            while queue:
                node, d = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                if d < depth:
                    for neighbor in G_fk.neighbors(node):
                        if neighbor not in visited:
                            queue.append((neighbor, d + 1))
            visited.discard(table)
            return json.dumps({"table": table, "depth": depth, "neighbors": sorted(list(visited))})

        elif tool_name == 'list_tables':
            return json.dumps({"tables": table_names, "count": len(table_names)})

        else:
            return f"Error: Unknown tool '{tool_name}'"

    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


# ============================================================
# Tool-Use LLM Loop
# ============================================================

TOOL_DEFINITIONS = """You have access to the following graph analysis tools. Call them by outputting JSON with "tool_call" key:

1. shortest_path(source, target) - Find shortest path between two tables
   Call: {"tool_call": "shortest_path", "args": {"source": "table_a", "target": "table_b"}}

2. connected_components() - Get all connected components in the graph
   Call: {"tool_call": "connected_components", "args": {}}

3. check_fk_adjacency(source, target) - Check if two tables have a direct FK relationship
   Call: {"tool_call": "check_fk_adjacency", "args": {"source": "table_a", "target": "table_b"}}

4. get_lineage_forward(source) - Get tables directly derived FROM a source table
   Call: {"tool_call": "get_lineage_forward", "args": {"source": "table_a"}}

5. get_lineage_reverse(target) - Get source tables that feed INTO a target table
   Call: {"tool_call": "get_lineage_reverse", "args": {"target": "table_a"}}

6. transitive_lineage(source, direction) - Get ALL transitively reachable tables via lineage
   Call: {"tool_call": "transitive_lineage", "args": {"source": "table_a", "direction": "forward"}}

7. get_component_of(table) - Get all tables in the same connected component
   Call: {"tool_call": "get_component_of", "args": {"table": "table_a"}}

8. get_fk_neighbors(table, depth) - Get all tables within N FK hops
   Call: {"tool_call": "get_fk_neighbors", "args": {"table": "table_a", "depth": 2}}

9. list_tables() - List all tables in the schema
   Call: {"tool_call": "list_tables", "args": {}}
"""

SYSTEM_PROMPT = """You are a database schema analyst with access to graph algorithm tools.
To answer questions, you MUST call the appropriate tool(s) first, then use the results to formulate your answer.

""" + TOOL_DEFINITIONS + """

WORKFLOW:
1. Read the question
2. Decide which tool(s) to call
3. Output a JSON with "tool_call" to invoke a tool
4. After receiving tool results, output your FINAL answer as JSON:
   {"answer_list": [...], "answer_int": N, "answer_bool": true/false, "answer_str": "...", "reasoning": "..."}

Rules:
- You may call UP TO 3 tools before answering
- After tool results are provided, you MUST output your final answer
- Set unused answer fields to null
- Use EXACT table names from tool results"""


def run_tool_use_question(question: str, answer_type: str, table_names: list,
                          G_fk: nx.Graph, G_dir: nx.DiGraph,
                          api_key: str, api_base: str, model: str,
                          max_tool_calls: int = 3) -> dict:
    """Run a single question through the tool-use loop."""
    import requests

    is_gemini = 'googleapis.com' in api_base
    
    # Brief schema summary (just table list, not full DDL)
    schema_summary = f"Schema: {len(table_names)} tables: {', '.join(table_names)}"
    
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f"{schema_summary}\n\nQuestion: {question}\n\n"
         f"Expected answer type: {answer_type}\n"
         f"Call the appropriate tool(s) to answer this question."},
    ]

    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    for turn in range(max_tool_calls + 1):
        body = {
            'model': model,
            'messages': messages,
            'max_tokens': 4096,
            'temperature': 0,
        }
        
        for attempt in range(5):
            try:
                resp = requests.post(
                    f'{api_base}/chat/completions',
                    headers=headers,
                    json=body,
                    timeout=120,
                )
                if resp.status_code == 429:
                    time.sleep(30 * (attempt + 1))
                    continue
                if resp.status_code != 200:
                    time.sleep(5)
                    continue
                break
            except Exception as e:
                time.sleep(5)
                continue
        else:
            return {'_raw_text': '', 'reasoning': 'API failure'}
        
        data = resp.json()
        content = data['choices'][0]['message'].get('content', '')
        if not content:
            return {'_raw_text': '', 'reasoning': 'Empty response'}
        
        # Try to parse as JSON
        parsed = _parse_json_response(content)
        
        # Check if it's a tool call
        if 'tool_call' in parsed:
            tool_name = parsed['tool_call']
            tool_args = parsed.get('args', {})
            tool_result = execute_tool(tool_name, tool_args, G_fk, G_dir, table_names)
            
            # Add assistant message and tool result to conversation
            messages.append({'role': 'assistant', 'content': content})
            messages.append({'role': 'user', 'content': 
                f"Tool result for {tool_name}:\n{tool_result}\n\n"
                f"You may call another tool or provide your FINAL answer as JSON with "
                f"answer_list/answer_int/answer_bool/answer_str and reasoning fields."
            })
            continue
        
        # Not a tool call — this should be the final answer
        parsed['_raw_text'] = content
        return parsed
    
    # Exceeded max tool calls — force final answer
    messages.append({'role': 'user', 'content': 
        'You have used all tool calls. Provide your FINAL answer NOW as JSON.'})
    
    resp = requests.post(f'{api_base}/chat/completions', headers=headers,
                         json={'model': model, 'messages': messages, 
                               'max_tokens': 4096, 'temperature': 0},
                         timeout=120)
    if resp.status_code == 200:
        content = resp.json()['choices'][0]['message'].get('content', '')
        parsed = _parse_json_response(content)
        parsed['_raw_text'] = content
        return parsed
    
    return {'_raw_text': '', 'reasoning': 'Max tool calls exceeded'}


# ============================================================
# Main Entry Point
# ============================================================

def run_tool_use(dataset_dir: Path, api_key: str = "",
                 api_base: str = "https://api.groq.com/openai/v1",
                 model: str = "llama-3.3-70b-versatile",
                 obfuscated: bool = False,
                 qa_file: str = None) -> list:
    """Run tool-use baseline on all Q&A pairs for a dataset."""
    if qa_file is None:
        qa_file = 'qa_pairs_obfuscated.json' if obfuscated else 'qa_pairs.json'
    
    with open(dataset_dir / qa_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    table_names, G_fk, G_dir = build_graph(dataset_dir, obfuscated)
    is_local = '127.0.0.1' in api_base or 'localhost' in api_base

    print(f"\n  Tool-Use baseline: {len(questions)} questions")
    print(f"  Graph: {G_fk.number_of_nodes()} tables, {G_fk.number_of_edges()} edges")

    results = []
    for i, q in enumerate(questions):
        raw_response = run_tool_use_question(
            q['question'], q['answer_type'], table_names,
            G_fk, G_dir, api_key, api_base, model,
        )
        predicted = extract_answer(raw_response, q['answer_type'])
        api_failure = not raw_response.get('_raw_text')

        results.append({
            'id': q['id'],
            'question': q['question'],
            'gold_answer': q['answer'],
            'predicted_answer': predicted,
            'answer_type': q['answer_type'],
            'type': q.get('type', q.get('category', '')),
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
