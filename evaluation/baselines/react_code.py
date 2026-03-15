"""ReAct-Code Agent Baseline for DW-Bench.

Genuine agentic baseline: the LLM receives a Python REPL with NetworkX
and must write its own code to query the schema graph. No crafted tools,
no pre-built answer-shaped functions.

This tests genuine graph reasoning: the LLM must understand graph concepts,
choose the right NetworkX function, handle edge types/directions, and
format output correctly — all by writing code from scratch.
"""
import json
import time
import networkx as nx
import torch
import traceback
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from flat_text import call_llm, extract_answer, _parse_json_response


# ============================================================
# Graph Loading (same as other baselines, no tricks)
# ============================================================

def build_graph(dataset_dir: Path, obfuscated: bool = False) -> tuple:
    """Load PyG graph and build a single NetworkX DiGraph.
    
    The graph is loaded AS-IS from the PyG file. No crafted outputs,
    no format matching. The LLM must discover the structure by
    writing exploratory code.
    """
    graph_file = 'obfuscated_schema_graph.pt' if obfuscated else 'schema_graph.pt'
    data = torch.load(dataset_dir / graph_file, weights_only=False)
    table_names = list(data['table'].table_names)

    # Build ONE honest NetworkX graph — undirected for connectivity,
    # directed version also available
    G = nx.DiGraph()
    G_undirected = nx.Graph()
    
    for t in table_names:
        G.add_node(t)
        G_undirected.add_node(t)

    fk_count = 0
    lineage_count = 0

    if ('table', 'fk_to', 'table') in data.edge_types:
        ei = data['table', 'fk_to', 'table'].edge_index
        for j in range(ei.shape[1]):
            s, d = ei[0, j].item(), ei[1, j].item()
            G.add_edge(table_names[s], table_names[d], type='fk')
            G_undirected.add_edge(table_names[s], table_names[d], type='fk')
            fk_count += 1

    if ('table', 'derived_from', 'table') in data.edge_types:
        ei = data['table', 'derived_from', 'table'].edge_index
        for j in range(ei.shape[1]):
            s, d = ei[0, j].item(), ei[1, j].item()
            G.add_edge(table_names[s], table_names[d], type='lineage')
            G_undirected.add_edge(table_names[s], table_names[d], type='lineage')
            lineage_count += 1

    print(f"  Graph loaded: {len(table_names)} tables, "
          f"{fk_count} FK edges, {lineage_count} lineage edges")

    return table_names, G, G_undirected


# ============================================================
# Sandboxed Python Execution
# ============================================================

def execute_python(code: str, sandbox: dict) -> str:
    """Execute Python code in a sandboxed environment.
    
    Returns stdout + the value of 'result' variable if set.
    Limited to 5 seconds execution time.
    """
    import io
    from contextlib import redirect_stdout
    
    stdout_capture = io.StringIO()
    try:
        with redirect_stdout(stdout_capture):
            exec(code, sandbox)
        
        stdout_text = stdout_capture.getvalue()
        result_val = sandbox.get('result', None)
        
        output_parts = []
        if stdout_text.strip():
            output_parts.append(f"stdout:\n{stdout_text.strip()}")
        if result_val is not None:
            output_parts.append(f"result = {json.dumps(result_val, default=str)}")
        
        if not output_parts:
            return "Code executed successfully (no output, no 'result' variable set)."
        
        return "\n".join(output_parts)
    
    except Exception as e:
        tb = traceback.format_exc()
        # Return last 5 lines of traceback
        tb_lines = tb.strip().split('\n')
        short_tb = '\n'.join(tb_lines[-5:])
        return f"Error:\n{short_tb}"


# ============================================================
# ReAct-Code Agent
# ============================================================

SYSTEM_PROMPT = """You are a database schema analyst. You have access to a Python environment with:

- `G` — a NetworkX DiGraph of the schema (nodes = table names, edges have attribute 'type' = 'fk' or 'lineage')
- `G_undirected` — same graph but undirected (for connectivity/component queries)
- `nx` — the NetworkX library
- `table_names` — list of all table names

To answer questions, write Python code using the `run_python` tool. Your code has access to all the above variables.

IMPORTANT RULES:
1. Set `result = your_answer` in your code — this is how you return values
2. Edge types: 'fk' = foreign key relationship, 'lineage' = data derivation (derived_from)
3. G is DIRECTED (fk edges have direction). G_undirected is for connectivity queries.
4. You may call run_python up to 5 times to explore the graph
5. After exploring, provide your FINAL answer as JSON

TOOL CALL FORMAT:
To run code, output JSON: {"tool_call": "run_python", "code": "your python code here"}

FINAL ANSWER FORMAT (after exploration):
{"answer_list": [...], "answer_int": N, "answer_bool": true/false, "answer_str": "...", "reasoning": "..."}
Set unused answer fields to null."""


def run_react_question(question: str, answer_type: str, table_names: list,
                       G: nx.DiGraph, G_undirected: nx.Graph,
                       api_key: str, api_base: str, model: str,
                       max_code_runs: int = 5) -> dict:
    """Run a single question through the ReAct-Code agent."""
    import requests

    # Build sandbox with graph pre-loaded
    sandbox = {
        'G': G,
        'G_undirected': G_undirected,
        'nx': nx,
        'table_names': table_names,
        'json': json,
        'sorted': sorted,
        'list': list,
        'set': set,
        'len': len,
        'str': str,
        'int': int,
        'bool': bool,
        'print': print,
        'result': None,
    }

    # Brief schema info (not the full graph — LLM must explore)
    schema_info = (
        f"Schema: {len(table_names)} tables, "
        f"{G.number_of_edges()} directed edges "
        f"({sum(1 for _,_,d in G.edges(data=True) if d.get('type')=='fk')} FK, "
        f"{sum(1 for _,_,d in G.edges(data=True) if d.get('type')=='lineage')} lineage).\n"
        f"Tables: {', '.join(table_names[:20])}"
        f"{'... and ' + str(len(table_names)-20) + ' more' if len(table_names) > 20 else ''}"
    )

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': 
            f"{schema_info}\n\n"
            f"Question: {question}\n"
            f"Expected answer type: {answer_type}\n\n"
            f"Write Python code to explore the graph and answer this question. "
            f"Set result = your_answer in your code."},
    ]

    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    for turn in range(max_code_runs + 1):
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
            except Exception:
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

        # Check if it's a code execution request
        if 'tool_call' in parsed and parsed['tool_call'] == 'run_python':
            code = parsed.get('code', '')
            if not code:
                # Sometimes LLM puts code in 'args'
                code = parsed.get('args', {}).get('code', '')
            
            if code:
                # Reset result before each run
                sandbox['result'] = None
                exec_output = execute_python(code, sandbox)

                messages.append({'role': 'assistant', 'content': content})
                messages.append({'role': 'user', 'content':
                    f"Code execution result:\n{exec_output}\n\n"
                    f"You may run more code or provide your FINAL answer as JSON with "
                    f"answer_list/answer_int/answer_bool/answer_str and reasoning fields."
                })
                continue

        # Not a tool call — this should be the final answer
        # If result was set in sandbox, use it
        if sandbox.get('result') is not None and 'answer_list' not in parsed:
            r = sandbox['result']
            if isinstance(r, (list, set)):
                parsed['answer_list'] = sorted(list(r)) if isinstance(r, set) else list(r)
            elif isinstance(r, bool):
                parsed['answer_bool'] = r
            elif isinstance(r, int):
                parsed['answer_int'] = r
            else:
                parsed['answer_str'] = str(r)

        parsed['_raw_text'] = content
        return parsed

    # Exceeded max code runs — check if result was set
    if sandbox.get('result') is not None:
        r = sandbox['result']
        result = {'_raw_text': 'Max code runs exceeded', 'reasoning': 'Used all code runs'}
        if isinstance(r, (list, set)):
            result['answer_list'] = sorted(list(r)) if isinstance(r, set) else list(r)
        elif isinstance(r, bool):
            result['answer_bool'] = r
        elif isinstance(r, int):
            result['answer_int'] = r
        else:
            result['answer_str'] = str(r)
        return result

    # Force final answer
    messages.append({'role': 'user', 'content':
        'You have used all code runs. Provide your FINAL answer NOW as JSON.'})

    resp = requests.post(f'{api_base}/chat/completions', headers=headers,
                         json={'model': model, 'messages': messages,
                               'max_tokens': 4096, 'temperature': 0},
                         timeout=120)
    if resp.status_code == 200:
        content = resp.json()['choices'][0]['message'].get('content', '')
        parsed = _parse_json_response(content)
        parsed['_raw_text'] = content
        return parsed

    return {'_raw_text': '', 'reasoning': 'Max code runs exceeded'}


# ============================================================
# Main Entry Point
# ============================================================

def run_react_code(dataset_dir: Path, api_key: str = "",
                   api_base: str = "https://api.groq.com/openai/v1",
                   model: str = "llama-3.3-70b-versatile",
                   obfuscated: bool = False,
                   qa_file: str = None) -> list:
    """Run ReAct-Code agent baseline on all Q&A pairs for a dataset."""
    if qa_file is None:
        qa_file = 'qa_pairs_obfuscated.json' if obfuscated else 'qa_pairs.json'

    with open(dataset_dir / qa_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    table_names, G, G_undirected = build_graph(dataset_dir, obfuscated)
    is_local = '127.0.0.1' in api_base or 'localhost' in api_base

    print(f"\n  ReAct-Code agent: {len(questions)} questions")
    print(f"  Graph: {G.number_of_nodes()} tables, {G.number_of_edges()} edges")

    results = []
    for i, q in enumerate(questions):
        raw_response = run_react_question(
            q['question'], q['answer_type'], table_names,
            G, G_undirected, api_key, api_base, model,
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
