"""ReAct-Code V2 Baseline for DW-Bench Tier 2.

Extends react_code.py with value-level data access in the sandbox:
  - value_data dict: table → list of row dicts
  - lineage_map: row-level provenance
  - reverse_lineage: forward impact lookup
  - Helper functions for querying

The LLM writes Python code to query both the schema graph AND row-level
data. No pre-built answer functions — the LLM must compose its own logic.
"""
import csv
import json
import time
import networkx as nx
import torch
import traceback
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from flat_text import call_llm, extract_answer, _parse_json_response


# ============================================================
# Data Loading
# ============================================================

def build_graph(dataset_dir: Path) -> tuple:
    """Load PyG graph and build NetworkX DiGraph."""
    data = torch.load(dataset_dir / 'schema_graph.pt', weights_only=False)
    table_names = list(data['table'].table_names)

    G = nx.DiGraph()
    G_undirected = nx.Graph()
    for t in table_names:
        G.add_node(t)
        G_undirected.add_node(t)

    if ('table', 'fk_to', 'table') in data.edge_types:
        ei = data['table', 'fk_to', 'table'].edge_index
        for j in range(ei.shape[1]):
            s, d = ei[0, j].item(), ei[1, j].item()
            G.add_edge(table_names[s], table_names[d], type='fk')
            G_undirected.add_edge(table_names[s], table_names[d], type='fk')

    if ('table', 'derived_from', 'table') in data.edge_types:
        ei = data['table', 'derived_from', 'table'].edge_index
        for j in range(ei.shape[1]):
            s, d = ei[0, j].item(), ei[1, j].item()
            G.add_edge(table_names[s], table_names[d], type='lineage')
            G_undirected.add_edge(table_names[s], table_names[d], type='lineage')

    return table_names, G, G_undirected


def load_value_data(dataset_dir: Path):
    """Load value CSVs and lineage map into memory."""
    value_dir = dataset_dir / 'value_data'
    lm_path = dataset_dir / 'lineage_map.json'

    with open(value_dir / '_manifest.json', 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    with open(lm_path, 'r', encoding='utf-8') as f:
        lineage_map = json.load(f)

    # Load all CSV data into memory (for sandbox access)
    table_data = {}
    for table_name in manifest:
        csv_path = value_dir / f'{table_name}.csv'
        if csv_path.exists():
            with open(csv_path, 'r', encoding='utf-8') as f:
                table_data[table_name] = list(csv.DictReader(f))

    # Build reverse lineage
    reverse_lineage = defaultdict(list)
    for target_table, rows_map in lineage_map.items():
        for row_key, row_info in rows_map.items():
            for src in row_info['sources']:
                for src_row_idx in src['rows']:
                    src_id = f"{src['table']}:row_{src_row_idx}"
                    reverse_lineage[src_id].append(
                        (target_table, row_key))

    return manifest, lineage_map, dict(reverse_lineage), table_data


# ============================================================
# Sandboxed Python Execution
# ============================================================

def execute_python(code: str, sandbox: dict) -> str:
    """Execute Python code in a sandboxed environment."""
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
            output_parts.append(
                f"result = {json.dumps(result_val, default=str)}")

        if not output_parts:
            return ("Code executed successfully "
                    "(no output, no 'result' variable set).")
        return "\n".join(output_parts)

    except Exception:
        tb = traceback.format_exc()
        tb_lines = tb.strip().split('\n')
        short_tb = '\n'.join(tb_lines[-5:])
        return f"Error:\n{short_tb}"


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT_V2 = """You are a data warehouse lineage analyst. You have access to a Python environment with:

**Schema Graph:**
- `G` — NetworkX DiGraph (nodes = table names, edges have 'type' = 'fk' or 'lineage')
- `G_undirected` — same graph but undirected
- `nx` — the NetworkX library
- `table_names` — list of all table names

**Value-Level Data:**
- `table_data` — dict mapping table_name → list of row dicts (each row is a dict of column→value)
  Example: table_data['raw_purchase_orders'][0] = {'supplier_name': 'Acme', 'amount': '100', ...}

- `lineage_map` — dict mapping target_table → {row_id: {sources: [{table, rows: [int]}]}}
  Tells you which source rows feed into each target row.
  Example: lineage_map['dim_supplier']['row_5']['sources'] = [{'table': 'raw_supplier_catalog', 'rows': [3, 7]}]

- `reverse_lineage` — dict mapping "table:row_id" → [(target_table, target_row_id), ...]
  Tells you where a source row flows downstream.
  Example: reverse_lineage['raw_purchase_orders:row_42'] = [('stg_orders', 'row_10'), ...]

- `manifest` — dict mapping table_name → {rows, columns, layer, silo}

**Helper functions available in sandbox:**
- `query_table(table, column, value)` — returns list of (row_idx, row_dict) matching column==value
- `get_row(table, row_idx)` — returns the row dict at index row_idx
- `trace_backward(table, row_id)` — returns sources list from lineage_map
- `trace_forward(table, row_id)` — returns downstream targets from reverse_lineage
- `cascade_count(table, row_id)` — BFS forward, returns {table: count} of affected rows

IMPORTANT RULES:
1. Set `result = your_answer` in your code — this is how you return values
2. You may call run_python up to 5 times to explore
3. After exploring, provide your FINAL answer as JSON

TOOL CALL FORMAT:
To run code: {"tool_call": "run_python", "code": "your python code here"}

FINAL ANSWER FORMAT:
{"answer_list": [...], "answer_int": N, "answer_bool": true/false, "answer_str": "...", "reasoning": "..."}
Set unused answer fields to null."""


# ============================================================
# Helper Functions for Sandbox
# ============================================================

def make_sandbox_helpers(table_data, lineage_map, reverse_lineage):
    """Create helper functions that the sandbox can use."""

    def query_table(table, column, value):
        """Find rows where column == value. Returns [(row_idx, row)]."""
        rows = table_data.get(table, [])
        matches = []
        for i, row in enumerate(rows):
            if column in row and str(row[column]) == str(value):
                matches.append((i, dict(row)))
        return matches

    def get_row(table, row_idx):
        """Get a specific row by index."""
        rows = table_data.get(table, [])
        if 0 <= row_idx < len(rows):
            return dict(rows[row_idx])
        return None

    def trace_backward(table, row_id):
        """Trace backward lineage for a row."""
        if table not in lineage_map:
            return []
        return lineage_map[table].get(row_id, {}).get('sources', [])

    def trace_forward(table, row_id):
        """Trace forward impact for a row."""
        src_id = f'{table}:{row_id}'
        return reverse_lineage.get(src_id, [])

    def cascade_count(table, row_id):
        """BFS forward through lineage, return {table: count}."""
        all_targets = defaultdict(set)
        queue = [f'{table}:{row_id}']
        visited = set()
        while queue:
            src_id = queue.pop(0)
            if src_id in visited:
                continue
            visited.add(src_id)
            for tgt_table, tgt_row in reverse_lineage.get(src_id, []):
                all_targets[tgt_table].add(tgt_row)
                queue.append(f'{tgt_table}:{tgt_row}')
        return {t: len(rows) for t, rows in sorted(all_targets.items())}

    return {
        'query_table': query_table,
        'get_row': get_row,
        'trace_backward': trace_backward,
        'trace_forward': trace_forward,
        'cascade_count': cascade_count,
    }


# ============================================================
# ReAct-Code V2 Agent
# ============================================================

def run_react_v2_question(question: str, answer_type: str,
                          table_names: list,
                          G: nx.DiGraph, G_undirected: nx.Graph,
                          table_data: dict, lineage_map: dict,
                          reverse_lineage: dict, manifest: dict,
                          api_key: str, api_base: str, model: str,
                          max_code_runs: int = 5) -> dict:
    """Run a single Tier 2 question through ReAct-Code V2."""
    import requests

    helpers = make_sandbox_helpers(table_data, lineage_map, reverse_lineage)

    sandbox = {
        # Graph
        'G': G, 'G_undirected': G_undirected,
        'nx': nx, 'table_names': table_names,
        # Value data
        'table_data': table_data,
        'lineage_map': lineage_map,
        'reverse_lineage': reverse_lineage,
        'manifest': manifest,
        # Helpers
        **helpers,
        # Python builtins
        'json': json, 'sorted': sorted, 'list': list, 'set': set,
        'len': len, 'str': str, 'int': int, 'bool': bool,
        'dict': dict, 'print': print, 'sum': sum, 'min': min,
        'max': max, 'any': any, 'all': all, 'enumerate': enumerate,
        'range': range, 'zip': zip, 'map': map, 'filter': filter,
        'isinstance': isinstance, 'type': type, 'tuple': tuple,
        'defaultdict': defaultdict,
        'result': None,
    }

    schema_info = (
        f"Schema: {len(table_names)} tables, "
        f"{G.number_of_edges()} edges. "
        f"Value data: {sum(len(v) for v in table_data.values())} rows "
        f"across {len(table_data)} tables.\n"
        f"Tables: {', '.join(table_names[:20])}"
        f"{'... +' + str(len(table_names)-20) + ' more' if len(table_names) > 20 else ''}"
    )

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT_V2},
        {'role': 'user', 'content':
            f"{schema_info}\n\n"
            f"Question: {question}\n"
            f"Expected answer type: {answer_type}\n\n"
            f"Write Python code to explore the data and answer."},
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
                    headers=headers, json=body, timeout=120)
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

        parsed = _parse_json_response(content)

        # Check for code execution
        if 'tool_call' in parsed and parsed['tool_call'] == 'run_python':
            code = parsed.get('code', '')
            if not code:
                code = parsed.get('args', {}).get('code', '')

            if code:
                sandbox['result'] = None
                exec_output = execute_python(code, sandbox)

                messages.append({'role': 'assistant', 'content': content})
                messages.append({'role': 'user', 'content':
                    f"Code execution result:\n{exec_output}\n\n"
                    f"You may run more code or provide your FINAL answer."})
                continue

        # Final answer — check sandbox result
        if sandbox.get('result') is not None and 'answer_list' not in parsed:
            r = sandbox['result']
            if isinstance(r, (list, set)):
                parsed['answer_list'] = (sorted(list(r))
                                          if isinstance(r, set) else list(r))
            elif isinstance(r, bool):
                parsed['answer_bool'] = r
            elif isinstance(r, int):
                parsed['answer_int'] = r
            else:
                parsed['answer_str'] = str(r)

        parsed['_raw_text'] = content
        return parsed

    # Exceeded max — check sandbox
    if sandbox.get('result') is not None:
        r = sandbox['result']
        result = {'_raw_text': 'Max code runs exceeded',
                  'reasoning': 'Used all code runs'}
        if isinstance(r, (list, set)):
            result['answer_list'] = (sorted(list(r))
                                      if isinstance(r, set) else list(r))
        elif isinstance(r, bool):
            result['answer_bool'] = r
        elif isinstance(r, int):
            result['answer_int'] = r
        else:
            result['answer_str'] = str(r)
        return result

    # Force final
    messages.append({'role': 'user', 'content':
        'Provide your FINAL answer NOW as JSON.'})
    try:
        resp = requests.post(f'{api_base}/chat/completions', headers=headers,
                             json={'model': model, 'messages': messages,
                                   'max_tokens': 4096, 'temperature': 0},
                             timeout=120)
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message'].get('content', '')
            parsed = _parse_json_response(content)
            parsed['_raw_text'] = content
            return parsed
    except Exception:
        pass
    return {'_raw_text': '', 'reasoning': 'Max code runs exceeded'}


# ============================================================
# Main Entry Point (with checkpoint save/resume)
# ============================================================

CHECKPOINT_EVERY = 25  # Save partial results every N questions


def _checkpoint_path(results_dir: Path, dataset: str, model_tag: str) -> Path:
    return results_dir / f'react_code_v2_original_{dataset}_{model_tag}.ckpt.json'


def run_react_code_v2(dataset_dir: Path, api_key: str = "",
                       api_base: str = "https://api.groq.com/openai/v1",
                       model: str = "llama-3.3-70b-versatile",
                       qa_file: str = None) -> list:
    """Run ReAct-Code V2 on all Tier 2 QA pairs with checkpoint save/resume."""
    if qa_file is None:
        qa_file = 'qa_pairs_tier2.json'

    with open(dataset_dir / qa_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    table_names, G, G_undirected = build_graph(dataset_dir)
    manifest, lineage_map, reverse_lineage, table_data = \
        load_value_data(dataset_dir)
    is_local = '127.0.0.1' in api_base or 'localhost' in api_base

    # Derive model_tag and dataset name for checkpoint path
    dataset_name = dataset_dir.name
    model_tag = model.split('/')[-1].replace(' ', '_')
    results_dir = dataset_dir.parent.parent / 'evaluation' / 'results'
    results_dir.mkdir(exist_ok=True)
    ckpt_path = _checkpoint_path(results_dir, dataset_name, model_tag)

    # --- Resume from checkpoint if it exists ---
    results = []
    start_idx = 0
    if ckpt_path.exists():
        try:
            with open(ckpt_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            start_idx = len(results)
            print(f"  Resuming from checkpoint: {start_idx}/{len(questions)} "
                  f"already done ({ckpt_path.name})")
        except Exception as e:
            print(f"  Warning: could not load checkpoint ({e}), starting fresh")
            results = []
            start_idx = 0

    print(f"\n  ReAct-Code V2 baseline: {len(questions)} Tier 2 questions")
    print(f"  Graph: {G.number_of_nodes()} tables, {G.number_of_edges()} edges")
    print(f"  Value data: {sum(len(v) for v in table_data.values()):,} rows "
          f"across {len(table_data)} tables")
    if start_idx > 0:
        print(f"  Starting at question {start_idx + 1} (resumed)")

    for i, q in enumerate(questions[start_idx:], start=start_idx):
        raw_response = run_react_v2_question(
            q['question'], q['answer_type'], table_names,
            G, G_undirected, table_data, lineage_map,
            reverse_lineage, manifest,
            api_key, api_base, model)

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

        # Print progress every 10 questions
        if (i + 1) % 10 == 0:
            failures = sum(1 for r in results if r['api_failure'])
            print(f"    Processed {i+1}/{len(questions)} "
                  f"(API failures: {failures})", flush=True)

        # Save checkpoint every CHECKPOINT_EVERY questions
        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(ckpt_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, default=str)
            print(f"    [CHECKPOINT] Saved {i+1} results to {ckpt_path.name}",
                  flush=True)

        if not is_local:
            time.sleep(2)  # Reduced from 4s — DeepInfra handles rate limits better

    # Clean up checkpoint on clean completion
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"  Checkpoint deleted (run complete).")

    return results
