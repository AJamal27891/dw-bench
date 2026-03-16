"""Tool-Use V2 Baseline for DW-Bench Tier 2.

Extends the original tool_use.py with 3 value-query tools:
  - query_table: filter rows from value_data CSVs
  - trace_row_lineage: follow lineage_map row-to-row
  - count_downstream: count cascade-affected rows

Does NOT modify tool_use.py — this is a standalone Tier 2 baseline.

Usage:
    Called by evaluate.py with --baseline tool_use_v2 --tier 2
"""
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from flat_text import call_llm, extract_answer, _parse_json_response


# ============================================================
# Value Data & Lineage Loading
# ============================================================

def load_value_data(dataset_dir: Path):
    """Load value CSV data and lineage map."""
    value_dir = dataset_dir / 'value_data'
    lm_path = dataset_dir / 'lineage_map.json'

    # Load manifest
    with open(value_dir / '_manifest.json', 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # Load lineage map
    with open(lm_path, 'r', encoding='utf-8') as f:
        lineage_map = json.load(f)

    # Build reverse lineage index
    reverse_lineage = defaultdict(list)
    for target_table, rows_map in lineage_map.items():
        for row_key, row_info in rows_map.items():
            for src in row_info['sources']:
                for src_row_idx in src['rows']:
                    src_id = f"{src['table']}:row_{src_row_idx}"
                    reverse_lineage[src_id].append(
                        (target_table, row_key))

    return manifest, lineage_map, reverse_lineage, value_dir


# ============================================================
# Value Tool Execution
# ============================================================

def execute_value_tool(tool_name: str, args: dict,
                       manifest: dict, lineage_map: dict,
                       reverse_lineage: dict,
                       value_dir: Path, table_names: list) -> str:
    """Execute a value-query tool."""
    try:
        if tool_name == 'query_table':
            table = args.get('table', '')
            column = args.get('column', '')
            value = args.get('value', '')
            limit = args.get('limit', 20)

            if table not in manifest:
                return json.dumps({'error': f"Table '{table}' not found"})

            csv_path = value_dir / f'{table}.csv'
            matches = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                row_idx = 0
                for row in reader:
                    if column in row and str(row[column]) == str(value):
                        matches.append({
                            'row_id': f'row_{row_idx}',
                            **{k: v for k, v in row.items()},
                        })
                        if len(matches) >= limit:
                            break
                    row_idx += 1

            return json.dumps({
                'table': table, 'filter': f'{column}={value}',
                'matches': len(matches),
                'rows': matches[:limit],
            })

        elif tool_name == 'trace_row_lineage':
            table = args.get('table', '')
            row_id = args.get('row_id', '')
            direction = args.get('direction', 'backward')

            if direction == 'backward':
                if table not in lineage_map:
                    return json.dumps({
                        'table': table, 'row_id': row_id,
                        'direction': 'backward', 'sources': [],
                    })
                row_info = lineage_map[table].get(row_id, {})
                sources = row_info.get('sources', [])
                return json.dumps({
                    'table': table, 'row_id': row_id,
                    'direction': 'backward', 'sources': sources,
                })
            else:  # forward
                src_id = f'{table}:{row_id}'
                targets = []
                for tgt_table, tgt_row in reverse_lineage.get(src_id, []):
                    targets.append({
                        'table': tgt_table, 'row_id': tgt_row,
                    })
                return json.dumps({
                    'table': table, 'row_id': row_id,
                    'direction': 'forward',
                    'downstream_rows': targets[:50],
                    'total': len(targets),
                })

        elif tool_name == 'count_downstream':
            table = args.get('table', '')
            row_id = args.get('row_id', '')

            # BFS forward through lineage
            all_targets = defaultdict(set)
            queue = [f'{table}:{row_id}']
            visited = set()
            while queue:
                src_id = queue.pop()
                if src_id in visited:
                    continue
                visited.add(src_id)
                for tgt_table, tgt_row in reverse_lineage.get(src_id, []):
                    all_targets[tgt_table].add(tgt_row)
                    queue.append(f'{tgt_table}:{tgt_row}')

            by_table = {t: len(rows) for t, rows in
                        sorted(all_targets.items())}
            total = sum(by_table.values())
            return json.dumps({
                'source': f'{table}:{row_id}',
                'total_downstream': total,
                'by_table': by_table,
            })

        elif tool_name == 'list_tables':
            return json.dumps({
                'tables': table_names, 'count': len(table_names),
            })

        elif tool_name == 'get_table_info':
            table = args.get('table', '')
            if table not in manifest:
                return json.dumps({'error': f"Table '{table}' not found"})
            info = manifest[table]
            # Read first 3 rows as sample
            csv_path = value_dir / f'{table}.csv'
            sample = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= 3:
                        break
                    sample.append(dict(row))
            return json.dumps({
                'table': table,
                'rows': info['rows'],
                'columns': info['columns'],
                'layer': info['layer'],
                'silo': info['silo'],
                'sample': sample,
            })

        else:
            return json.dumps({'error': f"Unknown tool '{tool_name}'"})

    except Exception as e:
        return json.dumps({'error': f'{tool_name}: {str(e)}'})


# ============================================================
# Tool Definitions for LLM
# ============================================================

V2_TOOL_DEFINITIONS = """You have access to the following tools for querying data warehouse value data and lineage:

1. query_table(table, column, value, limit) - Filter rows from a table
   Call: {"tool_call": "query_table", "args": {"table": "raw_purchase_orders", "column": "supplier_name", "value": "Acme Corp", "limit": 10}}

2. trace_row_lineage(table, row_id, direction) - Trace row-level lineage
   direction="backward": find source rows that feed into this row
   direction="forward": find downstream rows derived from this row
   Call: {"tool_call": "trace_row_lineage", "args": {"table": "dim_supplier", "row_id": "row_5", "direction": "backward"}}

3. count_downstream(table, row_id) - Count ALL transitively affected downstream rows
   Call: {"tool_call": "count_downstream", "args": {"table": "raw_purchase_orders", "row_id": "row_42"}}

4. list_tables() - List all tables
   Call: {"tool_call": "list_tables", "args": {}}

5. get_table_info(table) - Get table metadata and sample rows
   Call: {"tool_call": "get_table_info", "args": {"table": "dim_supplier"}}
"""

V2_SYSTEM_PROMPT = """You are a data warehouse lineage analyst with access to data query tools.
To answer questions about data provenance, forward impact, and row-level lineage, you MUST call the appropriate tool(s).

""" + V2_TOOL_DEFINITIONS + """

WORKFLOW:
1. Read the question carefully
2. Decide which tool(s) to call
3. Output JSON with "tool_call" to invoke a tool
4. After receiving tool results, output your FINAL answer as JSON:
   {"answer_list": [...], "answer_int": N, "answer_bool": true/false, "answer_str": "...", "reasoning": "..."}

Rules:
- You may call UP TO 5 tools before answering
- After tool results, output your final answer
- Set unused answer fields to null
- Use EXACT table and row names from tool results"""


# ============================================================
# Main Runner
# ============================================================

def run_tool_use_v2_question(question: str, answer_type: str,
                              table_names: list,
                              manifest: dict, lineage_map: dict,
                              reverse_lineage: dict, value_dir: Path,
                              api_key: str, api_base: str, model: str,
                              max_tool_calls: int = 5) -> dict:
    """Run a single Tier 2 question through the tool-use loop."""
    import requests

    schema_summary = (
        f"Schema: {len(table_names)} tables. "
        f"Layers: raw → staging → core (dim/fact) → mart. "
        f"Tables: {', '.join(table_names[:20])}..."
    )

    messages = [
        {'role': 'system', 'content': V2_SYSTEM_PROMPT},
        {'role': 'user', 'content':
            f"{schema_summary}\n\nQuestion: {question}\n\n"
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

        if 'tool_call' in parsed:
            tool_name = parsed['tool_call']
            tool_args = parsed.get('args', {})
            tool_result = execute_value_tool(
                tool_name, tool_args, manifest,
                lineage_map, reverse_lineage, value_dir, table_names)

            messages.append({'role': 'assistant', 'content': content})
            messages.append({'role': 'user', 'content':
                f"Tool result for {tool_name}:\n{tool_result}\n\n"
                f"You may call another tool or provide your FINAL answer."})
            continue

        parsed['_raw_text'] = content
        return parsed

    # Force final answer
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

    return {'_raw_text': '', 'reasoning': 'Max tool calls exceeded'}


def run_tool_use_v2(dataset_dir: Path, api_key: str = "",
                     api_base: str = "https://api.groq.com/openai/v1",
                     model: str = "llama-3.3-70b-versatile",
                     qa_file: str = None) -> list:
    """Run Tier 2 tool-use baseline on all QA pairs."""
    if qa_file is None:
        qa_file = 'qa_pairs_tier2.json'

    with open(dataset_dir / qa_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # Load schema graph for table names
    import torch
    data = torch.load(dataset_dir / 'schema_graph.pt', weights_only=False)
    table_names = data['table'].table_names

    # Load value data
    manifest, lineage_map, reverse_lineage, value_dir = \
        load_value_data(dataset_dir)
    is_local = '127.0.0.1' in api_base or 'localhost' in api_base

    print(f"\n  Tool-Use V2 baseline: {len(questions)} Tier 2 questions")
    print(f"  Value data: {sum(m['rows'] for m in manifest.values()):,} rows "
          f"across {len(manifest)} tables")

    results = []
    for i, q in enumerate(questions):
        raw_response = run_tool_use_v2_question(
            q['question'], q['answer_type'], table_names,
            manifest, lineage_map, reverse_lineage, value_dir,
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

        if (i + 1) % 10 == 0:
            failures = sum(1 for r in results if r['api_failure'])
            print(f"    Processed {i+1}/{len(questions)} "
                  f"(API failures: {failures})")

        if not is_local:
            time.sleep(4)

    return results
