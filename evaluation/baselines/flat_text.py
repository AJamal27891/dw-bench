"""Flat Text LLM baseline for DW-Bench.

Constructs a prompt containing ALL table DDLs as text context,
then asks the LLM to answer each question.

Uses Groq API (Llama 3.3 70B) with JSON mode for structured output.
"""
import json
import time
from pathlib import Path


def build_schema_context(data_dir: Path, obfuscated: bool = False) -> str:
    """Build text context from DDL/table information.

    For original: reads column CSVs and synthesizes DDL-like descriptions.
    For obfuscated: reads obfuscation_map.json and creates mapped descriptions.
    """
    import torch

    graph_file = ('obfuscated_schema_graph.pt' if obfuscated
                  else 'schema_graph.pt')
    data = torch.load(data_dir / graph_file, weights_only=False)
    table_names = data['table'].table_names

    # Build edge descriptions
    edges_text = []
    if ('table', 'fk_to', 'table') in data.edge_types:
        ei = data['table', 'fk_to', 'table'].edge_index
        for j in range(ei.shape[1]):
            s, d = ei[0, j].item(), ei[1, j].item()
            edges_text.append(f"  {table_names[s]} --FK--> {table_names[d]}")

    if ('table', 'derived_from', 'table') in data.edge_types:
        ei = data['table', 'derived_from', 'table'].edge_index
        for j in range(ei.shape[1]):
            s, d = ei[0, j].item(), ei[1, j].item()
            edges_text.append(
                f"  {table_names[s]} --DERIVED_FROM--> {table_names[d]}")

    context = (
        f"DATABASE SCHEMA ({len(table_names)} tables)\n"
        f"Tables: {', '.join(table_names)}\n\n"
        f"Relationships ({len(edges_text)} edges):\n"
        + '\n'.join(edges_text)
    )
    return context


def call_llm(system_prompt: str, user_prompt: str, api_key: str = "",
             api_base: str = "https://api.groq.com/openai/v1",
             model: str = "llama-3.3-70b-versatile",
             max_retries: int = 10) -> dict:
    """Call any OpenAI-compatible API with JSON mode and retry logic."""
    import requests
    import re

    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    is_local = '127.0.0.1' in api_base or 'localhost' in api_base
    is_gemini = 'googleapis.com' in api_base

    # Estimate input tokens (~4 chars per token)
    input_chars = len(system_prompt) + len(user_prompt)
    input_tokens_est = input_chars // 3  # conservative

    # Set max_tokens: Gemini has 1M context, local handled by server,
    # self-hosted (Cloudflare/vLLM) usually 32K context
    if is_gemini:
        max_out = 8192
    elif is_local:
        max_out = 8192
    else:
        # Self-hosted (Cloudflare/vLLM): use conservative output limit
        # Max gold answer is ~852 tokens, 2048 gives headroom for reasoning
        max_out = 2048

    body = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'max_tokens': max_out,
        'temperature': 0,
    }
    # Only add response_format for cloud APIs that support json_object
    # (skip for local LLMs AND Gemini which don't support it)
    if not is_local and not is_gemini:
        body['response_format'] = {'type': 'json_object'}

    timeout = 900 if is_local else 300  # 15 min local, 5 min cloud

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f'{api_base}/chat/completions',
                headers=headers,
                json=body,
                timeout=timeout,
            )

            if resp.status_code == 429:
                wait = min(60 * (attempt + 1), 180)
                print(f"    Rate limited, waiting {wait}s "
                      f"(attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                print(f"    API error {resp.status_code} "
                      f"(attempt {attempt+1}): {resp.text[:80]}")
                time.sleep(10 if not is_local else 2)
                continue

            data = resp.json()
            content = data['choices'][0]['message'].get('content', '')
            if not content:
                print(f"    Empty content (attempt {attempt+1}), retrying...")
                time.sleep(2)
                continue
            parsed = _parse_json_response(content)
            parsed['_raw_text'] = content  # preserve original LLM output
            return parsed

        except Exception as e:
            print(f"    Error on attempt {attempt+1}: {e}")
            time.sleep(5 if not is_local else 1)

    print(f"    FAILED after {max_retries} retries")
    return {'_raw_text': ''}  # API failure


def _parse_json_response(content: str) -> dict:
    """Extract JSON from LLM response with multiple fallback strategies."""
    import re

    # Pre-process: Strip <think>...</think> blocks (Qwen3, Phi-4-mini)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    # Strategy 1: Direct JSON parse
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: Extract from ```json code block
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find the largest {...} in the text
    # Match nested braces up to 3 levels
    pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    matches = re.findall(pattern, content, re.DOTALL)
    for match in sorted(matches, key=len, reverse=True):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Strategy 3.5: Recover partial answer_list from truncated JSON
    # The model started a valid JSON but it was cut off by max_tokens
    list_m = re.search(r'"answer_list"\s*:\s*\[', content)
    if list_m:
        # Extract all quoted strings after "answer_list": [
        after = content[list_m.end():]
        items = re.findall(r'"([^"]+)"', after)
        # Filter out JSON keys that might be caught
        json_keys = {'answer_int', 'answer_bool', 'answer_str', 'reasoning',
                     'answer_list', 'null', 'true', 'false'}
        items = [x for x in items if x not in json_keys and len(x) > 1]
        if items:
            result = {'answer_list': items, '_recovered_from_truncation': True}
            # Also try to grab reasoning
            reason_m = re.search(r'"reasoning"\s*:\s*"([^"]+)', content)
            if reason_m:
                result['reasoning'] = reason_m.group(1)[:200]
            return result

    # Strategy 4: Build JSON from text patterns
    result = {}
    # Look for reasoning
    reason_m = re.search(
        r'(?:reasoning|reason|explanation)[:\s]*["\']?(.+?)(?:["\']?\s*[,}]|$)',
        content, re.IGNORECASE | re.DOTALL)
    if reason_m:
        result['reasoning'] = reason_m.group(1).strip()[:200]

    # Look for list answers
    list_m = re.search(r'\[([^\]]+)\]', content)
    if list_m:
        items = re.findall(r'["\']([^"\']+)["\']', list_m.group(1))
        if items:
            result['answer_list'] = items

    # Look for integer answers
    int_m = re.search(
        r'(?:answer|count|number|result)[:\s]*(\d+)', content, re.IGNORECASE)
    if int_m:
        result['answer_int'] = int(int_m.group(1))

    # Look for boolean answers
    bool_m = re.search(
        r'(?:answer|result|connected|isolated)[:\s]*(true|false|yes|no)',
        content, re.IGNORECASE)
    if bool_m:
        val = bool_m.group(1).lower()
        result['answer_bool'] = val in ('true', 'yes')

    if result:
        return result

    # Strategy 5: Return raw text as reasoning so we can still analyze
    return {'reasoning': content[:300], '_unparsed': True}


SYSTEM_PROMPT = """You are a database schema analyst. Answer questions about database schemas strictly based on the provided context.

Respond ONLY in JSON with these exact keys. ALWAYS output the answer fields FIRST, reasoning LAST:
{
  "answer_list": ["item1", "item2"] or null,
  "answer_int": 5 or null,
  "answer_bool": true or false or null,
  "answer_str": "text" or null,
  "reasoning": "Brief explanation (1-2 sentences max)"
}

Rules:
- For questions asking "which tables", return answer_list
- For questions asking "how many", return answer_int
- For questions asking "are X and Y connected", return answer_bool
- For questions about shortest path, return answer_list (ordered)
- For questions where the answer is "no path", return answer_str: "no path"
- Set unused answer fields to null
- answer_list items must use EXACT table names from the context
- Keep reasoning SHORT — the answer fields are what matter"""


def extract_answer(response: dict, answer_type: str):
    """Extract the relevant answer from the LLM JSON response."""
    if not response:
        return None

    if answer_type in ('list', 'ordered_list'):
        ans = response.get('answer_list')
        if isinstance(ans, list):
            return sorted(ans) if answer_type == 'list' else ans
        return []
    elif answer_type in ('integer', 'number'):
        ans = response.get('answer_int')
        if ans is not None:
            try:
                return int(ans)
            except (ValueError, TypeError):
                pass
        return None
    elif answer_type == 'boolean':
        ans = response.get('answer_bool')
        if isinstance(ans, bool):
            return ans
        return None
    elif answer_type == 'string':
        ans = response.get('answer_str')
        return ans if ans else None
    return None


def run_flat_text(dataset_dir: Path, api_key: str = "",
                  api_base: str = "https://api.groq.com/openai/v1",
                  model: str = "llama-3.3-70b-versatile",
                  obfuscated: bool = False,
                  qa_file: str = None) -> list:
    """Run flat text baseline on all Q&A pairs for a dataset."""
    if qa_file is None:
        qa_file = ('qa_pairs_obfuscated.json' if obfuscated
                   else 'qa_pairs.json')
    with open(dataset_dir / qa_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    context = build_schema_context(dataset_dir, obfuscated)
    is_local = '127.0.0.1' in api_base or 'localhost' in api_base

    results = []
    for i, q in enumerate(questions):
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
