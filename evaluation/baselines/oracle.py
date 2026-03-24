"""Oracle Baseline for DW-Bench.

Injects the gold answer evidence directly into the LLM prompt,
simulating perfect graph retrieval. This proves:
  (a) Questions are well-formed and computationally solvable
  (b) The model CAN parse and format the answer correctly
  (c) Any failures in other baselines are retrieval/reasoning failures

Expected EM: ≥95% (edge cases in membership/combined_impact formatting
may cause minor mismatches even with perfect context).
"""
import json
import time
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from flat_text import call_llm


SYSTEM_PROMPT = """You are a database architect with perfect knowledge of the schema.
You are given the EXACT output of a graph algorithm that has already
computed the answer. Your job is to read the evidence and format
the answer in the requested JSON format.

You MUST respond with valid JSON containing:
{
  "reasoning": "Brief explanation of how the evidence answers the question",
  "answer": <the answer, matching the expected type>
}

IMPORTANT: Use ONLY the provided oracle evidence. Do not guess."""


def _format_gold_evidence(qa_pair: dict) -> str:
    """Convert gold answer + metadata into oracle evidence text."""
    answer = qa_pair['answer']
    subtype = qa_pair['subtype']
    answer_type = qa_pair['answer_type']

    evidence_parts = [
        f"[ORACLE GRAPH ALGORITHM OUTPUT]",
        f"Question subtype: {subtype}",
        f"Expected answer type: {answer_type}",
    ]

    if subtype == 'direct_fk':
        evidence_parts.append(
            f"FK adjacency check result: {answer}")
    elif subtype == 'join_path':
        evidence_parts.append(
            f"Shortest FK path found: {' -> '.join(answer)}")
    elif subtype == 'hop_count':
        evidence_parts.append(
            f"Shortest path length (hops): {answer}")
    elif subtype == 'forward':
        if isinstance(answer, list):
            evidence_parts.append(
                f"Tables directly derived from the source (exact list): {json.dumps(answer)}")
        else:
            evidence_parts.append(
                f"Tables directly derived from the source: {answer}")
    elif subtype == 'reverse':
        if isinstance(answer, list):
            evidence_parts.append(
                f"Source tables that feed into the target (exact list): {json.dumps(answer)}")
        else:
            evidence_parts.append(
                f"Source tables that feed into the target: {answer}")
    elif subtype == 'transitive':
        evidence_parts.append(
            f"Full transitive lineage chain: {answer}")
    elif subtype == 'combined_impact':
        evidence_parts.append(
            f"All tables affected (lineage + FK dependents): {answer}")
    elif subtype == 'multi_source':
        evidence_parts.append(
            f"Direct source tables feeding into the target: {answer}")
    elif subtype == 'count':
        evidence_parts.append(
            f"Number of connected components: {answer}")
    elif subtype == 'membership':
        evidence_parts.append(
            f"Component membership result: {answer}")
    elif subtype == 'isolation':
        if answer in ('no', False):
            evidence_parts.append(
                "These tables are NOT connected. They belong to different disconnected silos. The answer to the question is: no")
        else:
            evidence_parts.append(
                "These tables ARE connected through FK or lineage. The answer to the question is: yes")
    elif subtype == 'connected':
        evidence_parts.append(
            f"Connectivity check result: {answer}")
    elif subtype == 'full_enumeration':
        evidence_parts.append(
            f"All tables in the component: {answer}")
    else:
        evidence_parts.append(f"Computed answer: {answer}")

    return '\n'.join(evidence_parts)


def run_oracle(dataset_dir: Path, api_key: str = "",
               api_base: str = "https://api.groq.com/openai/v1",
               model: str = "llama-3.3-70b-versatile",
               obfuscated: bool = False,
               qa_file: str = None) -> list:
    """Run oracle baseline on a dataset.

    The oracle injects gold evidence into the prompt, testing whether
    the LLM can correctly format the answer when retrieval is perfect.
    """
    dataset_dir = Path(dataset_dir)

    # Load QA pairs
    if qa_file:
        qa_path = dataset_dir / qa_file
    elif obfuscated:
        qa_path = dataset_dir / 'qa_pairs_obfuscated.json'
    else:
        qa_path = dataset_dir / 'qa_pairs.json'

    with open(qa_path, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    print(f"\n  Oracle baseline: {len(qa_pairs)} questions from {qa_path.name}")

    results = []
    for i, qa in enumerate(qa_pairs):
        oracle_evidence = _format_gold_evidence(qa)

        user_prompt = (
            f"Oracle Evidence:\n{oracle_evidence}\n\n"
            f"Question: {qa['question']}\n\n"
            f"Format your answer as JSON with 'reasoning' and 'answer' fields. "
            f"The answer should be type: {qa['answer_type']}."
        )

        response = call_llm(SYSTEM_PROMPT, user_prompt, api_key,
                            api_base=api_base, model=model)

        predicted = response.get('answer', response.get('Answer', ''))

        results.append({
            'id': qa['id'],
            'question': qa['question'],
            'gold_answer': qa['answer'],
            'predicted_answer': predicted,
            'answer_type': qa['answer_type'],
            'type': qa.get('type', qa.get('category', '')),
            'subtype': qa['subtype'],
            'difficulty': qa['difficulty'],
            'raw_response': response.get('_raw_text', ''),
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(qa_pairs):
            em = sum(1 for r in results if r['predicted_answer'] == r['gold_answer'])
            print(f"    [{i+1}/{len(qa_pairs)}] running EM: {em}/{len(results)} "
                  f"({em/len(results)*100:.0f}%)")

    return results
