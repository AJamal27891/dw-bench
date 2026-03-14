"""Vector-RAG baseline for DW-Bench.

Instead of providing ALL schema information, retrieves the top-K most 
relevant table descriptions using cosine similarity with the question.

Uses sentence-transformers for embedding tables/questions, then feeds
only the relevant subset to the LLM.
"""
import json
import time
import numpy as np
from pathlib import Path


def build_table_chunks(data_dir: Path, obfuscated: bool = False) -> list:
    """Build one text chunk per table describing its edges.
    
    Each chunk = table name + its FK/lineage edges.
    Returns list of dicts: {'table': name, 'text': description}
    Also returns all_lineage edges for global context.
    """
    import torch

    graph_file = ('obfuscated_schema_graph.pt' if obfuscated
                  else 'schema_graph.pt')
    data = torch.load(data_dir / graph_file, weights_only=False)
    table_names = data['table'].table_names

    # Build edge map per table
    edges_by_table = {t: [] for t in table_names}
    # Store ALL directed lineage and FK edges globally
    all_lineage = []  # (source, target) tuples
    all_fk = []       # (source, target) tuples

    if ('table', 'fk_to', 'table') in data.edge_types:
        ei = data['table', 'fk_to', 'table'].edge_index
        for j in range(ei.shape[1]):
            s, d = ei[0, j].item(), ei[1, j].item()
            src, dst = table_names[s], table_names[d]
            edges_by_table[src].append(f"  {src} --FK--> {dst}")
            edges_by_table[dst].append(f"  {src} --FK--> {dst}")
            all_fk.append((src, dst))

    if ('table', 'derived_from', 'table') in data.edge_types:
        ei = data['table', 'derived_from', 'table'].edge_index
        for j in range(ei.shape[1]):
            s, d = ei[0, j].item(), ei[1, j].item()
            src, dst = table_names[s], table_names[d]
            edges_by_table[src].append(
                f"  {src} --DERIVED_FROM--> {dst}")
            edges_by_table[dst].append(
                f"  {src} --DERIVED_FROM--> {dst}")
            all_lineage.append((src, dst))

    chunks = []
    for t in table_names:
        edges = list(set(edges_by_table[t]))  # deduplicate
        text = f"Table: {t}\n"
        if edges:
            text += "Relationships:\n" + "\n".join(edges)
        else:
            text += "No relationships (isolated table)"
        chunks.append({'table': t, 'text': text})

    return chunks, all_lineage, all_fk


def embed_texts(texts: list, model_name: str = 'all-MiniLM-L6-v2'):
    """Embed a list of texts using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False,
                              normalize_embeddings=True)
    return embeddings


def retrieve_top_k(query_emb, chunk_embs, chunks, k=10):
    """Retrieve top-K most similar chunks to the query."""
    # Cosine similarity (embeddings are already normalized)
    sims = query_emb @ chunk_embs.T
    top_indices = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_indices]


def run_vector_rag(dataset_dir: Path, api_key: str = "",
                   api_base: str = "https://api.groq.com/openai/v1",
                   model: str = "llama-3.3-70b-versatile",
                   obfuscated: bool = False, top_k: int = 15,
                   qa_file: str = None) -> list:
    """Run Vector-RAG baseline on all Q&A pairs."""
    from baselines.flat_text import call_llm, _parse_json_response
    from baselines.flat_text import SYSTEM_PROMPT, extract_answer

    # Load questions
    if qa_file is None:
        qa_file = ('qa_pairs_obfuscated.json' if obfuscated
                   else 'qa_pairs.json')
    with open(dataset_dir / qa_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # Build and embed table chunks (now also returns global edges)
    chunks, all_lineage, all_fk = build_table_chunks(dataset_dir, obfuscated)
    chunk_texts = [c['text'] for c in chunks]
    print(f"    Embedding {len(chunks)} table chunks...")
    chunk_embs = embed_texts(chunk_texts)

    # Build global lineage summary (appended to every context)
    lineage_summary = ""
    if all_lineage:
        lineage_summary = "\n\nALL LINEAGE RELATIONSHIPS (global):\n"
        for src, dst in all_lineage:
            lineage_summary += f"  {src} --DERIVED_FROM--> {dst}\n"

    # Embed all questions at once for efficiency
    q_texts = [q['question'] for q in questions]
    print(f"    Embedding {len(q_texts)} questions...")
    q_embs = embed_texts(q_texts)

    is_local = '127.0.0.1' in api_base or 'localhost' in api_base

    results = []
    for i, q in enumerate(questions):
        # Retrieve top-K relevant table chunks
        retrieved = retrieve_top_k(q_embs[i], chunk_embs, chunks, k=top_k)
        
        # Build context from retrieved chunks + global lineage
        context = (
            f"DATABASE SCHEMA (showing {len(retrieved)} most relevant "
            f"tables out of {len(chunks)} total)\n\n"
            + "\n\n".join(c['text'] for c in retrieved)
            + lineage_summary
        )

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
            'retrieved_tables': [c['table'] for c in retrieved],
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
