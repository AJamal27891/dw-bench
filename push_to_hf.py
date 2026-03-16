"""Push DW-Bench data to HuggingFace Hub.

Usage:
    pip install huggingface_hub datasets
    huggingface-cli login
    python push_to_hf.py

This uploads:
  - Tier 1 schema-level QA (1,046 questions, 5 datasets)
  - Tier 2 value-level QA (433 questions, Syn-Logistics)
  - Schema graphs (.pt files)
  - Lineage map (row-level derivation data)
  - Value-level data (CSV tables)

Repo: https://huggingface.co/datasets/AJamal27891/dw-bench
"""
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
HF_REPO = "AJamal27891/dw-bench"
BASE = Path(__file__).parent
DATASETS_DIR = BASE / "datasets"
STAGING = Path("_hf_staging")

DS_ALL = ['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm', 'syn_logistics']


def stage_data():
    """Stage all data for HuggingFace upload."""
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir()

    # --- Tier 1: Schema-level QA ---
    tier1_dir = STAGING / "tier1"
    tier1_dir.mkdir()

    all_tier1 = []
    for ds in DS_ALL:
        ds_dir = DATASETS_DIR / ds
        qa_file = ds_dir / "qa_pairs.json"
        if qa_file.exists():
            questions = json.load(open(qa_file, encoding='utf-8'))
            for q in questions:
                q['dataset'] = ds
            all_tier1.extend(questions)
            print(f"  Tier 1: {ds} — {len(questions)} questions")

    # Write as JSONL
    with open(tier1_dir / "test.jsonl", 'w', encoding='utf-8') as f:
        for q in all_tier1:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    print(f"  Tier 1 total: {len(all_tier1)} questions → test.jsonl")

    # Obfuscated variants
    all_obf = []
    for ds in DS_ALL:
        ds_dir = DATASETS_DIR / ds
        qa_file = ds_dir / "qa_pairs_obfuscated.json"
        if qa_file.exists():
            questions = json.load(open(qa_file, encoding='utf-8'))
            for q in questions:
                q['dataset'] = ds
            all_obf.extend(questions)

    if all_obf:
        with open(tier1_dir / "test_obfuscated.jsonl", 'w', encoding='utf-8') as f:
            for q in all_obf:
                f.write(json.dumps(q, ensure_ascii=False) + '\n')
        print(f"  Tier 1 obfuscated: {len(all_obf)} questions → test_obfuscated.jsonl")

    # Extended variants
    all_ext = []
    for ds in DS_ALL:
        ds_dir = DATASETS_DIR / ds
        qa_file = ds_dir / "qa_pairs_extended.json"
        if qa_file.exists():
            questions = json.load(open(qa_file, encoding='utf-8'))
            for q in questions:
                q['dataset'] = ds
            all_ext.extend(questions)

    if all_ext:
        with open(tier1_dir / "test_extended.jsonl", 'w', encoding='utf-8') as f:
            for q in all_ext:
                f.write(json.dumps(q, ensure_ascii=False) + '\n')
        print(f"  Tier 1 extended: {len(all_ext)} questions → test_extended.jsonl")

    # --- Tier 2: Value-level QA ---
    tier2_dir = STAGING / "tier2"
    tier2_dir.mkdir()

    syn_dir = DATASETS_DIR / "syn_logistics"
    for v2_file in syn_dir.glob("qa_pairs_v2*.json"):
        questions = json.load(open(v2_file, encoding='utf-8'))
        out_name = v2_file.stem.replace("qa_pairs_", "") + ".jsonl"
        with open(tier2_dir / out_name, 'w', encoding='utf-8') as f:
            for q in questions:
                q['dataset'] = 'syn_logistics'
                f.write(json.dumps(q, ensure_ascii=False) + '\n')
        print(f"  Tier 2: {v2_file.name} — {len(questions)} questions → {out_name}")

    # --- Schema graphs ---
    schemas_dir = STAGING / "schemas"
    schemas_dir.mkdir()

    for ds in DS_ALL:
        ds_dir = DATASETS_DIR / ds
        for pt_file in ds_dir.glob("*.pt"):
            dest = schemas_dir / ds / pt_file.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(pt_file, dest)
            mb = pt_file.stat().st_size / 1024 / 1024
            print(f"  Schema: {ds}/{pt_file.name} ({mb:.1f} MB)")

    # --- Lineage data ---
    lineage_dir = STAGING / "lineage_data"
    lineage_dir.mkdir()

    lineage_map = syn_dir / "lineage_map.json"
    if lineage_map.exists():
        shutil.copy2(lineage_map, lineage_dir / "lineage_map.json")
        mb = lineage_map.stat().st_size / 1024 / 1024
        print(f"  Lineage: lineage_map.json ({mb:.1f} MB)")

    # --- Value data (CSVs) ---
    value_data = syn_dir / "value_data"
    if value_data.exists():
        dest = STAGING / "value_data"
        shutil.copytree(value_data, dest)
        n_files = sum(1 for _ in dest.glob("*.csv"))
        print(f"  Value data: {n_files} CSV files")

    # --- Dataset card ---
    write_dataset_card()

    return STAGING


def write_dataset_card():
    """Create HuggingFace dataset card."""
    card = """---
language:
- en
license: mit
task_categories:
- question-answering
tags:
- data-warehouse
- graph-reasoning
- schema-understanding
- benchmark
- llm-evaluation
- data-lineage
pretty_name: DW-Bench
size_categories:
- 1K<n<10K
dataset_info:
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: answer_type
    dtype: string
  - name: subtype
    dtype: string
  - name: difficulty
    dtype: string
  - name: dataset
    dtype: string
  splits:
  - name: test
    num_examples: 1046
  - name: test_obfuscated
    num_examples: 1046
  - name: tier2
    num_examples: 433
---

# DW-Bench: Graph Topology Reasoning over Data Warehouse Schemas

[![Paper](https://img.shields.io/badge/NeurIPS-2026-blue)](https://github.com/AJamal27891/dw-bench)
[![Journal](https://img.shields.io/badge/J.%20Big%20Data-Springer-orange)](https://github.com/AJamal27891/dw-bench)

## Overview

**DW-Bench** is the first benchmark for evaluating LLMs on graph topology reasoning over data warehouse schemas.

### Tier 1: Schema-Level (1,046 questions)
- 13 subtypes across 3 categories (Lineage, Route, Silo)
- 3 difficulty levels (Easy, Medium, Hard)
- 5 datasets: AdventureWorks, TPC-DS, TPC-DI, OMOP CDM, Syn-Logistics

### Tier 2: Value-Level (433 questions)
- 8 subtypes requiring interactive row-level data access
- Syn-Logistics dataset with full value-level lineage

## Data Structure

```
dw-bench/
├── tier1/
│   ├── test.jsonl              # 1,046 Tier 1 questions
│   ├── test_obfuscated.jsonl   # Obfuscated variant
│   └── test_extended.jsonl     # Hard+Medium only
├── tier2/
│   └── v2.jsonl                # 433 Tier 2 questions
├── schemas/                    # PyG HeteroData graphs
├── lineage_data/               # Row-level lineage map
└── value_data/                 # CSV tables for Tier 2
```

## Usage

```python
from datasets import load_dataset

ds = load_dataset("AJamal27891/dw-bench", data_dir="tier1")
print(ds["test"][0])
```

## Citation

```bibtex
@inproceedings{ahmed2026dwbench,
  title={DW-Bench: Benchmarking LLMs on Data Warehouse Graph Topology Reasoning},
  author={Ahmed, Ahmed G.A.H.},
  booktitle={NeurIPS Datasets and Benchmarks},
  year={2026}
}
```
"""
    with open(STAGING / "README.md", 'w', encoding='utf-8') as f:
        f.write(card)
    print("  Dataset card: README.md")


def push_to_hub():
    """Push staged data to HuggingFace."""
    api = HfApi()

    # Create repo if needed
    try:
        create_repo(HF_REPO, repo_type="dataset", exist_ok=True)
        print(f"\nRepo: https://huggingface.co/datasets/{HF_REPO}")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload
    api.upload_folder(
        folder_path=str(STAGING),
        repo_id=HF_REPO,
        repo_type="dataset",
        commit_message="Upload DW-Bench v1.0: Tier 1 (1,046 Qs) + Tier 2 (433 Qs) + schemas + lineage data",
    )
    print(f"\n✓ Pushed to https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    print("="*60)
    print("Staging DW-Bench data for HuggingFace...")
    print("="*60)

    staging = stage_data()

    print(f"\n{'='*60}")
    print(f"Staged to: {staging.resolve()}")
    print(f"{'='*60}")

    response = input("\nPush to HuggingFace? (y/n): ").strip().lower()
    if response == 'y':
        push_to_hub()
    else:
        print("Skipped push. Data staged and ready.")
        print(f"To push later: python {__file__}")
