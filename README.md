# DW-Bench: Benchmarking LLMs on Data Warehouse Graph Topology Reasoning

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026-blue)](paper/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)

**DW-Bench** is the first benchmark for evaluating whether Large Language Models can reason about the *graph structure* of data warehouse schemas — data lineage, foreign key paths, and connectivity — rather than just generating SQL queries.

## Key Findings (Gemini 2.5 Flash)

| Baseline | Easy (226) | Medium (383) | Hard (61) | **Avg** |
|---|:---:|:---:|:---:|:---:|
| **Flat Text** | 83.2% | **77.8%** | 14.8% | **74.0%** |
| **Vector RAG** | **81.9%** | 71.3% | 5.3% | 68.2% |
| **Graph-Aug** | 78.4% | 74.2% | **17.5%** | 69.9% |

> **All baselines collapse on hard structural tasks** (≤17.5% EM).
> `combined_impact` — requiring simultaneous lineage + FK traversal — remains
> unsolved at ≤6.5% EM, establishing a clear frontier for GNN-augmented approaches.

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torch_geometric openai requests sentence-transformers faiss-cpu networkx
```

### 2. Run Evaluation

```bash
# Using a cloud API (Gemini)
python run.py \
  --api-base https://generativelanguage.googleapis.com/v1beta/openai \
  --model gemini-2.5-flash \
  --api-key YOUR_API_KEY

# Using a local model (LM Studio, Ollama, vLLM)
python run.py \
  --api-base http://localhost:1234/v1 \
  --model microsoft/phi-4-mini-reasoning

# Run specific conditions
python run.py --api-base ... --model ... --condition obfuscated
python run.py --api-base ... --model ... --condition extended
python run.py --api-base ... --model ... --condition all

# Run a single baseline on a single dataset
python run.py --api-base ... --model ... --baseline flat_text --dataset tpc-ds
```

### 3. View Results

```bash
python view_results.py              # Summary tables
python integrity_check.py           # Validate result files
```

## Project Structure

```
dw-bench/
├── run.py                          # 🏃 Unified evaluation runner
├── view_results.py                 # 📊 Results summary viewer
├── integrity_check.py              # ✅ Result file validator
│
├── datasets/                       # 📁 Schema graphs + QA pairs
│   ├── adventureworks/             #    102 tables, 136 FK, 39 lineage
│   ├── tpc-ds/                     #     24 tables, 70 FK
│   ├── tpc-di/                     #     35 tables, 29 FK, 21 lineage
│   └── omop_cdm/                   #     37 tables, 74 FK, 21 lineage
│
├── evaluation/                     # 🔬 Evaluation pipeline
│   ├── evaluate.py                 #    Main evaluation orchestrator
│   ├── metrics.py                  #    Scoring (EM, F1, path validation)
│   ├── baselines/                  #    Baseline implementations
│   │   ├── flat_text.py            #      Full schema as text
│   │   ├── vector_rag.py           #      FAISS embedding retrieval
│   │   ├── graph_aug.py            #      NetworkX graph algorithms
│   │   └── gnn_llm.py             #      GNN encoder (future work)
│   └── results/                    #    Evaluation outputs (JSON)
│
├── scripts/                        # 🛠️ Data pipeline scripts
│   ├── generate_qa.py              #    Generate QA pairs from graph
│   ├── generate_extended_qa.py     #    Generate extended (hard) QA
│   ├── obfuscate_schema.py         #    Obfuscation protocol
│   ├── build_lineage.py            #    Build lineage edges
│   ├── compute_features.py         #    Node feature computation
│   ├── validate_qa.py              #    QA validation
│   └── validate_extended_qa.py     #    Extended QA validation
│
├── paper/                          # 📄 LaTeX paper source
│   ├── main.tex                    #    Modular driver
│   ├── sections/                   #    Individual sections
│   ├── neurips_2026.sty            #    NeurIPS style file
│   └── references.bib              #    Bibliography
│
└── docs/                           # 📚 Documentation
    └── data_pipeline.md            #    Data pipeline guide
```

## Benchmark Design

### Datasets (198 tables total)

| Dataset | Domain | Tables | FK Edges | Lineage | Questions | Components |
|---|---|:---:|:---:|:---:|:---:|:---:|
| AdventureWorks | Retail/HR | 102 | 136 | 39 | 204 | 11 |
| TPC-DS | Analytics | 24 | 70 | 0 | 127 | 1 |
| TPC-DI | ETL | 35 | 29 | 21 | 181 | 2 |
| OMOP CDM | Healthcare | 37 | 74 | 21 | 158 | 3 |
| **Total** | | **198** | **309** | **81** | **670** | |

### Question Types (13 subtypes, 3 difficulty levels)

| Category | Subtype | Difficulty | Description |
|---|---|---|---|
| **Lineage** | `forward` | Easy | Direct lineage targets |
| | `reverse` | Easy | Source tables for a DW table |
| | `transitive` | Hard | Multi-hop lineage chains |
| | `combined_impact` | Hard | Lineage + FK dependents |
| | `multi_source` | Medium | Tables with 3+ sources |
| **Route** | `direct_fk` | Easy | FK adjacency check |
| | `join_path` | Medium | FK shortest path |
| | `hop_count` | Medium | Path length |
| **Silo** | `count` | Easy | Number of connected components |
| | `membership` | Hard | Which component contains X? |
| | `isolation` | Medium | Is table X isolated? |
| | `connected` | Medium | Are X and Y connected? |
| | `full_enumeration` | Hard | List all tables in a component |

### Evaluation Conditions

- **Original** — real table names (tests reasoning + memorization)
- **Obfuscated** — names replaced with `Table_A`, `Table_B`, ... (tests pure structural reasoning)
- **Extended** — hard + medium questions only (harder subset)

## Baselines

| Baseline | Context | Approach |
|---|---|---|
| **Flat Text (FT)** | Full schema as text | Complete graph, maximal context |
| **Vector RAG (VR)** | Top-k retrieved chunks | FAISS embedding similarity (k=15) |
| **Graph-Aug (GA)** | Graph algorithm output | BFS subgraphs, shortest paths, components |

All baselines use the same LLM and system prompt. GA uses NetworkX graph algorithms (not a trained GNN).

## Obfuscation Protocol

The obfuscation condition replaces all table names with random identifiers to measure memorization vs. structural reasoning:

| Baseline | Original EM | Obfuscated EM | Δ (memorization penalty) |
|---|:---:|:---:|:---:|
| Flat Text | 73.9% | 64.9% | **−9.0%** |
| Vector RAG | 68.4% | 50.1% | **−18.2%** |
| Graph-Aug | 70.4% | 49.1% | **−21.3%** |

## Adding a New Model

1. Ensure your model serves an OpenAI-compatible API
2. Run: `python run.py --api-base YOUR_URL --model YOUR_MODEL --condition all`
3. Results saved to `evaluation/results/` with model name in filename
4. Verify: `python integrity_check.py`

## Adding a New Dataset

1. Create `datasets/your_dataset/schema_graph.pt` (PyG HeteroData)
2. Run `python scripts/generate_qa.py --dataset your_dataset`
3. Run `python scripts/obfuscate_schema.py --dataset your_dataset`
4. Run `python scripts/generate_extended_qa.py --dataset your_dataset`
5. Evaluate: `python run.py --api-base ... --model ... --dataset your_dataset`

## Citation

```bibtex
@inproceedings{dwbench2026,
  title={DW-Bench: Benchmarking LLMs on Data Warehouse Graph Topology Reasoning},
  author={[Authors]},
  booktitle={NeurIPS Datasets and Benchmarks},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
