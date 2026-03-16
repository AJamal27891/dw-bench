# DW-Bench: Benchmarking LLMs on Data Warehouse Graph Topology Reasoning

[![NeurIPS 2026](https://img.shields.io/badge/Paper-NeurIPS%202026-4361EE)](paper/)
[![Journal of Big Data](https://img.shields.io/badge/Paper-Journal%20of%20Big%20Data-FF6B35)](paper_journal/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg)](https://python.org)
[![Datasets](https://img.shields.io/badge/Datasets-5-orange)](datasets/)
[![Questions](https://img.shields.io/badge/Tier%201-1%2C046%20questions-blueviolet)]()
[![Questions](https://img.shields.io/badge/Tier%202-433%20questions-blueviolet)]()

**DW-Bench** is the first benchmark for evaluating whether Large Language Models can reason about the *graph topology* of data warehouse schemas — data lineage chains, foreign key paths, connected components, and row-level provenance — rather than just generating SQL queries.

---

## 📊 Main Results

### Tier 1: Schema-Level Reasoning (1,046 questions, 5 datasets)

| Baseline | Gemini Micro-EM | DeepSeek Micro-EM |
|:---|:---:|:---:|
| Flat Text (FT) | 76.8 ± 2.4 | 69.7 ± 2.8 |
| Vector-RAG (VR) | 73.2 ± 2.6 | 71.2 ± 2.8 |
| Graph-Aug (GA) | 75.7 ± 2.5 | 77.0 ± 2.6 |
| **Tool-Use (TU)** | **89.3 ± 1.9** | **90.4 ± 1.7** |
| ReAct-Code (RC) | 81.4 ± 2.4 | 82.1 ± 4.2† |
| Oracle | 97.0 ± 1.0 | 97.2 ± 1.0 |

> *95% bootstrap CIs (2000 resamples). †DeepSeek RC: 2/5 datasets.*

### Key Figures

<p align="center">
  <img src="paper/figures/difficulty_comparison.png" width="85%" alt="EM by difficulty level">
</p>

**Left**: All baselines plateau on hard questions at ~60% while Oracle achieves >97%. **Right**: DeepSeek shows similar patterns but larger gaps between static baselines.

<p align="center">
  <img src="paper/figures/subtype_heatmap.png" width="55%" alt="Oracle-gap heatmap">
</p>

**Gap to Oracle by subtype** (darker = larger gap). `combined_impact` and `membership` remain universally hard across all baselines — the frontier for structural graph reasoning.

### Tier 2: Value-Level Reasoning (433 questions, Syn-Logistics)

| Baseline | Overall EM | cascade_count | row_provenance | multi_hop_trace |
|:---|:---:|:---:|:---:|:---:|
| FT<sub>v2</sub> | 30% | 0% | 0% | 30% |
| GA<sub>v2</sub> | 31% | 0% | 8% | 30% |
| **TU<sub>v2</sub>** | **59%** | **100%** | **92%** | 27% |

> Tools are **essential** for data-access subtypes (100% vs 0%) yet **insufficient** for compositional reasoning (27%).

### Obfuscation (Contamination Control)

| Baseline | Gemini Δ | DeepSeek Δ |
|:---|:---:|:---:|
| Flat Text | −14.9% | −27.8% |
| Vector-RAG | −26.0% | −31.7% |
| Graph-Aug | −27.9% | −24.3% |
| **Tool-Use** | **−3.4%** | **−4.2%** |

> Tool-Use is nearly **obfuscation-invariant** — confirming algorithmic graph access over semantic name memorization.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torch_geometric openai requests sentence-transformers faiss-cpu networkx
```

### 2. Run Evaluation

```bash
# Using Gemini
python run.py \
  --api-base https://generativelanguage.googleapis.com/v1beta/openai \
  --model gemini-2.5-flash \
  --api-key YOUR_API_KEY

# Using DeepSeek
python run.py \
  --api-base https://api.deepseek.com/v1 \
  --model deepseek-chat \
  --api-key YOUR_API_KEY

# Using a local model (LM Studio, Ollama, vLLM)
python run.py \
  --api-base http://localhost:1234/v1 \
  --model microsoft/phi-4-mini-reasoning

# Run specific conditions
python run.py --api-base ... --model ... --condition obfuscated
python run.py --api-base ... --model ... --condition extended
python run.py --api-base ... --model ... --condition all

# Single baseline + dataset
python run.py --api-base ... --model ... --baseline tool_use --dataset tpc-ds
```

### 3. View Results

```bash
python view_results.py              # Summary tables
python integrity_check.py           # Validate result files
```

---

## 📁 Project Structure

```
dw-bench/
├── run.py                          # 🏃 Unified evaluation runner
├── view_results.py                 # 📊 Results viewer
├── integrity_check.py              # ✅ Result validator
│
├── datasets/                       # 📁 Schema graphs + QA pairs
│   ├── adventureworks/             #    102 tables, 136 FK, 39 lineage
│   ├── tpc-ds/                     #     24 tables, 70 FK
│   ├── tpc-di/                     #     35 tables, 29 FK, 21 lineage
│   ├── omop_cdm/                   #     37 tables, 74 FK, 21 lineage
│   └── syn_logistics/              #     64 tables, 96 FK, 35 lineage
│
├── evaluation/                     # 🔬 Evaluation pipeline
│   ├── evaluate.py                 #    Main evaluation orchestrator
│   ├── metrics.py                  #    Scoring (EM, F1, path validation)
│   ├── baselines/                  #    6 baseline implementations
│   │   ├── flat_text.py            #      Full schema as text
│   │   ├── vector_rag.py           #      FAISS embedding retrieval
│   │   ├── graph_aug.py            #      BFS graph neighborhoods
│   │   ├── tool_use.py             #      Agentic graph tools
│   │   ├── react_code.py           #      Python/NetworkX code gen
│   │   └── oracle.py               #      Gold algorithmic output
│   └── results/                    #    Evaluation outputs (JSON)
│
├── paper/                          # 📄 NeurIPS 2026 (conference)
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   └── neurips_2026.sty
│
├── paper_journal/                  # 📄 Journal of Big Data (Springer)
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   └── sn-jnl.cls
│
└── scripts/                        # 🛠️ Data pipeline
    ├── generate_qa.py
    ├── obfuscate_schema.py
    └── ...
```

---

## 🧪 Benchmark Design

### Datasets (262 tables, 521 edges)

| Dataset | Domain | Tables | FK | Lineage | Tier 1 Qs | Tier 2 Qs | Silos |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| AdventureWorks | Retail/HR | 102 | 136 | 39 | 208 | — | 11 |
| TPC-DS | Analytics | 24 | 70 | 0 | 127 | — | 1 |
| TPC-DI | ETL | 35 | 29 | 21 | 181 | — | 2 |
| OMOP CDM | Healthcare | 37 | 74 | 21 | 158 | — | 3 |
| Syn-Logistics | Supply Chain | 64 | 96 | 35 | 372 | 433 | 5 |
| **Total** | | **262** | **405** | **116** | **1,046** | **433** | |

### Question Taxonomy (Tier 1: 13 subtypes, 3 difficulty levels)

| Category | Subtype | Difficulty | Description |
|:---|:---|:---|:---|
| **Lineage** | `forward` | Easy | Direct lineage targets |
| | `reverse` | Easy | Source tables for a DW table |
| | `transitive` | Hard | Multi-hop lineage chains |
| | `combined_impact` | Hard | Lineage + FK dependents |
| | `multi_source` | Medium | Tables with 3+ sources |
| **Route** | `direct_fk` | Easy | FK adjacency check |
| | `join_path` | Medium | FK shortest path |
| | `hop_count` | Medium | Path length |
| **Silo** | `count` | Easy | Number of components |
| | `membership` | Hard | Which component contains X? |
| | `isolation` | Medium | Is table X isolated? |
| | `connected` | Medium | Are X and Y connected? |
| | `full_enumeration` | Hard | List all tables in a silo |

### Tier 2: Value-Level (8 subtypes)

`cascade_count` · `row_provenance` · `row_impact` · `value_origin` · `multi_hop_trace` · `value_propagation` · `cross_silo_reachability` · `shared_source`

---

## 🔧 Six Baselines

| Baseline | Context | Approach |
|:---|:---|:---|
| **Flat Text (FT)** | Full schema as text | Complete graph, maximal context |
| **Vector RAG (VR)** | Top-k retrieved chunks | Sentence-BERT + FAISS (k=15) |
| **Graph-Aug (GA)** | BFS neighborhood | 3-hop subgraph from mentioned tables |
| **Tool-Use (TU)** | Schema + 9 graph tools | Agentic: multi-turn tool calling |
| **ReAct-Code (RC)** | Schema + Python exec | Code generation on NetworkX graph |
| **Oracle** | Gold algorithmic output | Upper bound (perfect retrieval) |

---

## 🔄 Evaluation Conditions

- **Original** — Real table names (tests reasoning + memorization)
- **Obfuscated** — Names replaced with `Table_A`, `Table_B`, ... (tests pure structural reasoning)
- **Extended** — Hard + medium questions only (harder subset)

---

## ➕ Extending DW-Bench

### Adding a New Model

```bash
python run.py --api-base YOUR_URL --model YOUR_MODEL --condition all
python integrity_check.py
```

### Adding a New Dataset

1. Create `datasets/your_dataset/schema_graph.pt` (PyG HeteroData)
2. Run `python scripts/generate_qa.py --dataset your_dataset`
3. Run `python scripts/obfuscate_schema.py --dataset your_dataset`
4. Evaluate: `python run.py --api-base ... --model ... --dataset your_dataset`

---

## 📄 Papers

### Conference Version (NeurIPS 2026)

```
paper/
├── main.tex        # 10 pages, neurips_2026 template
├── sections/       # Abstract through Appendix
└── figures/        # 6 figures (colorblind-safe, bootstrap CIs)
```

### Journal Version (Journal of Big Data, Springer)

```
paper_journal/
├── main.tex        # ~15 pages, sn-jnl template
├── sections/       # Expanded sections with formal definitions
└── figures/        # Shared figures
```

---

## 📝 Citation

If you use DW-Bench in your research, please cite:

```bibtex
@inproceedings{ahmed2026dwbench,
  title     = {DW-Bench: Benchmarking LLMs on Data Warehouse Graph Topology Reasoning},
  author    = {Ahmed, Ahmed G.A.H.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)
               Datasets and Benchmarks Track},
  year      = {2026},
  url       = {https://github.com/AJamal27891/dw-bench}
}
```

For the extended journal version:

```bibtex
@article{ahmed2026dwbench_journal,
  title     = {DW-Bench: A Two-Tier Benchmark for Graph Topology Reasoning
               over Data Warehouse Schemas},
  author    = {Ahmed, Ahmed G.A.H.},
  journal   = {Journal of Big Data},
  publisher = {Springer},
  year      = {2026},
  url       = {https://github.com/AJamal27891/dw-bench}
}
```

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
