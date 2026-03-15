# Session Handover — DW-Bench Project

**Date**: March 15, 2026
**Last commit**: `18c71ff` on `https://github.com/AJamal27891/dw-bench`

---

## Current State: Tier 1 COMPLETE ✅

### What Was Done This Session

1. **Scoring bug discovered and fixed**: `extract_answer()` didn't handle `answer_type='number'` — affected 70 Syn-Logistics questions (hop_count + count). All baselines showed 0% on these when they actually had correct answers.

2. **All results rescored**: 9 Syn-Logistics result files re-extracted and rescored. Real-world datasets NOT affected (they use `'integer'`).

3. **Tool-Use baseline created**: `evaluation/baselines/tool_use.py` — 9 graph algorithm tools, multi-turn LLM conversation. Integrated into `evaluate.py`.

4. **Paper fully updated**: Abstract, introduction, experiments (Table 6), discussion, conclusion, appendix — all corrected. Removed all false "0% structural gap" claims. New narrative: **model-dependent gap** (Gemini 87%, DeepSeek 67-73% on standard baselines, DeepSeek Tool-Use 89.2%).

5. **Hop-distance analysis**: 45.4% of questions exceed 3-hop BFS radius. Added to appendix.

6. **dw-gnn repo created**: `d:\job_assignments\PyG_Opensource_contribution\dw-gnn` — Paper 2 skeleton with docs (proposed_plan, project_story, inference_pipeline).

### Final Corrected Syn-Logistics Results

| Baseline | Gemini 2.5 Flash | DeepSeek-V3 |
|---|---|---|
| Oracle | 97.8% | 100.0% |
| **Tool-Use** | **86.8%** | **89.2%** ⭐ |
| Flat Text | 86.6% | 73.7% |
| Graph-Aug | 83.1% | 66.7% |
| Vector-RAG | 80.9% | 65.9% |

### Key Files Modified
- `evaluation/baselines/flat_text.py` — line 255: `'integer'` → `('integer', 'number')`
- `evaluation/baselines/tool_use.py` — NEW (9 graph tools + multi-turn loop)
- `evaluation/evaluate.py` — Added `tool_use` baseline
- `paper/sections/abstract.tex` — Rewritten (model-dependent gap)
- `paper/sections/experiments.tex` — Table 6 corrected, Key Findings 8-9 rewritten
- `paper/sections/conclusion.tex` — Rewritten (3 honest findings)
- `paper/sections/discussion.tex` — Rewritten (model-dependent gap lead)
- `paper/sections/appendix.tex` — Corrected per-subtype table, added hop-distance + tool-use sections

---

## Next Task: Tier 2 — Value-Node Extension

**Full plan**: `docs/tier2_implementation_plan.md`

### Summary

Add actual row data to schema graphs, creating questions about **lineage-aware value tracing** — e.g., "Which raw rows contributed to this aggregated metric?" These are unsolvable by SQL (lineage edges aren't FK constraints) and unsolvable by FT (millions of rows exceed context windows).

### Immediate Next Steps (New Session)

1. **Start with Syn-Logistics** — we control the data completely
   - Build `scripts/generate_value_data.py`
   - Generate ~1.5M value nodes across 64 tables
   - Build row-level lineage maps
   - Generate ~300 value-level questions

2. **Adapt baselines**
   - GA: BFS on value-node graph
   - VR: Embed rows as chunks
   - Tool-Use: `query_table()` + `trace_lineage()` tools
   - FT: Prove it's impossible (context overflow)

3. **Run and analyze**
   - GA/VR/TU on Tier 2
   - Compare: Tier 1 scores (87%) vs Tier 2 scores (expected: much lower)

4. **Extend to real datasets** (AW, TPC-DI, OMOP)

### Strategic Decision Made This Session

- **Paper 2 (GNN) deprioritized** — scoring bug fix showed Gemini already handles schema-level reasoning (87%). GNN adds marginal value.
- **Consolidated approach**: Keep DW-Bench as a **2-tier benchmark paper** — schema + values in one paper. GNN solution becomes future work / PyG PR.
- **dw-gnn repo** exists at `d:\job_assignments\PyG_Opensource_contribution\dw-gnn` but is on hold pending Tier 2 results.

---

## Repo Locations

| Repo | Path | Purpose |
|---|---|---|
| dw-bench | `d:\job_assignments\PyG_Opensource_contribution\dw-bench` | Main benchmark (Paper 1 + Tier 2) |
| dw-gnn | `d:\job_assignments\PyG_Opensource_contribution\dw-gnn` | Paper 2 GNN (on hold) |

## Running Processes

Kill any lingering Python processes from previous evaluation runs before starting new work.

## API Keys

Located in `dw-bench/.env`:
- `GOOGLE_API_KEY` — Gemini 2.5 Flash
- `DEEPSEEK_API_KEY` — DeepSeek-V3
