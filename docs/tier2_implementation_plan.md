# Tier 2: Value-Node Extension — Implementation Plan

## Context

Tier 1 (schema topology, 1,042 questions) is **complete and pushed** (commit `18c71ff`).
Tier 1 showed that frontier LLMs handle schema-level reasoning well (Gemini 87%, DeepSeek Tool-Use 89.2%) because schemas fit in context.

Tier 2 adds **value-level nodes** (actual row data) to the graph, creating questions that require lineage-aware data traversal — something no LLM can brute-force because millions of value nodes exceed all context windows.

---

## Data Acquisition Plan

### 1. Syn-Logistics (We Define) ⭐ Start Here

**Source**: Generate synthetic row data ourselves.

| Table Layer | # Tables | Rows/Table | Example Columns |
|---|---|---|---|
| Raw (OLTP) | 16 | 5,000-10,000 | order_id, supplier_id, product_sku, timestamp, quantity, price |
| Staging | 16 | 3,000-8,000 | Cleaned/deduped versions of raw with audit columns |
| Dimension | 16 | 500-2,000 | Surrogate keys, descriptive attributes |
| Mart | 16 | 200-1,000 | Aggregated metrics, KPIs |

**Total**: ~64 tables × ~3,000 avg rows × ~8 columns = **~1.5M value nodes**

**Lineage Implementation**: Each row tracks its `source_row_ids` through a lookup table:
```python
# lineage_map.json per table pair
{
    "stg_purchase_orders": {
        "row_42": {"source_table": "raw_purchase_orders", "source_rows": [101, 102, 103]},
        ...
    }
}
```

**Effort**: ~4-6 hours (Python data generator)

### 2. AdventureWorks

**Source**: Microsoft official sample database (free download)
- URL: https://learn.microsoft.com/en-us/sql/samples/adventureworks-install-configure
- Contains both OLTP (Person, Sales, Production) and DW (Dim*, Fact*) databases
- CSV exports available via standard SQL Server backup restore + bcp

**Lineage**: Already defined in our schema graph (39 `derived_from` edges).
We trace actual row correspondences via shared business keys:
- `OLTP.Person.Person.BusinessEntityID` → `DW.DimCustomer.CustomerKey`
- `OLTP.Sales.SalesOrderHeader.SalesOrderID` → `DW.FactInternetSales.SalesOrderNumber`

**Total**: ~102 tables × ~1,000-20,000 rows = **~2-5M value nodes**

**Effort**: ~3-4 hours (download + extract + map lineage)

### 3. TPC-DS

**Source**: Official TPC-DS data generator (`dsdgen`)
- URL: https://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp
- Generate at Scale Factor 1 (1GB) → ~25 tables × ~10K-1M rows

**Lineage**: TPC-DS has NO lineage edges (pure star/snowflake FK graph).
Options:
- a) Skip Tier 2 for TPC-DS (it's FK-only, lineage questions don't apply)
- b) Add synthetic lineage edges (dishonest — violates TPC-DS spec)

**Recommendation**: Skip TPC-DS for Tier 2. Focus on datasets WITH lineage.

**Effort**: ~2 hours if included

### 4. TPC-DI

**Source**: Official TPC-DI data generator (`DIGen`)
- The TPC-DI spec is specifically about **data integration** (ETL)
- Has explicit staging → dimension → fact transformation flows
- Best real-world dataset for lineage-tracing questions

**Total**: ~35 tables × ~5K-50K rows = **~1-5M value nodes**

**Effort**: ~3-4 hours (generate + extract lineage maps)

### 5. OMOP CDM

**Source**: Synthea synthetic patient generator
- URL: https://synthea.mitre.org/
- Generates realistic patient records mapped to OMOP CDM
- 1,000 patients → ~37 tables × ~5K-50K rows

**Lineage**: OHDSI ETL conventions (vocabulary → standardized → analysis tables).
Row-level: `source_value` columns in OMOP explicitly track source origins.

**Total**: ~37 tables × ~5K rows = **~500K value nodes**

**Effort**: ~4-5 hours (generate + map)

---

## Value-Level Question Types

### Category: Data Provenance (Backward Lineage Tracing)

| Subtype | Example | Answer Type | Difficulty |
|---|---|---|---|
| `row_provenance` | "Which raw_purchase_orders rows contributed to the aggregated metric in mart_supplier_scorecard where supplier='Acme'?" | list of row IDs | hard |
| `value_origin` | "The value 'Widget-A' in dim_product.name — which raw table did it originate from?" | string (table name) | medium |
| `multi_hop_trace` | "Trace the lineage: which stg_ table processed the raw record with order_id=5432 before it reached the mart?" | ordered_list | medium |

### Category: Forward Impact Analysis

| Subtype | Example | Answer Type | Difficulty |
|---|---|---|---|
| `row_impact` | "If raw_purchase_orders row with order_id=5432 is deleted, which mart table rows lose data?" | list | hard |
| `value_propagation` | "Does the value supplier_id=7 in raw_suppliers propagate to any mart-level aggregate?" | boolean | medium |
| `cascade_count` | "How many downstream rows across all tables are affected if we delete all rows where region='West' from raw_shipments?" | number | hard |

### Category: Cross-Silo Data Queries

| Subtype | Example | Answer Type | Difficulty |
|---|---|---|---|
| `cross_silo_reachability` | "Does any data from the HR silo's raw_employees flow into the Finance silo's mart_revenue?" | boolean | hard |
| `shared_source` | "Which raw source tables feed data into BOTH mart_delivery_performance AND mart_supplier_scorecard?" | list | medium |

### Target: ~300-500 questions across 8 subtypes

---

## Baseline Adaptations for Tier 2

### Flat Text: ❌ IMPOSSIBLE
Cannot fit millions of rows in context. This IS the point — proves the LLM Context Asymptote.

### Graph-Augmented: ✅ Adapt
- BFS from question-referenced nodes (e.g., `supplier='Acme'`)
- Extract k-hop subgraph WITH value data
- Feed extracted subgraph text to LLM
- Challenge: BFS on value graph is much larger; need smart filtering

### Vector-RAG: ✅ Adapt
- Embed each table's rows as chunks
- Retrieve top-k relevant chunks per question
- Challenge: multi-hop questions span multiple tables; single-hop retrieval fails

### Tool-Use: ✅ Adapt
- Give LLM tools: `query_table(table, filter)`, `trace_lineage(table, row_id, direction)`
- LLM must chain tool calls to follow lineage
- Challenge: LLM must figure out multi-hop tool chaining

### Oracle: ✅ Same approach
- Inject gold evidence (the actual lineage trace result)
- Proves questions are solvable

---

## PyG Graph Structure for Tier 2

```python
data = HeteroData()

# Node types
data['table'].x = ...          # N_tables × d (structural features)
data['row'].x = ...            # N_rows × d (value embeddings)

# Edge types
data['table', 'fk_to', 'table'].edge_index = ...           # Schema FK
data['table', 'derived_from', 'table'].edge_index = ...     # Schema lineage
data['row', 'belongs_to', 'table'].edge_index = ...         # Row → table membership
data['row', 'fk_ref', 'row'].edge_index = ...               # Row-level FK references
data['row', 'derived_from_row', 'row'].edge_index = ...     # Row-level lineage
```

This creates a **multi-scale heterogeneous graph**: schema-level + row-level.

---

## Implementation Order

### Phase A: Syn-Logistics Value Data (Week 1)
1. Build `generate_value_data.py` — creates row-level data for each table
2. Build `generate_lineage_map.py` — tracks row-to-row lineage
3. Build `generate_value_qa.py` — creates Tier 2 questions
4. Build value-node PyG graph
5. Test: can Oracle achieve 100%?

### Phase B: Baselines on Syn-Logistics (Week 1-2)
6. Adapt `graph_aug.py` for value-node graphs
7. Adapt `vector_rag.py` for row-level chunks
8. Adapt `tool_use.py` with `query_table()` and `trace_lineage()` tools
9. Run baselines, analyze results

### Phase C: Real-World Datasets (Week 2-3)
10. Download/generate row data for AW, TPC-DI, OMOP
11. Build lineage maps for each
12. Generate value-level questions
13. Run baselines on all datasets

### Phase D: Paper Integration (Week 3)
14. Add Tier 2 section to paper
15. Update abstract, conclusion with 2-tier findings
16. Push final version

---

## Success Criteria

Tier 2 is a success if:
1. **FT is provably impossible** (context overflow confirmed)
2. **GA/VR score significantly lower than Tier 1** (proving the scale gap)
3. **Oracle still achieves ~100%** (questions are well-formed)
4. **Tool-Use provides an interesting middle ground** (can chain lineage tools?)
5. **At least 3 datasets have value-level results**

---

## Files to Create

| File | Purpose |
|---|---|
| `datasets/syn_logistics/value_data/` | Generated row data CSVs |
| `datasets/syn_logistics/lineage_map.json` | Row-level lineage tracking |
| `datasets/syn_logistics/qa_pairs_tier2.json` | Value-level questions |
| `datasets/syn_logistics/value_schema_graph.pt` | PyG HeteroData with value nodes |
| `evaluation/baselines/graph_aug_v2.py` | Value-aware GA baseline |
| `evaluation/baselines/tool_use_v2.py` | Value-aware tool-use baseline |
| `scripts/generate_value_data.py` | Synthetic row data generator |
| `scripts/generate_value_qa.py` | Value-level question generator |
