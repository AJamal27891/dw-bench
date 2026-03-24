"""Build OMOP CDM v5.4 schema graph for DW-Bench.

Based on the official OHDSI CDM v5.4 specification:
https://ohdsi.github.io/CommonDataModel/cdm54.html

This creates the schema_graph.pt file matching the DW-Bench format.
"""
import torch
from torch_geometric.data import HeteroData

# ──────────────────────────────────────────────────────────────────────
# OMOP CDM v5.4 Tables (grouped by category)
# ──────────────────────────────────────────────────────────────────────

# Clinical Data Tables
CLINICAL_TABLES = [
    "person",
    "observation_period",
    "visit_occurrence",
    "visit_detail",
    "condition_occurrence",
    "drug_exposure",
    "procedure_occurrence",
    "device_exposure",
    "measurement",
    "observation",
    "death",
    "note",
    "note_nlp",
    "specimen",
    "fact_relationship",
    "episode",
    "episode_event",
]

# Health System Tables
HEALTH_SYSTEM_TABLES = [
    "location",
    "care_site",
    "provider",
]

# Health Economics Tables
HEALTH_ECONOMICS_TABLES = [
    "payer_plan_period",
    "cost",
]

# Derived Elements Tables
DERIVED_TABLES = [
    "drug_era",
    "dose_era",
    "condition_era",
]

# Standardized Vocabulary Tables
VOCABULARY_TABLES = [
    "concept",
    "vocabulary",
    "domain",
    "concept_class",
    "concept_relationship",
    "relationship",
    "concept_synonym",
    "concept_ancestor",
    "source_to_concept_map",
    "drug_strength",
]

# Metadata Tables
METADATA_TABLES = [
    "cdm_source",
    "metadata",
]

ALL_TABLES = (CLINICAL_TABLES + HEALTH_SYSTEM_TABLES +
              HEALTH_ECONOMICS_TABLES + DERIVED_TABLES +
              VOCABULARY_TABLES + METADATA_TABLES)

# ──────────────────────────────────────────────────────────────────────
# FK Relationships (source → target based on official CDM spec)
# These represent "source table has a FK column referencing target table"
# ──────────────────────────────────────────────────────────────────────

FK_EDGES = [
    # Person references
    ("person", "location"),           # location_id
    ("person", "provider"),           # provider_id
    ("person", "care_site"),          # care_site_id
    ("person", "concept"),            # gender, race, ethnicity concept_ids

    # Visit
    ("visit_occurrence", "person"),
    ("visit_occurrence", "care_site"),
    ("visit_occurrence", "provider"),
    ("visit_occurrence", "concept"),   # visit_concept_id
    ("visit_detail", "visit_occurrence"),
    ("visit_detail", "person"),
    ("visit_detail", "care_site"),
    ("visit_detail", "provider"),

    # Clinical events → person, visit, provider, concept
    ("condition_occurrence", "person"),
    ("condition_occurrence", "visit_occurrence"),
    ("condition_occurrence", "provider"),
    ("condition_occurrence", "concept"),

    ("drug_exposure", "person"),
    ("drug_exposure", "visit_occurrence"),
    ("drug_exposure", "provider"),
    ("drug_exposure", "concept"),

    ("procedure_occurrence", "person"),
    ("procedure_occurrence", "visit_occurrence"),
    ("procedure_occurrence", "provider"),
    ("procedure_occurrence", "concept"),

    ("device_exposure", "person"),
    ("device_exposure", "visit_occurrence"),
    ("device_exposure", "provider"),
    ("device_exposure", "concept"),

    ("measurement", "person"),
    ("measurement", "visit_occurrence"),
    ("measurement", "provider"),
    ("measurement", "concept"),

    ("observation", "person"),
    ("observation", "visit_occurrence"),
    ("observation", "provider"),
    ("observation", "concept"),

    ("death", "person"),
    ("death", "concept"),

    ("note", "person"),
    ("note", "visit_occurrence"),
    ("note", "provider"),
    ("note", "concept"),
    ("note_nlp", "note"),
    ("note_nlp", "concept"),

    ("specimen", "person"),
    ("specimen", "concept"),

    ("episode", "person"),
    ("episode", "concept"),
    ("episode_event", "episode"),

    # Observation period
    ("observation_period", "person"),

    # Health system
    ("care_site", "location"),
    ("care_site", "concept"),
    ("provider", "care_site"),
    ("provider", "concept"),

    # Health economics
    ("payer_plan_period", "person"),
    ("payer_plan_period", "concept"),
    ("cost", "concept"),

    # Derived/Era tables
    ("drug_era", "person"),
    ("drug_era", "concept"),
    ("dose_era", "person"),
    ("dose_era", "concept"),
    ("condition_era", "person"),
    ("condition_era", "concept"),

    # Vocabulary internal FKs
    ("concept", "vocabulary"),
    ("concept", "domain"),
    ("concept", "concept_class"),
    ("concept_relationship", "concept"),  # concept_id_1, concept_id_2
    ("concept_relationship", "relationship"),
    ("concept_ancestor", "concept"),       # ancestor_concept_id, descendant_concept_id
    ("concept_synonym", "concept"),
    ("source_to_concept_map", "concept"),
    ("source_to_concept_map", "vocabulary"),
    ("drug_strength", "concept"),

    # Fact relationship
    ("fact_relationship", "concept"),      # relationship_concept_id
]

# ──────────────────────────────────────────────────────────────────────
# Lineage (DERIVED_FROM) edges — ETL data flow
# Source data → transformed CDM tables
# ──────────────────────────────────────────────────────────────────────

# In OMOP, the derived tables (eras) are computed FROM clinical tables
# Also, vocabulary tables feed INTO clinical concept resolutions
DERIVED_FROM_EDGES = [
    # Era tables derived from clinical tables
    ("drug_exposure", "drug_era"),          # drug_era computed from drug_exposure
    ("drug_exposure", "dose_era"),          # dose_era computed from drug_exposure
    ("condition_occurrence", "condition_era"),  # condition_era from conditions

    # NLP derived from notes
    ("note", "note_nlp"),

    # Episode events from episodes
    ("episode", "episode_event"),

    # Vocabulary tables feed concept resolution
    ("concept", "condition_occurrence"),    # concept resolution
    ("concept", "drug_exposure"),
    ("concept", "procedure_occurrence"),
    ("concept", "measurement"),
    ("concept", "observation"),
    ("concept", "device_exposure"),
    ("concept", "visit_occurrence"),
    ("concept", "death"),
    ("concept", "specimen"),
    ("concept", "episode"),

    # Source mapping feeds data loading
    ("source_to_concept_map", "person"),
    ("source_to_concept_map", "condition_occurrence"),
    ("source_to_concept_map", "drug_exposure"),
    ("source_to_concept_map", "procedure_occurrence"),
    ("source_to_concept_map", "measurement"),
    ("source_to_concept_map", "observation"),
]

# ──────────────────────────────────────────────────────────────────────
# Build the HeteroData graph
# ──────────────────────────────────────────────────────────────────────

def build_omop_graph():
    """Build OMOP CDM schema graph in DW-Bench format."""
    data = HeteroData()
    data.dataset_name = "omop_cdm"

    name_to_idx = {name: i for i, name in enumerate(ALL_TABLES)}
    n = len(ALL_TABLES)

    # Store table names
    data['table'].table_names = ALL_TABLES

    # Compute structural features [N, 6]:
    # [out_degree, in_degree, norm_degree, lineage_degree, betweenness, pagerank]
    out_degree = [0] * n
    in_degree = [0] * n

    # FK edges
    fk_src, fk_dst = [], []
    for src, dst in FK_EDGES:
        if src in name_to_idx and dst in name_to_idx:
            si, di = name_to_idx[src], name_to_idx[dst]
            fk_src.append(si)
            fk_dst.append(di)
            out_degree[si] += 1
            in_degree[di] += 1

    data['table', 'fk_to', 'table'].edge_index = torch.tensor(
        [fk_src, fk_dst], dtype=torch.long)

    # Lineage edges
    lin_src, lin_dst = [], []
    lineage_degree = [0] * n
    for src, dst in DERIVED_FROM_EDGES:
        if src in name_to_idx and dst in name_to_idx:
            si, di = name_to_idx[src], name_to_idx[dst]
            lin_src.append(si)
            lin_dst.append(di)
            lineage_degree[si] += 1
            lineage_degree[di] += 1

    data['table', 'derived_from', 'table'].edge_index = torch.tensor(
        [lin_src, lin_dst], dtype=torch.long)

    # Compute features
    total_edges = len(fk_src) + len(lin_src)
    max_degree = max(max(out_degree), max(in_degree), 1)

    features = []
    for i in range(n):
        norm_d = (out_degree[i] + in_degree[i]) / max_degree
        # Simple betweenness/pagerank approximation
        betweenness = norm_d * 0.5  # Simplified
        pagerank = (in_degree[i] + 1) / (total_edges + n)
        features.append([
            float(out_degree[i]),
            float(in_degree[i]),
            norm_d,
            float(lineage_degree[i]),
            betweenness,
            pagerank,
        ])

    data['table'].x = torch.tensor(features, dtype=torch.float32)

    return data


if __name__ == '__main__':
    data = build_omop_graph()
    n = len(ALL_TABLES)
    fk_count = data['table', 'fk_to', 'table'].edge_index.shape[1]
    lin_count = data['table', 'derived_from', 'table'].edge_index.shape[1]

    print(f"OMOP CDM v5.4 Schema Graph")
    print(f"  Tables: {n}")
    print(f"  FK edges: {fk_count}")
    print(f"  Lineage edges: {lin_count}")
    print(f"  Features: {data['table'].x.shape}")

    # Validate: no self-loops, no invalid indices
    for etype in [('table', 'fk_to', 'table'), ('table', 'derived_from', 'table')]:
        ei = data[etype].edge_index
        assert ei.min() >= 0, f"Negative index in {etype}"
        assert ei.max() < n, f"Index >= n in {etype}"
        self_loops = (ei[0] == ei[1]).sum().item()
        print(f"  {etype[1]}: self-loops={self_loops}")

    # Save
    out_path = "datasets/omop_cdm/schema_graph.pt"
    torch.save(data, out_path)
    print(f"\n  Saved to: {out_path}")

    # Print table list
    print(f"\n  Tables:")
    for i, t in enumerate(ALL_TABLES):
        print(f"    {i:2d}. {t}")
