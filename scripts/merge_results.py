"""Merge new 8-question results into existing result files.

Usage: python merge_results.py
  - Reads backed-up results from backup_20260321/
  - Reads new 8-question results from results/ (created by rerun)
  - Merges them (removing stale old-ID duplicates)
  - Recomputes metrics
  - Overwrites results/ files with merged data
"""
import json
import sys
from pathlib import Path

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
from evaluate import compute_aggregate_metrics
from metrics import score_answer

results_dir = Path(__file__).parent.parent / "evaluation" / "results"
backup_dir = results_dir / "backup_20260321"

files = [
    "flat_text_original_adventureworks_gemini-2.5-flash.json",
    "flat_text_original_adventureworks_deepseek-chat.json",
    "vector_rag_original_adventureworks_gemini-2.5-flash.json",
    "vector_rag_original_adventureworks_deepseek-chat.json",
    "graph_aug_original_adventureworks_gemini-2.5-flash.json",
    "graph_aug_original_adventureworks_deepseek-chat.json",
]

# Stale IDs to remove (old numbering that got replaced)
stale_ids = {
    "adventureworks_lineage_multisrc_065",
    "adventureworks_lineage_multisrc_066",
    "adventureworks_lineage_multisrc_067",
    "adventureworks_lineage_multisrc_068",
}

for fname in files:
    backup_path = backup_dir / fname
    new_path = results_dir / fname
    
    if not backup_path.exists():
        print(f"SKIP {fname}: no backup found")
        continue
    if not new_path.exists():
        print(f"SKIP {fname}: no new results found")
        continue
    
    # Load old results (from backup)
    old = json.load(open(backup_path, encoding="utf-8"))
    old_n = len(old["results"])
    
    # Load new results (the 8-question rerun)
    new = json.load(open(new_path, encoding="utf-8"))
    new_n = len(new["results"])
    
    # Remove stale IDs from old results
    merged = [r for r in old["results"] if r["id"] not in stale_ids]
    removed = old_n - len(merged)
    
    # Add new results (skip any already present)
    existing_ids = {r["id"] for r in merged}
    added = 0
    for r in new["results"]:
        if r["id"] not in existing_ids:
            merged.append(r)
            added += 1
    
    # Recompute metrics
    metrics = compute_aggregate_metrics(merged)
    
    # Build final output — preserve original structure
    output = {
        "baseline": old.get("baseline", new.get("baseline", "")),
        "dataset": old.get("dataset", new.get("dataset", "")),
        "obfuscated": old.get("obfuscated", new.get("obfuscated", False)),
        "model": old.get("model", new.get("model", "")),
        "metrics": metrics,
        "results": merged,
    }
    if "model_tag" in old:
        output["model_tag"] = old["model_tag"]
    if "summary" in old:
        output["summary"] = old["summary"]
    
    # Save
    with open(new_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"{fname}: {old_n} - {removed} stale + {added} new = {len(merged)} final (micro_em={metrics['micro_em']:.4f})")

print("\nDone! Run audit_results.py to verify all baselines now have N=208 for AW.")
