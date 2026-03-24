"""Create qa_pairs_missing.json with only the 8 missing questions."""
import json

qa_path = r"d:\job_assignments\PyG_Opensource_contribution\dw-bench\datasets\adventureworks\qa_pairs.json"
out_path = r"d:\job_assignments\PyG_Opensource_contribution\dw-bench\datasets\adventureworks\qa_pairs_missing.json"

missing_ids = [
    "adventureworks_lineage_combined_065",
    "adventureworks_lineage_combined_066",
    "adventureworks_lineage_combined_067",
    "adventureworks_lineage_combined_068",
    "adventureworks_lineage_multisrc_072",
    "adventureworks_lineage_multisrc_073",
    "adventureworks_lineage_multisrc_074",
    "adventureworks_lineage_multisrc_075",
]

qa = json.load(open(qa_path, encoding="utf-8"))
missing = [q for q in qa if q["id"] in missing_ids]
print(f"Found {len(missing)} of {len(missing_ids)} missing questions")
for q in missing:
    print(f"  {q['id']}: {q['question'][:80]}...")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(missing, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")
