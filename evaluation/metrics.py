"""Evaluation metrics for DW-Bench Q&A benchmark.

Supports:
  - Exact match (int, bool, string)
  - Set F1 (precision, recall, f1) for list answers
  - Ordered list match for join paths (with alternative path validation)
"""
import re


def exact_match(predicted, gold) -> bool:
    """Exact match for int, bool, string answers."""
    # Normalize yes/no <-> true/false for boolean-like answers
    BOOL_MAP = {'yes': True, 'no': False, 'true': True, 'false': False}
    
    def to_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.lower().strip() in BOOL_MAP:
            return BOOL_MAP[v.lower().strip()]
        return None
    
    gb = to_bool(gold)
    pb = to_bool(predicted)
    if gb is not None and pb is not None:
        return gb == pb
    
    if isinstance(gold, bool):
        if isinstance(predicted, bool):
            return predicted == gold
        if isinstance(predicted, str):
            return predicted.lower() == str(gold).lower()
        return False
    if isinstance(gold, int):
        try:
            return int(predicted) == gold
        except (ValueError, TypeError):
            return False
    if isinstance(gold, str):
        if predicted is None:
            return False
        return str(predicted).lower().strip() == gold.lower().strip()
    return predicted == gold


def normalize_row_ref(ref: str) -> str:
    """Normalize table:row_N variants to canonical form.

    Handles:
      'table: rows [N]'  → 'table:row_N'
      'table row_N'      → 'table:row_N'
      'table:row_N'      → 'table:row_N' (already canonical)
    """
    if not isinstance(ref, str):
        return str(ref)
    ref = ref.strip()
    # Handle "table: rows [N]" or "table: row [N]" → "table:row_N"
    match = re.search(r'(\w+)[\s:]+rows?\s*[\[\(]?(\d+)', ref)
    if match:
        return f"{match.group(1)}:row_{match.group(2)}"
    # Handle "table row_N" → "table:row_N"
    match = re.search(r'(\w+)\s+row_(\d+)', ref)
    if match:
        return f"{match.group(1)}:row_{match.group(2)}"
    return ref


def set_f1(predicted: list, gold: list) -> dict:
    """Set-based F1 score for list answers.

    Treats both predicted and gold as sets, computes:
      - Precision: |predicted ∩ gold| / |predicted|
      - Recall:    |predicted ∩ gold| / |gold|
      - F1:        harmonic mean
    """
    if not isinstance(predicted, list):
        predicted = []
    if not isinstance(gold, list):
        gold = []

    pred_set = {normalize_row_ref(p) for p in predicted}
    gold_set = {normalize_row_ref(g) for g in gold}

    if not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact": True}
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact": False}

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(gold_set) if gold_set else 0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0)
    is_exact = (pred_set == gold_set)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact": is_exact,
    }


def _is_valid_fk_path(path: list, fk_adj: dict, name_to_idx: dict) -> bool:
    """Check if every hop in the path is a valid FK edge (undirected)."""
    for i in range(len(path) - 1):
        a_idx = name_to_idx.get(path[i])
        b_idx = name_to_idx.get(path[i + 1])
        if a_idx is None or b_idx is None:
            return False
        # Check undirected: a->b or b->a
        if b_idx not in fk_adj.get(a_idx, set()) and a_idx not in fk_adj.get(b_idx, set()):
            return False
    return True


def ordered_list_match(predicted: list, gold: list,
                      fk_adj: dict = None, name_to_idx: dict = None) -> dict:
    """Match for ordered lists (join paths).

    Returns set F1 + an order_correct flag.
    If fk_adj and name_to_idx are provided, also checks whether predicted
    is a valid alternative shortest FK path (same length, same endpoints,
    every hop is a valid FK edge).
    """
    f1_result = set_f1(predicted, gold)
    # Check if order matches exactly
    order_correct = (predicted == gold) if isinstance(predicted, list) else False
    f1_result["order_correct"] = order_correct

    # Check if prediction is a valid alternative shortest path
    valid_alt_path = False
    if (not f1_result["exact"] and fk_adj is not None and name_to_idx is not None
            and isinstance(predicted, list) and isinstance(gold, list)
            and len(predicted) >= 2 and len(gold) >= 2):
        same_length = len(predicted) == len(gold)
        same_endpoints = (predicted[0] == gold[0] and predicted[-1] == gold[-1])
        if same_length and same_endpoints:
            valid_alt_path = _is_valid_fk_path(predicted, fk_adj, name_to_idx)

    f1_result["valid_alt_path"] = valid_alt_path
    if valid_alt_path:
        # Override: this is a correct answer via an alternative path
        f1_result["exact"] = True
        f1_result["f1"] = 1.0
        f1_result["precision"] = 1.0
        f1_result["recall"] = 1.0
    return f1_result


def score_answer(predicted, gold, answer_type: str, **kwargs) -> dict:
    """Score a single prediction against gold, dispatching by answer type.

    For ordered_list (join paths), pass fk_adj and name_to_idx kwargs
    to enable validation of alternative shortest paths.
    """
    if answer_type in ("integer", "boolean", "string"):
        match = exact_match(predicted, gold)
        return {"exact_match": match, "f1": 1.0 if match else 0.0}
    elif answer_type == "list":
        result = set_f1(predicted, gold)
        return {"exact_match": result["exact"], "f1": result["f1"],
                "precision": result["precision"], "recall": result["recall"]}
    elif answer_type == "ordered_list":
        result = ordered_list_match(
            predicted, gold,
            fk_adj=kwargs.get('fk_adj'),
            name_to_idx=kwargs.get('name_to_idx'))
        return {"exact_match": result["exact"], "f1": result["f1"],
                "precision": result["precision"], "recall": result["recall"],
                "order_correct": result["order_correct"],
                "valid_alt_path": result.get("valid_alt_path", False)}
    else:
        match = exact_match(predicted, gold)
        return {"exact_match": match, "f1": 1.0 if match else 0.0}
