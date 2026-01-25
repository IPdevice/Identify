import json
import os
from typing import Any, Dict, List, Tuple, Optional


DATA_PATH = ".res.json"

# Cache preprocessed keywords to avoid repeated processing
# Use hash of keyword tuple as key to ensure same keyword list uses same cache
_keywords_cache: Dict[Tuple, List[str]] = {}


def ensure_rule_levels(rules: List[Dict[str, Any]]) -> None:
    for rule in rules:
        if "level" not in rule:
            rule["level"] = 1


def load_rules_with_lock(rules_path: str, compile_for_performance: bool = True) -> List[Dict[str, Any]]:
    """
    Load rule file
    
    Args:
        rules_path: Path to rule file
        compile_for_performance: Whether to compile for performance optimization (not used currently, keep interface compatibility)
    
    Returns:
        List of rules
    """
    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"Rule file not found: {rules_path}")
    
    with open(rules_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            rules: List[Dict[str, Any]] = []
        else:
            rules: List[Dict[str, Any]] = json.loads(content)
    
    # Ensure rules have level field
    ensure_rule_levels(rules)
    
    return rules


def save_rules_with_lock(rules_path: str, rules: List[Dict[str, Any]]) -> None:
    """
    Save rule file
    
    Args:
        rules_path: Path to rule file
        rules: List of rules to save
    """
    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)


def flatten_to_text(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, (str, int, float, bool)):
        return str(data)
    if isinstance(data, list):
        return "\n".join(flatten_to_text(x) for x in data)
    if isinstance(data, dict):
        parts: List[str] = []
        for k, v in data.items():
            parts.append(str(k))
            parts.append(flatten_to_text(v))
        return "\n".join(parts)
    return json.dumps(data, ensure_ascii=False)


def rule_matches_text(rule: Dict[str, Any], text: str) -> bool:
    """
    Optimized rule matching function using more efficient string search strategy
    - Use cache to avoid repeated preprocessing of keywords
    - Sort keywords by length, check shorter keywords first to quickly exclude unmatched rules
    """
    keywords = rule.get("keywords", [])
    if not keywords:
        return False
    
    # Convert keyword list to tuple as cache key (same keyword list uses same cache)
    # First normalize to string and lowercase to ensure consistency
    normalized_keywords = tuple(
        str(kw).lower() for kw in keywords
    )
    
    # Try to get preprocessed keywords from cache
    if normalized_keywords not in _keywords_cache:
        # Preprocess keywords: already converted to lowercase, only need to sort by length
        # Shorter keywords first to quickly exclude unmatched rules
        processed_keywords = list(normalized_keywords)
        processed_keywords.sort(key=len)
        _keywords_cache[normalized_keywords] = processed_keywords
    
    processed_keywords = _keywords_cache[normalized_keywords]
    
    # Quick check: if shortest keyword not in text, return False immediately
    # Use efficient string search (Python's 'in' operation is already highly optimized)
    for kw in processed_keywords:
        if kw not in text:
            return False
    
    return True


def choose_final_label(matched_rules: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    if not matched_rules:
        return "", []

    # Group by level, prioritize smallest level (smaller number means higher level)
    level_to_rules: Dict[int, List[Dict[str, Any]]] = {}
    for r in matched_rules:
        level_to_rules.setdefault(int(r.get("level", 1)), []).append(r)

    chosen_level = min(level_to_rules.keys())
    candidate_rules = level_to_rules[chosen_level]

    # Calculate mode label among highest level rules
    label_count: Dict[str, int] = {}
    for r in candidate_rules:
        label = r.get("label", "")
        label_count[label] = label_count.get(label, 0) + 1

    # Choose label with highest frequency; take first occurrence in case of tie
    max_count = max(label_count.values())
    final_label = None
    for r in candidate_rules:
        label = r.get("label", "")
        if label_count[label] == max_count:
            final_label = label
            break

    if final_label is None:
        final_label = ""

    return final_label, candidate_rules


def downgrade_inconsistent_rules(all_matched: List[Dict[str, Any]], final_label: str, highest_level_rules: List[Dict[str, Any]]) -> List[int]:
    """Only downgrade inconsistent labeled rules at highest level"""
    downgraded_indices: List[int] = []
    
    # Find highest level
    if not highest_level_rules:
        return downgraded_indices
    
    highest_level = min(int(r.get("level", 1)) for r in highest_level_rules)
    
    # Only downgrade rules at highest level with inconsistent labels
    for idx, r in enumerate(all_matched):
        if int(r.get("level", 1)) == highest_level and r.get("label", "") != final_label:
            r["level"] = highest_level + 1
            downgraded_indices.append(idx)
    
    return downgraded_indices


def clean1(sample, rules_path: str, rules: Optional[List[Dict[str, Any]]] = None, verbose: bool = True) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Check if sample matches rules
    
    Args:
        sample: Sample data to check
        rules_path: Path to rule file
        rules: Optional list of rules (avoids repeated file reading if provided)
        verbose: Whether to output detailed information
    
    Returns:
        Tuple[bool, Optional[List[Dict]], Optional[str]]:
            - is_matched: Whether any rule was matched
            - updated_rules: Updated list of rules (if rules were modified), None otherwise
            - final_label: Final label (if matched), None otherwise
    """
    # Read from file if rules list not provided
    if rules is None:
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"Rule file not found: {rules_path}")

        with open(rules_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                rules: List[Dict[str, Any]] = []
            else:
                rules: List[Dict[str, Any]] = json.loads(content)
    else:
        # Use passed rules list, create copy to avoid modifying original list
        rules = [rule.copy() for rule in rules]

    ensure_rule_levels(rules)

    sample_text = flatten_to_text(sample).lower()

    matched_rules: List[Dict[str, Any]] = []
    matched_indices: List[int] = []
    for i, rule in enumerate(rules):
        if rule_matches_text(rule, sample_text):
            matched_rules.append(rule)
            matched_indices.append(i)

    if verbose:
        print(f"Number of rules matched by sample: {len(matched_rules)}")
    
    # Return False immediately if no rules matched
    if not matched_rules:
        if verbose:
            print("No rules matched")
        return False, None, None
    
    labels = list({r.get("label", "") for r in matched_rules})
    if verbose:
        print(f"\nMatched label set: {labels}")
    
        # Output detailed information of matched rules
        print("\n=== Details of Matched Rules ===")
        for i, rule in enumerate(matched_rules):
            print(f"\nRule {i+1}:")
            print(f"  Label: {rule.get('label', 'N/A')}")
            print(f"  Level: {rule.get('level', 'N/A')}")
            print(f"  Keywords: {rule.get('keywords', [])}")

    final_label, candidate_rules = choose_final_label(matched_rules)
    if verbose:
        print(f"\nFinal label: {final_label if final_label else 'None'} (Total {len(candidate_rules)} rules at candidate level)")

    downgraded_local_indices: List[int] = []
    rules_modified = False
    if matched_rules and final_label:
        downgraded_local_indices = downgrade_inconsistent_rules(matched_rules, final_label, candidate_rules)
        if downgraded_local_indices:
            rules_modified = True

    # Sync modifications in matched_rules back to original rules (objects are same reference, indices only for display)
    downgraded_global_indices = [matched_indices[i] for i in downgraded_local_indices]
    if verbose:
        if downgraded_global_indices:
            print(f"Downgraded rule indices: {downgraded_global_indices}")
        else:
            print("No rules need to be downgraded")

    # Note: Do not save file here, let caller save uniformly at end of batch to avoid repeated saving
    # Removed duplicate save logic, label saving managed by main program
    
    # Return match result, updated rules list (if modified), and final label
    if rules_modified:
        return True, rules, final_label
    else:
        return True, None, final_label


if __name__ == "__main__":
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    sample = data[10]
    is_matched, updated_rules, final_label = clean1(sample, "../rules.json")
    if is_matched:
        print("Matched rules")
        print(f"Final label: {final_label}")
    else:
        print("No rules matched")