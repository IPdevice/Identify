from hmac import new
import json
import copy
from typing import Any, Dict, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(__file__))
from fast_index_wrapper import ImprovedDynamicIndexer
from clean_rules_1 import flatten_to_text, rule_matches_text, choose_final_label, downgrade_inconsistent_rules
from collections import Counter
match_ratio_threshold=0.05

def extract_keywords_from_input(keyword_input) -> List[str]:
    """
    Unified keyword extraction function to extract list of keyword strings from various input formats
    
    Args:
        keyword_input: Can be list, dict, or string
        
    Returns:
        List of keyword strings
    """
    keywords = []
    if isinstance(keyword_input, list):
        for kw in keyword_input:
            if isinstance(kw, dict) and 'keywords' in kw:
                keywords.append(str(kw['keywords']).strip())
            elif isinstance(kw, str):
                keywords.append(kw.strip())
            else:
                keywords.append(str(kw).strip())
    elif isinstance(keyword_input, dict) and 'keywords' in keyword_input:
        return extract_keywords_from_input(keyword_input['keywords'])
    elif isinstance(keyword_input, str):
        keywords.append(keyword_input.strip())
    
    # Filter empty keywords
    return [kw for kw in keywords if kw]

def lists_equal_fast(list1: List, list2: List) -> bool:
    """
    Fast comparison to check if two lists are equal
    Use direct comparison for short lists, check length and hash first for long lists
    """
    if len(list1) != len(list2):
        return False
    # Direct comparison is faster for short lists
    if len(list1) < 100:
        return list1 == list2
    # For long lists, check hash first, then full comparison if hashes match
    if hash(tuple(list1)) != hash(tuple(list2)):
        return False
    return list1 == list2

def jaccard_similarity(a: List[str], b: List[str]) -> float:
    """Calculate Jaccard similarity between two keyword lists.

    Normalize by lowercasing and stripping whitespace.
    """
    set_a = {str(x).strip() for x in a if str(x).strip()}
    set_b = {str(x).strip() for x in b if str(x).strip()}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection_size = len(set_a & set_b)
    union_size = len(set_a | set_b)
    return intersection_size / union_size if union_size else 0.0

def clean_rules(new_rule: Dict[str, Any], threshold: float = 0.5, 
                 rules: List[Dict[str, Any]] = None, rules_file: str = './rules.json') -> List[Dict[str, Any]]:
    # Read from file if rules parameter not provided
    if rules is None:
        try:
            with open(rules_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    rules = []
                else:
                    rules = json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            rules = []
    
    # Initialize "level" field (do not save here, let caller save uniformly to avoid duplicate saves)
    for rule in rules:
        if "level" not in rule:
            rule["level"] = 1

    new_keywords = new_rule.get('keywords', [])
    results: List[Dict[str, Any]] = []
    
    # Perform similarity filtering
    print(f"Starting similarity filtering with threshold: {threshold}")
    
    # Iterate through all rules and calculate similarity
    for i, rule in enumerate(rules):
        rule_keywords = rule.get('keywords', [])
        if not rule_keywords:  # Skip rules without keywords
            continue
            
        # Calculate Jaccard similarity
        similarity = jaccard_similarity(new_keywords, rule_keywords)
        
        if similarity >= threshold:
            results.append({
                'similarity': similarity,
                'keywords': rule_keywords,
                'label': rule.get('label', ''),
                'level': rule.get('level', 1),
                'rule_index': i
            })
    
    print(f"Found {len(results)} candidate rules after similarity filtering")
    
    # No deduplication, keep all results directly
    print(f"No deduplication performed, keeping {len(results)} rules")
    
    # Sort by similarity
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return results

def check_keywords(s, keywords):
    """
    Check if all keywords in the keyword list exist in the string (case-sensitive, only need to contain all keywords, no longer require exact phrase boundary matching)

    Args:
        s: String or dictionary to check
        keywords: List of keywords

    Returns:
        1: If all keywords exist in the string
        0: If at least one keyword does not exist in the string
    """
    if not keywords:
        return 1

    # Convert to string if s is a dictionary
    if isinstance(s, dict):
        # Concatenate all values of the dictionary into a string
        s = ' '.join(str(v) for v in s.values() if v)
    elif not isinstance(s, str):
        s = str(s)

    # Case-sensitive check if all keywords are in the string
    for keyword in keywords:
        if keyword not in s:
            return 0

    return 1

def is_matched(list1, list2):
    """
    Determine if two lists of the same length match according to rules
    
    Matching rules:
    - 0 can match 0 or 1
    - 1 can only match 1
    
    Args:
        list1: First list
        list2: Second list
        
    Returns:
        True if lists have same length and match according to rules; False otherwise
    """
    # First check if lengths are the same
    if len(list1) != len(list2):
        return False
    
    # Check if elements match at each position
    for a, b in zip(list1, list2):
        # Matching rules: 0 can match 0 or 1, 1 can only match 1
        if a == 1 and b != 1:
            return False
    
    return True


def update_rules(rule_path,data_path,res_path,new_rule,num):
    threshold = 0.35
    
    # Read rules.json once to avoid repeated reading
    try:
        with open(rule_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                rules = []
            else:
                rules = json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError):
        rules = []
    
    # Pass rules parameter to avoid re-reading file inside clean_rules
    matches = clean_rules(new_rule, threshold, rules=rules, rules_file=rule_path) ###rule subset
    if matches and len(matches) > 15:
        import random
        indices = random.sample(range(len(matches)), 15)
        matches = [matches[i] for i in indices]
    
    # Use unified keyword extraction function
    to_test = extract_keywords_from_input(new_rule['keywords'])
    new_label=new_rule['label']
    import random
    
    # Read built index to get actual total number of documents
    from whoosh.index import open_dir
    import os
    
    # Use absolute path uniformly to ensure consistency with index building location
    index_path = './improved_dynamic_index'
    
    # Create indexer instance (only once)
    # Use C++ engine for acceleration (if available), automatically fall back to Whoosh
    indexer = ImprovedDynamicIndexer(index_path, create_new=False, use_cpp_engine=True)
    
    try:
        # Get total number of documents (compatible with C++ engine and Whoosh)
        # Prefer Whoosh (since C++ engine may be empty when create_new=False)
        total_docs = 0
        if indexer.ix is not None:
            # Prefer Whoosh (compatible with existing indexes)
            with indexer.ix.searcher() as searcher:
                total_docs = searcher.doc_count()
                print(f"Total documents in index: {total_docs} (read from Whoosh)")
        elif indexer.use_cpp_engine and indexer.cpp_engine is not None:
            # If Whoosh is unavailable, try to get from C++ engine
            total_docs = indexer.cpp_engine.get_doc_count()
            print(f"Total documents in index: {total_docs} (read from C++ engine)")
        else:
            print("Warning: Unable to get total document count, index may not be initialized")
            return False, []
        
        if total_docs == 0:
            print("=" * 60)
            print("Error: 0 documents in index")
            print("Possible reasons:")
            print("  1. Index not built yet - please run build_index_fast.py first to build index")
            print("  2. Index built with old version code (C++ engine mode not persisted to Whoosh)")
            print("     - Please re-run build_index_fast.py to rebuild index")
            print(f"  3. Index path: {index_path}")
            print("=" * 60)
            return False, []
    except Exception as e:
        print(f"Unable to read index: {e}")
        return False, []
    
    # Randomly sample documents based on actual total document count
    sample_num = min(100000, total_docs)  # Reduce sample size

    # Use inverted index: search each keyword individually, then calculate intersection (AND logic)
    print(f"Starting search for {len(to_test)} keywords (using inverted index AND logic)...")
    
    # to_test is already extracted string list, use directly
    keywords_to_search = [kw.strip() for kw in to_test if kw and str(kw).strip()]
    
    if not keywords_to_search:
        print("Warning: No valid keywords to search")
        return False, []
    
    print(f"Actual keywords to search: {keywords_to_search}")
    
    # Optimization: prefer exact AND match query, much faster than separate searches then intersection
    matched_doc_ids = set()
    
    if len(keywords_to_search) == 1:
        # Single keyword, search directly with sample limit
        keyword = keywords_to_search[0]
        results = indexer.search(keyword, mode="OR", limit=min(sample_num * 2, 100000), k=sample_num)
        
        for result in results:
            try:
                if isinstance(result, dict):
                    doc_id = int(result.get('doc_id', -1))
                elif isinstance(result, (int, str)):
                    doc_id = int(result)
                else:
                    doc_id = int(getattr(result, 'doc_id', -1))
                
                if 0 <= doc_id < sample_num:
                    matched_doc_ids.add(doc_id)
            except (ValueError, TypeError, AttributeError):
                continue
        print(f"Keyword '{keyword}': Found {len(matched_doc_ids)} matches in first {sample_num} documents")
    else:
        # Multiple keywords: prefer exact AND match query (more efficient)
        try:
            results = indexer.search_exact_terms(keywords_to_search, limit=min(sample_num * 2, 100000), k=sample_num)
            
            # Ensure results is not None
            if results is None:
                results = []
            
            for result in results:
                try:
                    if isinstance(result, dict):
                        doc_id = int(result.get('doc_id', -1))
                    elif isinstance(result, (int, str)):
                        doc_id = int(result)
                    else:
                        doc_id = int(getattr(result, 'doc_id', -1))
                    
                    if 0 <= doc_id < sample_num:
                        matched_doc_ids.add(doc_id)
                except (ValueError, TypeError, AttributeError):
                    continue
            
            print(f"Using exact AND match: Found {len(matched_doc_ids)} matches in first {sample_num} documents")
        except Exception as e:
            # Fallback: separate searches with limited scope
            print(f"Exact match search failed, using separate search mode: {e}")
            keyword_doc_sets = []
            
            for keyword in keywords_to_search:
                doc_ids = set()
                # Key optimization: use k parameter to limit search scope to sample, reduce result processing
                # Also reduce limit since only need results within sample range
                results = indexer.search(keyword, mode="OR", limit=min(sample_num * 2, 100000), k=sample_num)
                
                for result in results:
                    try:
                        if isinstance(result, dict):
                            doc_id = int(result.get('doc_id', -1))
                        elif isinstance(result, (int, str)):
                            doc_id = int(result)
                        else:
                            doc_id = int(getattr(result, 'doc_id', -1))
                        
                        if 0 <= doc_id < sample_num:
                            doc_ids.add(doc_id)
                    except (ValueError, TypeError, AttributeError):
                        continue
                
                keyword_doc_sets.append(doc_ids)
                print(f"Keyword '{keyword}': Found {len(doc_ids)} matches in first {sample_num} documents")
            
            # Calculate intersection
            if keyword_doc_sets:
                matched_doc_ids = keyword_doc_sets[0]
                for doc_set in keyword_doc_sets[1:]:
                    matched_doc_ids = matched_doc_ids & doc_set
    
    match_ratio = len(matched_doc_ids) / sample_num if sample_num > 0 else 0
    print(f'Matched documents: {len(matched_doc_ids)}, Total samples: {sample_num}, Match ratio: {match_ratio:.4f}')
    if match_ratio > match_ratio_threshold:
        print(f'New rule match ratio is {match_ratio:.2%} in randomly selected {sample_num} samples, matching range is too large, discarding this rule')
        return False, []

    if len(matches)==0:
        print('No similar rules found, use directly')
        # Add new rule directly (using already read rules)
        new_rule['level'] = 1
        rules.append(new_rule)
        with open(rule_path, 'w', encoding='utf-8') as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
        print("New rule added with level 1")
        return True,[]
    
    # Use unified keyword extraction function
    new_rule_keywords = extract_keywords_from_input(new_rule['keywords'])
    keywords = new_rule_keywords.copy()  # Create copy to avoid modifying original data
    print(f"Found {len(matches)} similar rules:")
    for i, match in enumerate(matches):
        # Use unified keyword extraction function
        match_keywords = extract_keywords_from_input(match['keywords'])
        print(f"  Rule {i}: Original file index {match['rule_index']}, Similarity: {match['similarity']:.3f}, Label: {match['label']}, Keywords: {match_keywords}")
        keywords.extend(match_keywords)
    keywords=list(set(keywords))
    
    # Indexer instance already created above, use directly
    
    # Optimization: query matched documents for each rule directly from inverted index, no need to read document content and check_keywords
    search_k = num  # Only for new rule search scope limit
    print(f"New rule search scope limit: k={search_k} (only search first {num} documents)")
    
    # Read labels from final_label.txt
    labels = []
    try:
        with open('./final_label.txt', 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
    except (FileNotFoundError, IOError) as e:
        print(f"Failed to read final_label.txt: {e}")
        return False, []
    
    # For each similar rule, query matched document IDs directly with inverted index AND query
    # Note: similar rules should also be limited to first search_k documents, consistent with new rule
    column_sets = []  # column_sets[i] is set of document IDs matched by rule i
    rule_label = []
    all_doc_ids_set = set()
    # Single keyword search result cache: avoid repeated search for same keyword (defined in advance for later use)
    single_keyword_cache_new = {}
    print(f"Querying matched documents for {len(matches)} similar rules directly using inverted index (limited to first {search_k} documents)...")
    
    for match in matches:
        # Use unified keyword extraction function
        keyword_list = extract_keywords_from_input(match['keywords'])
        
        rule_label.append(match['label'])
        
        # Use inverted index AND query to get matched document ID set directly
        # For similar rules, also limit to first search_k documents
        limit = 100000
        if len(keyword_list) == 0:
            matched_doc_ids = set()
        elif len(keyword_list) == 1:
            # Single keyword: use OR mode (since only one term), limit to first search_k documents
            results = indexer.search(keyword_list[0], mode="OR", limit=limit, k=search_k)
            matched_doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 'doc_id' in r and 0 <= int(r.get('doc_id', -1)) < search_k}
        else:
            # Multiple keywords: prefer exact AND query, limit to first search_k documents
            try:
                results = indexer.search_exact_terms(keyword_list, limit=limit, k=search_k)
                matched_doc_ids = {int(r.get('doc_id', -1)) for r in results if isinstance(r, dict) and 'doc_id' in r and 0 <= int(r.get('doc_id', -1)) < search_k}
            except Exception:
                # Fallback: separate searches then intersection (with cache), limit to first search_k documents
                keyword_doc_sets = []
                for kw in keyword_list:
                    kw_str = str(kw).strip()
                    # Check single keyword cache (if available)
                    if kw_str in single_keyword_cache_new:
                        # Filter cached results to first search_k documents
                        cached_doc_ids = single_keyword_cache_new[kw_str]
                        doc_ids = {doc_id for doc_id in cached_doc_ids if 0 <= doc_id < search_k}
                    else:
                        results = indexer.search(kw_str, mode="OR", limit=limit, k=search_k)
                        doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 'doc_id' in r and 0 <= int(r.get('doc_id', -1)) < search_k}
                        single_keyword_cache_new[kw_str] = doc_ids  # Cache results (range limited)
                    keyword_doc_sets.append(doc_ids)
                if keyword_doc_sets:
                    matched_doc_ids = keyword_doc_sets[0]
                    for doc_set in keyword_doc_sets[1:]:
                        matched_doc_ids = matched_doc_ids & doc_set
                else:
                    matched_doc_ids = set()
        
        column_sets.append(matched_doc_ids)
        all_doc_ids_set.update(matched_doc_ids)
        print(f"  Rule '{match['label']}': Matched {len(matched_doc_ids)} documents (within first {search_k} documents)")
    
    # Query new rule as well (to_test is already extracted string list)
    new_rule_keywords_list = [kw.strip() for kw in to_test if kw and str(kw).strip()]
    
    print(f"New rule keyword list (for query): {new_rule_keywords_list}")
    
    new_rule_doc_ids_set = set()
    if len(new_rule_keywords_list) > 0:
        # For keywords containing spaces, use search_exact_terms to ensure whole phrase matching
        # Check for keywords with spaces
        has_phrase_keywords = any(' ' in str(kw) for kw in new_rule_keywords_list)
        and_query_success = False  # Flag if AND query succeeded
        
        if has_phrase_keywords or len(new_rule_keywords_list) > 1:
            # Optimization strategy: prefer inverted index AND query (most efficient)
            # 1. If no phrase keywords, use search(query_str, mode="AND") directly - one query, inverted index does AND directly
            # 2. If has phrase keywords, use search_exact_terms or separate searches + intersection
            
            if not has_phrase_keywords:
                # All keywords are single terms, prefer AND query (inverted index does AND directly, most efficient)
                try:
                    # Concatenate keyword list into query string
                    query_str = ' '.join([str(k).strip() for k in new_rule_keywords_list])
                    limit = 100000
                    results = indexer.search(query_str, mode="AND", limit=limit, k=search_k)
                    # k=search_k already filtered at lower level, no need to filter again
                    new_rule_doc_ids_set = {int(r.get('doc_id', -1)) for r in results if isinstance(r, dict) and 'doc_id' in r and int(r.get('doc_id', -1)) >= 0}
                    and_query_success = True  # Mark AND query success, skip subsequent fallback logic
                except Exception as e:
                    # AND query failed, fallback to separate searches + intersection
                    # Possible reasons:
                    # 1. C++ engine not initialized or index empty (cpp_doc_count == 0)
                    # 2. Whoosh query parsing failed (query string contains special characters like parentheses, quotes, operators)
                    # 3. Whoosh index file corrupted or inaccessible
                    # 4. Query string empty or malformed
                    # 5. Field does not exist or schema mismatch
                    print(f"  Warning: AND query failed, falling back to separate searches + intersection")
                    print(f"    Query string: '{query_str}'")
                    print(f"    Error type: {type(e).__name__}")
                    print(f"    Error details: {e}")
                    # Continue to fallback logic below
            
            # If has phrase keywords or AND query failed, use fallback strategy
            if not and_query_success:
                # For large number of keywords (>=4), use separate searches + cache directly
                if len(new_rule_keywords_list) >= 4 or (len(new_rule_keywords_list) >= 3 and not has_phrase_keywords):
                    # Directly use separate searches + cache (faster)
                    # Important: for phrase keywords (containing spaces), use search_exact_terms to ensure whole phrase matching
                    # For single terms, use search method
                    # Optimization strategy: separate searches + incremental intersection + short-circuit optimization
                    keyword_results = []
                    limit = 100000
                    for kw in new_rule_keywords_list:
                        kw_str = str(kw).strip()
                        # Check single keyword cache
                        # Note: cache may contain results for all documents (cached during similar rule queries), need filtering
                        if kw_str in single_keyword_cache_new:
                            cached_doc_ids = single_keyword_cache_new[kw_str]
                            # Filter to search_k range (since new rule only needs first search_k documents)
                            doc_ids = {doc_id for doc_id in cached_doc_ids if 0 <= doc_id < search_k}
                        else:
                            # For phrase keywords (containing spaces), use search_exact_terms to ensure whole phrase matching
                            if ' ' in kw_str:
                                try:
                                    results = indexer.search_exact_terms([kw_str], limit=limit, k=search_k)
                                    # k=search_k already filtered at lower level, no need to filter again
                                    doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 'doc_id' in r and int(r.get('doc_id', -1)) >= 0}
                                except Exception:
                                    # If search_exact_terms fails, fallback to normal search
                                    results = indexer.search(kw_str, mode="OR", limit=limit, k=search_k)
                                    # k=search_k already filtered at lower level, no need to filter again
                                    doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 'doc_id' in r}
                            else:
                                # Single term, use normal search
                                results = indexer.search(kw_str, mode="OR", limit=limit, k=search_k)
                                # k=search_k already filtered at lower level, no need to filter again
                                doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 'doc_id' in r}
                            # Note: cache here is for limited range (k=search_k), different from similar rule query cache (k=None)
                            # But this is correct since new rule only needs testing on first search_k documents
                            single_keyword_cache_new[kw_str] = doc_ids
                        
                        # Short-circuit optimization: if any keyword has empty result set, return empty set directly
                        if len(doc_ids) == 0:
                            new_rule_doc_ids_set = set()
                            break
                        
                        keyword_results.append((len(doc_ids), doc_ids))
                    else:
                        # Only calculate intersection if all keywords have results
                        # Optimization: sort by result set size, intersect from smallest to largest (faster to narrow down search space)
                        keyword_results.sort(key=lambda x: x[0])  # Sort by result set size
                        
                        # Incremental intersection: start with smallest result set, narrow down gradually
                        new_rule_doc_ids_set = keyword_results[0][1]  # Start with smallest result set
                        for _, doc_set in keyword_results[1:]:
                            new_rule_doc_ids_set = new_rule_doc_ids_set & doc_set
                            # Early exit if intersection becomes empty
                            if len(new_rule_doc_ids_set) == 0:
                                break
            elif not and_query_success:
                # For 2-3 keywords with phrase keywords, try search_exact_terms
                # Note: this branch won't execute if AND query succeeded
                try:
                    limit = 100000
                    results = indexer.search_exact_terms(new_rule_keywords_list, limit=limit, k=search_k)
                    all_result_doc_ids = [int(r.get('doc_id', -1)) for r in results if isinstance(r, dict)]
                    
                    # If search_exact_terms returns empty, try separate searches + incremental intersection
                    if len(results) == 0:
                        # Optimization strategy: separate searches + incremental intersection + short-circuit optimization
                        keyword_results = []
                        for kw in new_rule_keywords_list:
                            kw_str = str(kw).strip()
                            # Check single keyword cache
                            # Note: cache may contain results for all documents (cached during similar rule queries), need filtering
                            if kw_str in single_keyword_cache_new:
                                cached_doc_ids = single_keyword_cache_new[kw_str]
                                # Filter to search_k range (since new rule only needs first search_k documents)
                                doc_ids = {doc_id for doc_id in cached_doc_ids if 0 <= doc_id < search_k}
                            else:
                                # For phrase keywords (containing spaces), use search_exact_terms to ensure whole phrase matching
                                if ' ' in kw_str:
                                    try:
                                        results = indexer.search_exact_terms([kw_str], limit=limit, k=search_k)
                                        # k=search_k already filtered at lower level, no need to filter again
                                        doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 'doc_id' in r and int(r.get('doc_id', -1)) >= 0}
                                    except Exception:
                                        # If search_exact_terms fails, fallback to normal search
                                        results = indexer.search(kw_str, mode="OR", limit=limit, k=search_k)
                                        # k=search_k already filtered at lower level, no need to filter again
                                        doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 'doc_id' in r}
                                else:
                                    # Single term, use normal search
                                    results = indexer.search(kw_str, mode="OR", limit=limit, k=search_k)
                                    # k=search_k already filtered at lower level, no need to filter again
                                    doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 'doc_id' in r}
                                # Note: cache here is for limited range (k=search_k), different from similar rule query cache (k=None)
                                # But this is correct since new rule only needs testing on first search_k documents
                                single_keyword_cache_new[kw_str] = doc_ids
                            
                            # Short-circuit optimization: if any keyword has empty result set, return empty set directly
                            if len(doc_ids) == 0:
                                new_rule_doc_ids_set = set()
                                break
                            
                            keyword_results.append((len(doc_ids), doc_ids))
                        else:
                            # Only calculate intersection if all keywords have results
                            # Optimization: sort by result set size, intersect from smallest to largest (faster to narrow down search space)
                            keyword_results.sort(key=lambda x: x[0])  # Sort by result set size
                            
                            # Incremental intersection: start with smallest result set, narrow down gradually
                            new_rule_doc_ids_set = keyword_results[0][1]  # Start with smallest result set
                            for _, doc_set in keyword_results[1:]:
                                new_rule_doc_ids_set = new_rule_doc_ids_set & doc_set
                                # Early exit if intersection becomes empty
                                if len(new_rule_doc_ids_set) == 0:
                                    break
                    else:
                        # search_exact_terms has results, use them
                        # k=search_k already filtered at lower level, no need to filter again
                        new_rule_doc_ids_set = {int(r.get('doc_id', -1)) for r in results if isinstance(r, dict) and 'doc_id' in r and int(r.get('doc_id', -1)) >= 0}
                        if len(all_result_doc_ids) > 0 and len(new_rule_doc_ids_set) == 0:
                            print(f"  Warning: Found {len(all_result_doc_ids)} matches but all filtered out due to doc_id outside search_k={search_k} range!")
                except Exception as e:
                    # Exception handling: use search_exact_terms for phrase keywords, search for single terms
                    # Optimization strategy: separate searches + incremental intersection + short-circuit optimization
                    keyword_results = []
                    limit = 100000
                    for kw in new_rule_keywords_list:
                        kw_str = str(kw).strip()
                        # Check single keyword cache
                        if kw_str in single_keyword_cache_new:
                            doc_ids = single_keyword_cache_new[kw_str]
                        else:
                            # For phrase keywords (containing spaces), use search_exact_terms to ensure whole phrase matching
                            if ' ' in kw_str:
                                try:
                                    results = indexer.search_exact_terms([kw_str], limit=limit, k=search_k)
                                    # k=search_k already filtered at lower level, no need to filter again
                                    doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 'doc_id' in r and int(r.get('doc_id', -1)) >= 0}
                                except Exception:
                                    # If search_exact_terms fails, fallback to normal search
                                    results = indexer.search(kw_str, mode="OR", limit=limit, k=search_k)
                                    # k=search_k already filtered at lower level, no need to filter again
                                    doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 'doc_id' in r}
                            else:
                                # Single term, use normal search
                                results = indexer.search(kw_str, mode="OR", limit=limit, k=search_k)
                                # k=search_k already filtered at lower level, no need to filter again
                                doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 'doc_id' in r}
                            # Note: cache here is for limited range (k=search_k), different from similar rule query cache (k=None)
                            # But this is correct since new rule only needs testing on first search_k documents
                            single_keyword_cache_new[kw_str] = doc_ids
                        
                        # Short-circuit optimization: if any keyword has empty result set, return empty set directly
                        if len(doc_ids) == 0:
                            new_rule_doc_ids_set = set()
                            break
                        
                        keyword_results.append((len(doc_ids), doc_ids))
                    else:
                        # Only calculate intersection if all keywords have results
                        # Optimization: sort by result set size, intersect from smallest to largest (faster to narrow down search space)
                        keyword_results.sort(key=lambda x: x[0])  # Sort by result set size
                        
                        # Incremental intersection: start with smallest result set, narrow down gradually
                        new_rule_doc_ids_set = keyword_results[0][1]  # Start with smallest result set
                        for _, doc_set in keyword_results[1:]:
                            new_rule_doc_ids_set = new_rule_doc_ids_set & doc_set
                            # Early exit if intersection becomes empty
                            if len(new_rule_doc_ids_set) == 0:
                                break
        else:
            # Single keyword without spaces, use normal search (with cache)
            kw_str = str(new_rule_keywords_list[0]).strip()
            # Note: cache may contain results for all documents (cached during similar rule queries), need filtering
            if kw_str in single_keyword_cache_new:
                cached_doc_ids = single_keyword_cache_new[kw_str]
                # Filter to search_k range (since new rule only needs first search_k documents)
                new_rule_doc_ids_set = {doc_id for doc_id in cached_doc_ids if 0 <= doc_id < search_k}
            else:
                limit = 100000
                results = indexer.search(kw_str, mode="OR", limit=limit, k=search_k)
                # k=search_k already filtered at lower level, no need to filter again
                new_rule_doc_ids_set = {int(r['doc_id']) for r in results if isinstance(r, dict) and 'doc_id' in r}
                # Note: cache here is for limited range (k=search_k)
                single_keyword_cache_new[kw_str] = new_rule_doc_ids_set
    else:
        print("Warning: New rule has no valid keywords to search")
    
    print(f"Documents matched by new rule: {len(new_rule_doc_ids_set)}")
    if len(new_rule_doc_ids_set) == 0:
        print("Warning: New rule matched no documents, possible reasons:")
        print("  1. Keywords do not exist in index")
        print("  2. Index data does not match search data")
        print("  3. search_k parameter limited search scope")
    
    all_doc_ids_set.update(new_rule_doc_ids_set)
    # Ensure only include matches within first search_k documents
    all_doc_ids = sorted([doc_id for doc_id in all_doc_ids_set if 0 <= doc_id < search_k])
    print(f"All rules involve {len(all_doc_ids)} documents (limited to first {search_k} documents)")
    
    # Get labels for these documents
    categories = []
    for doc_id in all_doc_ids:
        if 0 <= doc_id < len(labels):
            categories.append(labels[doc_id])
        else:
            categories.append("Unknown|Unknown")
    
    # Build column: convert doc_id sets in column_sets to position markers in all_doc_ids
    column = []
    for matched_doc_ids in column_sets:
        match_list = [1 if doc_id in matched_doc_ids else 0 for doc_id in all_doc_ids]
        column.append(match_list)
    
    # Build new_rule_match
    new_rule_match = [1 if doc_id in new_rule_doc_ids_set else 0 for doc_id in all_doc_ids]
    print(f"new_rule_match length: {len(new_rule_match)}, Matches: {sum(new_rule_match)}")

    # Query result cache: avoid repeated queries for same keyword combinations
    query_cache = {}
    # Single keyword search result cache: avoid repeated searches for same keywords (reuse cache from new rule query)
    single_keyword_cache = single_keyword_cache_new  # Directly reuse cache created above
    
    # Construct match vectors on all_doc_ids based on inverted index
    # Note: logic must be exactly the same as building new_rule_match
    def match_vector_for_keywords(keywords_list):
        if not keywords_list:
            return [0] * len(all_doc_ids)
        
        # Use tuple of keywords as cache key (sorted and deduplicated to ensure order doesn't affect cache)
        cache_key = tuple(sorted(str(k).strip() for k in keywords_list))
        if cache_key in query_cache:
            return query_cache[cache_key]
        
        matched_ids = set()
        # Check for keywords with spaces
        has_phrase_keywords = any(' ' in str(k) for k in keywords_list)
        

        if has_phrase_keywords or len(keywords_list) > 1:
            # Optimization strategy: prefer inverted index AND query (most efficient)
            # 1. If no phrase keywords, use search(query_str, mode="AND") directly - one query, inverted index does AND directly
            # 2. If has phrase keywords, use search_exact_terms or separate searches + intersection
            
            if not has_phrase_keywords:
                # All keywords are single terms, prefer AND query (inverted index does AND directly, most efficient)
                try:
                    # Concatenate keyword list into query string
                    query_str = ' '.join([str(k).strip() for k in keywords_list])
                    limit = 100000
                    results = indexer.search(query_str, mode="AND", limit=limit, k=search_k)
                    matched_ids = {int(r.get('doc_id', -1)) for r in results if isinstance(r, dict) and 0 <= int(r.get('doc_id', -1)) < search_k}
                except Exception as e:
                    # AND query failed, fallback to separate searches + intersection
                    print(f"Warning: AND query failed, falling back to separate searches: {e}")
                    # Continue to fallback logic below
                    has_phrase_keywords = True  # Trigger fallback logic
            
            # If has phrase keywords or AND query failed, use fallback strategy
            if has_phrase_keywords:
                # For large number of keywords (>=4), use separate searches + cache directly
                if len(keywords_list) >= 4 or (len(keywords_list) >= 3 and not has_phrase_keywords):
                    # Optimization strategy: separate searches + incremental intersection + short-circuit optimization
                    # Important: for phrase keywords (containing spaces), use search_exact_terms to ensure whole phrase matching
                    # For single terms, use search method
                    limit = 100000
                    
                    # Step 1: Search all keywords (prefer cache)
                    keyword_results = []
                    for kw in keywords_list:
                        kw_str = str(kw).strip()
                        # Check single keyword cache
                        if kw_str in single_keyword_cache:
                            doc_ids = single_keyword_cache[kw_str]
                        else:
                            # For phrase keywords (containing spaces), use search_exact_terms to ensure whole phrase matching
                            if ' ' in kw_str:
                                try:
                                    results = indexer.search_exact_terms([kw_str], limit=limit, k=search_k)
                                    doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 0 <= int(r['doc_id']) < search_k}
                                except Exception:
                                    # If search_exact_terms fails, fallback to normal search
                                    results = indexer.search(kw_str, mode="OR", limit=limit, k=search_k)
                                    doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 0 <= int(r['doc_id']) < search_k}
                            else:
                                # Single term, use normal search
                                results = indexer.search(kw_str, mode="OR", limit=limit, k=search_k)
                                doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 0 <= int(r['doc_id']) < search_k}
                            single_keyword_cache[kw_str] = doc_ids
                        
                        # Short-circuit optimization: if any keyword has empty result set, return empty set directly
                        if len(doc_ids) == 0:
                            matched_ids = set()
                            break
                        
                        keyword_results.append((len(doc_ids), doc_ids))
                    else:
                        # Only calculate intersection if all keywords have results
                        # Optimization: sort by result set size, intersect from smallest to largest (faster to narrow down search space)
                        keyword_results.sort(key=lambda x: x[0])  # Sort by result set size
                        
                        # Incremental intersection: start with smallest result set, narrow down gradually
                        matched_ids = keyword_results[0][1]  # Start with smallest result set
                        for _, doc_set in keyword_results[1:]:
                            matched_ids = matched_ids & doc_set
                            # Early exit if intersection becomes empty
                            if len(matched_ids) == 0:
                                break
                else:
                    # For 2-3 keywords (or with phrase keywords), try search_exact_terms
                    try:
                        limit = 100000
                        results = indexer.search_exact_terms([str(k) for k in keywords_list], limit=limit, k=search_k)
                        all_result_doc_ids = [int(r.get('doc_id', -1)) for r in results if isinstance(r, dict)]
                        
                        # If search_exact_terms returns empty, try separate searches + incremental intersection
                        # For phrase keywords (containing spaces), use search_exact_terms to ensure whole phrase matching
                        # For single terms, use search method
                        if len(results) == 0:
                            # Optimization strategy: separate searches + incremental intersection + short-circuit optimization
                            keyword_results = []
                            limit = 100000
                            for kw in keywords_list:
                                kw_str = str(kw).strip()
                                # Check single keyword cache
                                if kw_str in single_keyword_cache:
                                    doc_ids = single_keyword_cache[kw_str]
                                else:
                                    # For phrase keywords (containing spaces), use search_exact_terms to ensure whole phrase matching
                                    if ' ' in kw_str:
                                        try:
                                            results = indexer.search_exact_terms([kw_str], limit=limit, k=search_k)
                                            doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 0 <= int(r['doc_id']) < search_k}
                                        except Exception:
                                            # If search_exact_terms fails, fallback to normal search
                                            results = indexer.search(kw_str, mode="OR", limit=limit, k=search_k)
                                            doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 0 <= int(r['doc_id']) < search_k}
                                    else:
                                        # Single term, use normal search
                                        results = indexer.search(kw_str, mode="OR", limit=limit, k=search_k)
                                        doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 0 <= int(r['doc_id']) < search_k}
                                    single_keyword_cache[kw_str] = doc_ids
                                
                                # Short-circuit optimization: if any keyword has empty result set, return empty set directly
                                if len(doc_ids) == 0:
                                    matched_ids = set()
                                    break
                                
                                keyword_results.append((len(doc_ids), doc_ids))
                            else:
                                # Only calculate intersection if all keywords have results
                                # Optimization: sort by result set size, intersect from smallest to largest (faster to narrow down search space)
                                keyword_results.sort(key=lambda x: x[0])  # Sort by result set size
                                
                                # Incremental intersection: start with smallest result set, narrow down gradually
                                matched_ids = keyword_results[0][1]  # Start with smallest result set
                                for _, doc_set in keyword_results[1:]:
                                    matched_ids = matched_ids & doc_set
                                    # Early exit if intersection becomes empty
                                    if len(matched_ids) == 0:
                                        break
                        else:
                            # search_exact_terms has results, use them
                            matched_ids = {int(r.get('doc_id', -1)) for r in results if isinstance(r, dict) and 0 <= int(r.get('doc_id', -1)) < search_k}
                            if len(all_result_doc_ids) > 0 and len(matched_ids) == 0:
                                print(f"Warning: Found {len(all_result_doc_ids)} matches but all filtered out due to doc_id outside search_k={search_k} range!")
                    except Exception as e:
                        # Exception handling: use search_exact_terms for phrase keywords, search for single terms
                        # Optimization strategy: separate searches + incremental intersection + short-circuit optimization
                        keyword_results = []
                        limit = 100000
                        for kw in keywords_list:
                            kw_str = str(kw).strip()
                            # Check single keyword cache
                            if kw_str in single_keyword_cache:
                                doc_ids = single_keyword_cache[kw_str]
                            else:
                                # For phrase keywords (containing spaces), use search_exact_terms to ensure whole phrase matching
                                if ' ' in kw_str:
                                    try:
                                        results = indexer.search_exact_terms([kw_str], limit=limit, k=search_k)
                                        doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 0 <= int(r['doc_id']) < search_k}
                                    except Exception:
                                        # If search_exact_terms fails, fallback to normal search
                                        results = indexer.search(kw_str, mode="OR", limit=limit, k=search_k)
                                        doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 0 <= int(r['doc_id']) < search_k}
                                else:
                                    # Single term, use normal search
                                    results = indexer.search(kw_str, mode="OR", limit=limit, k=search_k)
                                    doc_ids = {int(r['doc_id']) for r in results if isinstance(r, dict) and 0 <= int(r['doc_id']) < search_k}
                                single_keyword_cache[kw_str] = doc_ids
                            
                            # Short-circuit optimization: if any keyword has empty result set, return empty set directly
                            if len(doc_ids) == 0:
                                matched_ids = set()
                                break
                            
                            keyword_results.append((len(doc_ids), doc_ids))
                        else:
                            # Only calculate intersection if all keywords have results
                            # Optimization: sort by result set size, intersect from smallest to largest (faster to narrow down search space)
                            keyword_results.sort(key=lambda x: x[0])  # Sort by result set size
                            
                            # Incremental intersection: start with smallest result set, narrow down gradually
                            matched_ids = keyword_results[0][1]  # Start with smallest result set
                            for _, doc_set in keyword_results[1:]:
                                matched_ids = matched_ids & doc_set
                                # Early exit if intersection becomes empty
                                if len(matched_ids) == 0:
                                    break
        else:
            # Single keyword without spaces, use normal search (with cache)
            kw_str = str(keywords_list[0]).strip()
            if kw_str in single_keyword_cache:
                matched_ids = single_keyword_cache[kw_str]
            else:
                limit = 100000
                results = indexer.search(kw_str, mode="OR", limit=limit, k=search_k)
                matched_ids = {int(r['doc_id']) for r in results if 0 <= int(r['doc_id']) < search_k}
                single_keyword_cache[kw_str] = matched_ids
        
        result = [1 if doc_id in matched_ids else 0 for doc_id in all_doc_ids]
        # Cache results
        query_cache[cache_key] = result
        return result
    # Above using inverted index to get categories, column, new_rule_match

    remove_rules_index=[]
    print(f"Initializing remove_rules_index: {remove_rules_index}")
    print(f"Starting error rate check for {len(column)} rules...")
    for j in range(len(column)):
        rule=column[j]
        rule_label1=rule_label[j]
        mat=0
        wrong=0
        wrong_cases = []  # Record specific error cases
        for i in range(len(rule)):
            if rule[i]==1:
                mat+=1
                # Exclude "processing error" cases from error statistics
                if rule_label1!=categories[i] and categories[i] != "Processing Error":
                    wrong+=1
                    wrong_cases.append(f"Document {i}: Rule label '{rule_label1}' != Actual label '{categories[i]}'")
        # Print statistics for each rule
        if mat > 0:
            error_rate = wrong / mat
            print(f"  Rule {j} (Label: {rule_label1}): Matches={mat}, Errors={wrong}, Error rate={error_rate:.2f}")
            if error_rate > 0.2:
                remove_rules_index.append(j)
                print(f"    ✓ Error rate {error_rate:.2f} > 0.2, added to remove_rules_index")
        else:
            print(f"  Rule {j} (Label: {rule_label1}): Matches=0, skipped")
    remove_rules_index=list(set(remove_rules_index))
    print(f"Deduplicated remove_rules_index: {remove_rules_index}")

    if not matches:
        top_level_indices = []
    else:
        # Find highest level
        max_level = min([m.get('level', 999) for m in matches])
        # Collect all indices (in matches) with level equal to highest level
        top_level_indices = [i for i, m in enumerate(matches) if m.get('level', 999) == max_level]
    print("List of highest level indices in matches:", top_level_indices)
    print(f"to_test (new rule keywords): {to_test}")
    # new_rule_match built using inverted index
    print(f"new_rule_match length: {len(new_rule_match)}, Matches: {sum(new_rule_match)}")
    mat=0
    wrong=0
    wrong_cases = []  # Record specific error cases for new rule
    for i in range(len(new_rule_match)):
        if new_rule_match[i]==1:
            mat+=1
            # Exclude "processing error" cases from error statistics
            if categories[i]!=new_label and categories[i] != "Processing Error":
                wrong+=1
                wrong_cases.append(f"Document {i}: New rule label '{new_label}' != Actual label '{categories[i]}'")
    if wrong>0.2*mat:     ##If error rate > 0.2
        print('New rule has too many errors, cannot be used, need to delete other rules in list and new rule')
        print(f"  New rule label: {new_label}")
        print(f"  Total matches: {mat}, Errors: {wrong}")
        print(f"  Error rate: {wrong/mat:.2f} > 0.2")
        print(f"  Error details:")
        for case in wrong_cases[:5]:  # Only show first 5 error cases
            print(f"    {case}")
        if len(wrong_cases) > 5:
            print(f"    ... {len(wrong_cases) - 5} more error cases")
        print(f"Current remove_rules_index: {remove_rules_index}")
        return False,remove_rules_index


    ###If generalization ability is weak
    print('new_rule match status:',new_rule_match)
    new_key=copy.deepcopy(to_test)
    updated_candidate=list(set(top_level_indices)-set(remove_rules_index))
    
    # First loop: check if new_rule_match is subset of column[i]
    for i in updated_candidate:
        if rule_label[i]==new_label:
            # Call is_matched only once to avoid repeated calls
            is_subset = is_matched(new_rule_match,column[i])
            if is_subset:
                # Use unified keyword extraction function
                match_keywords = extract_keywords_from_input(matches[i]['keywords'])
                updated_key=list(set(new_key)&set(match_keywords))
                # Construct match vector on all_doc_ids using inverted index (with cache)
                temp = match_vector_for_keywords(updated_key)
                if temp and lists_equal_fast(temp, column[i]):
                    new_key=updated_key

    print(f"\nFinal new_key: {new_key}")
    # Recalculate new_rule_match using inverted index
   
    # Second loop: check if column[i] is subset of new_rule_match
    for i in updated_candidate:
        if rule_label[i]==new_label:
            # Call is_matched only once to avoid repeated calls
            is_subset = is_matched(column[i],new_rule_match)
            if is_subset:   #Previous is subset
                # Use unified keyword extraction function
                match_keywords = extract_keywords_from_input(matches[i]['keywords'])
                updated_key=list(set(new_key)&set(match_keywords))
                # Construct match vector on all_doc_ids using inverted index (with cache)
                temp = match_vector_for_keywords(updated_key)
                if temp and lists_equal_fast(temp, new_rule_match):
                    new_key=updated_key
                    remove_rules_index.append(i)
               
    new_rule['keywords']=new_key 
    print('new_key--------',new_key)
    print('remove_rules_index:',remove_rules_index)

    # Check if generalized keyword count is at least 2
    # Use unified keyword extraction function to get keyword list
    final_keywords = extract_keywords_from_input(new_key)
    valid_keywords = [kw for kw in final_keywords if kw and str(kw).strip()]
    
    if len(valid_keywords) < 2:
        print(f"Generalized rule has fewer than 2 keywords ({len(valid_keywords)}), rejecting this rule")
        print(f"Keyword list: {valid_keywords}")
        return False, remove_rules_index

    new_rule['level'] = 1
    # Use already read rules to avoid repeated file reading

    # sample_num = min(10000, total_docs)  # Reduce sample size
    # # Batch search optimization: search all keywords at once, then filter samples
    # print(f"Starting batch search for {len(new_key)} keywords...")
    # all_search_results = []
    # for word in new_key:
    #     results = indexer.search(word, mode="AND", k=sample_num)
    #     all_search_results.extend(results)
    
    # # Count matches based on search results
    # matched_doc_ids = set()
    # for result in all_search_results:
    #     matched_doc_ids.add(int(result['doc_id']))
    # match_ratio = len(matched_doc_ids) / sample_num
    # print('len(matched_doc_ids):',len(matched_doc_ids),';match_ratio',match_ratio,';sample number',sample_num)
    # if match_ratio > 0.07:
    #     print(f'New rule match ratio is {match_ratio:.2%} in randomly selected {sample_num} samples, matching range is too large, discarding this rule')
    #     return False, []
    # else:
    #     return False,remove_rules_index
    rules.append(new_rule)
    with open(rule_path, 'w', encoding='utf-8') as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    print("Updated new rule added with level 1")
          
    return True,remove_rules_index

def keep_rules_from_file(rules_file, matches, remove_indices):
    """
    Keep rules not marked for deletion in rules.json file
    
    Args:
        rules_file: Path to rule file
        matches: List of matched rules containing rule_index field
        remove_indices: List of indices (in matches) of rules to delete
    """
    import json
    
    # Read rule file
    try:
        with open(rules_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                rules = []
            else:
                rules = json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError):
        rules = []
    
    # Get original file indices of rules to delete
    original_indices_to_remove = set()
    for idx in remove_indices:
        if 0 <= idx < len(matches):
            original_idx = matches[idx]['rule_index']
            original_indices_to_remove.add(original_idx)
    
    # Create new rule list keeping only non-deleted rules
    kept_rules = []
    for i, rule in enumerate(rules):
        if i not in original_indices_to_remove:
            kept_rules.append(rule)
    
    print(f"Keeping {len(kept_rules)} rules, deleting {len(original_indices_to_remove)} rules:")
    for idx in original_indices_to_remove:
        if 0 <= idx < len(rules):
            rule = rules[idx]
            print(f"  Deleting rule {idx}: Label={rule.get('label', 'Unknown')}, Keywords={rule.get('keywords', [])}")
    
    # Write back to file
    with open(rules_file, 'w', encoding='utf-8') as f:
        json.dump(kept_rules, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully kept {len(kept_rules)} rules")

    
if __name__ == "__main__":
    new_rule={'label': 'Network Infrastructure|Perimeter Security Devices', 'keywords':['firewall', 'Check Point Firewall', 'firewall_host', 'smartcenter_host']}
    rule_path='./rules.json'
    # Get matches before calling update_rules since update_rules modifies rules.json
    threshold = 0.15
    matches_before = clean_rules(new_rule, threshold, rules_file=rule_path)
    
    keep,rm=update_rules('./rules.json','/home/zhangwj/res/SonicWall_search_results.json','./res_result.json',new_rule,140)
    
    # Delete rules if needed, using matches before call
    if rm:
        print(f"\nNeed to delete {len(rm)} rules")
        keep_rules_from_file(rule_path, matches_before, rm)