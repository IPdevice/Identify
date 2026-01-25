import json
import os
from collections import defaultdict
import pprint
import re

class DeviceClassificationAnalyzer:
    def __init__(self, file_path, is_line_delimited_json=False, auto_fix=True):
        """Initialize analyzer and load JSON data
        :param file_path: Path to data file
        :param is_line_delimited_json: Whether the file is in line-delimited JSON format (one JSON object per line)
        :param auto_fix: Whether to automatically attempt to fix simple JSON format errors
        """
        self.file_path = file_path
        self.is_line_delimited_json = is_line_delimited_json
        self.auto_fix = auto_fix
        self.data = self.load_data()
        self.consolidated_data = self.consolidate_data()
    
    def _extract_top_level_json_objects(self, content):
        """Extract top-level JSON object strings from arbitrary text, ignoring brackets inside quotes.
        Suitable for array files containing noise or mixed text, returns list of object strings.
        """
        objects = []
        brace_level = 0
        in_string = False
        escape = False
        start_idx = None
        for idx, ch in enumerate(content):
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            else:
                if ch == '"':
                    in_string = True
                    continue
                if ch == '{':
                    if brace_level == 0:
                        start_idx = idx
                    brace_level += 1
                elif ch == '}':
                    if brace_level > 0:
                        brace_level -= 1
                        if brace_level == 0 and start_idx is not None:
                            obj_str = content[start_idx:idx+1]
                            objects.append(obj_str)
                            start_idx = None
        return objects

    def _attempt_fix_json(self, json_str):
        """Attempt to fix common JSON format errors"""
        if not json_str:
            return None
            
        # Attempt to fix common issues
        fixed = json_str.strip()
        
        # Remove possible trailing commas
        if fixed.endswith(','):
            fixed = fixed[:-1]
            
        # Check brace matching
        open_braces = fixed.count('{')
        close_braces = fixed.count('}')
        if open_braces > close_braces:
            fixed += '}' * (open_braces - close_braces)
        elif close_braces > open_braces:
            fixed = '{' * (close_braces - open_braces) + fixed
            
        # Check bracket matching
        open_brackets = fixed.count('[')
        close_brackets = fixed.count(']')
        if open_brackets > close_brackets:
            fixed += ']' * (open_brackets - close_brackets)
        elif close_brackets > open_brackets:
            fixed = '[' * (close_brackets - open_brackets) + fixed
            
        return fixed
    
    def _attempt_fix_rule_json(self, rule_json_str):
        """Fix common format issues in the "generated_rules" field and return the fixed string"""
        if not rule_json_str:
            return rule_json_str
        fixed = rule_json_str
        
        # 1) Fix cases like: "keyword": "port": 2404  =>  "keyword": "port: 2404"
        fixed = re.sub(r'("关键词"\s*:\s*")([^"]+)"\s*:\s*(\d+)', r'\1\2: \3"', fixed)
        
        # 2) Fix cases like: "keyword": "port": "2404" => "keyword": "port: 2404"
        fixed = re.sub(r'("关键词"\s*:\s*")([^"]+)"\s*:\s*"(\d+)"', r'\1\2: \3"', fixed)
        
        # 3) Remove trailing commas in objects or arrays
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        
        # 4) Balance braces/brackets count (conservative fix)
        open_braces = fixed.count('{')
        close_braces = fixed.count('}')
        if open_braces > close_braces:
            fixed += '}' * (open_braces - close_braces)
        open_brackets = fixed.count('[')
        close_brackets = fixed.count(']')
        if open_brackets > close_brackets:
            fixed += ']' * (open_brackets - close_brackets)
        
        return fixed
        
    def load_data(self):
        """Load JSON data, support regular JSON and line-delimited JSON formats, add error repair attempts"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                stripped = content.lstrip()

                # First try to parse the entire content (auto detect array or object JSON)
                try:
                    parsed_all = json.loads(content)
                    # If it's a list, filter out dictionary elements
                    if isinstance(parsed_all, list):
                        return [item for item in parsed_all if isinstance(item, dict)]
                    # If it's a single dictionary, return it wrapped in a list
                    if isinstance(parsed_all, dict):
                        return [parsed_all]
                except Exception:
                    # If parsing entire content fails, try to extract top-level objects from full text
                    extracted = self._extract_top_level_json_objects(content)
                    extracted_objs = []
                    for i, obj_str in enumerate(extracted, 1):
                        try:
                            obj = json.loads(obj_str)
                            if isinstance(obj, dict):
                                extracted_objs.append(obj)
                        except Exception:
                            if self.auto_fix:
                                try:
                                    fixed_obj_str = self._attempt_fix_json(obj_str)
                                    obj = json.loads(fixed_obj_str)
                                    if isinstance(obj, dict):
                                        extracted_objs.append(obj)
                                except Exception:
                                    pass
                    if extracted_objs:
                        return extracted_objs
                    # If declared as line-delimited JSON, or still failed, parse line by line

                if self.is_line_delimited_json or stripped[:1] not in ('[', '{'):
                    # Process line-delimited JSON format (one JSON object per line)
                    fixed = []
                    for line_num, line in enumerate(content.splitlines(), 1):
                        line = line.strip()
                        if line:
                            try:
                                # First try direct parsing
                                parsed = json.loads(line)
                                # Ensure parsing result is a dictionary, extract first element if it's a list
                                if isinstance(parsed, list) and len(parsed) > 0:
                                    parsed = parsed[0]
                                if isinstance(parsed, dict):
                                    fixed.append(parsed)
                                else:
                                    print(f"Warning: Parsing result of line {line_num} is not a dictionary, skipped")
                            except Exception as e:
                                # If auto-fix is enabled, try to fix and parse again
                                if self.auto_fix:
                                    try:
                                        fixed_line = self._attempt_fix_json(line)
                                        if fixed_line:
                                            parsed = json.loads(fixed_line)
                                            # Ensure fixed result is a dictionary
                                            if isinstance(parsed, list) and len(parsed) > 0:
                                                parsed = parsed[0]
                                            if isinstance(parsed, dict):
                                                fixed.append(parsed)
                                                print(f"Fixed JSON format error in line {line_num}")
                                                continue
                                            else:
                                                print(f"Warning: Fixed result of line {line_num} is not a dictionary, skipped")
                                    except Exception:
                                        pass  # Fix failed, continue processing
                                
                                # Cannot fix, record error and skip
                                print(f"Warning: Skipping invalid JSON in line {line_num}: {str(e)}")
                                print(f"Content preview: {line[:50]}...")
                    return fixed

                # If not line-delimited and full parsing failed, return empty list
                return []
                    
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found")
            return []
        except json.JSONDecodeError:
            print(f"Error: File {self.file_path} is not a valid JSON format")
            return []
        except Exception as e:
            print(f"Error occurred while loading data: {str(e)}")
            return []
    
    def consolidate_data(self):
        """Consolidate data, handle possible nested structures"""
        consolidated = []
        # Check if data is a nested list
        if isinstance(self.data, list) and all(isinstance(item, list) for item in self.data):
            for sublist in self.data:
                # Only add dictionary-type elements
                for item in sublist:
                    if isinstance(item, dict):
                        consolidated.append(item)
                    else:
                        print(f"Warning: Non-dictionary type data found, skipped")
        elif isinstance(self.data, list):
            # Filter out non-dictionary type elements
            for item in self.data:
                if isinstance(item, dict):
                    consolidated.append(item)
                else:
                    print(f"Warning: Non-dictionary type data found, skipped")
        return consolidated
    
    def get_category_distribution(self):
        """Get main category distribution statistics"""
        distribution = defaultdict(int)
        for item in self.consolidated_data:
            if '最终结果' in item and '大类' in item['最终结果']:
                category = item['最终结果']['大类']
                distribution[category] += 1
        return dict(distribution)
    
    def get_subcategory_distribution(self):
        """Get subcategory distribution statistics"""
        distribution = defaultdict(lambda: defaultdict(int))
        for item in self.consolidated_data:
            if '最终结果' in item and '大类' in item['最终结果'] and '小类' in item['最终结果']:
                category = item['最终结果']['大类']
                subcategory = item['最终结果']['小类']
                
                # Handle case where subcategory might be a dictionary
                if isinstance(subcategory, dict) and '小类' in subcategory:
                    subcat_name = subcategory['小类']
                else:
                    subcat_name = subcategory
                
                distribution[category][subcat_name] += 1
        
        # Convert to regular dictionary
        return {k: dict(v) for k, v in distribution.items()}
    
    def extract_rules_by_category(self, category=None, subcategory=None):
        """Extract rules by category"""
        rules = defaultdict(list)
        
        for item in self.consolidated_data:
            if '最终结果' not in item or '生成的规则' not in item:
                continue
                
            # Get category information of current item
            current_category = item['最终结果'].get('大类')
            current_subcategory = item['最终结果'].get('小类')
            
            # Handle case where subcategory might be a dictionary
            if isinstance(current_subcategory, dict) and '小类' in current_subcategory:
                current_subcat_name = current_subcategory['小类']
            else:
                current_subcat_name = current_subcategory
            
            # Filter by category
            if category and current_category != category:
                continue
            if subcategory and current_subcat_name != subcategory:
                continue
            
            # Parse rules
            try:
                rule_data = json.loads(item['生成的规则'])
                rules_key = f"{current_category}|{current_subcat_name}"
                rules[rules_key].append(rule_data)
            except json.JSONDecodeError:
                if self.auto_fix:
                    try:
                        fixed_rule = self._attempt_fix_rule_json(item['生成的规则'])
                        rule_data = json.loads(fixed_rule)
                        rules_key = f"{current_category}|{current_subcat_name}"
                        rules[rules_key].append(rule_data)
                    except Exception:
                        print(f"Warning: Invalid rule format for member {item.get('member_id')}, preview: {str(item.get('生成的规则'))[:60]}...")
                        continue
        
        return dict(rules)
    
    def get_member_distribution(self):
        """Get member ID distribution"""
        member_clusters = defaultdict(set)
        for item in self.consolidated_data:
            # Ensure item is a dictionary type
            if not isinstance(item, dict):
                continue
                
            member_id = item.get('member_id')
            cluster_id = item.get('cluster_id')
            if member_id and cluster_id is not None:
                member_clusters[member_id].add(cluster_id)
        
        # Convert to number of clusters each member participated in
        return {k: len(v) for k, v in member_clusters.items()}
    
    def analyze_keyword_frequency(self, top_n=10):
        """Analyze keyword frequency in rules by category
        :param top_n: Return top N most frequent keywords for each category
        :return: Keyword frequency statistics organized by category
        """
        # Structure: {category: {field_source: {keyword: occurrence_count}}}
        keyword_freq = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Get all rules
        all_rules = self.extract_rules_by_category()
        
        for category, rules in all_rules.items():
            for rule in rules:
                if '关键词' in rule:
                    for keyword_info in rule['关键词']:
                        field_source = keyword_info.get('字段来源', 'Unknown Source')
                        keyword = keyword_info.get('关键词', '')
                        if keyword:
                            keyword_freq[category][field_source][keyword] += 1
        
        # Process results, keep only top N frequent keywords for each category
        result = {}
        for category, field_sources in keyword_freq.items():
            category_result = {}
            for field_source, keywords in field_sources.items():
                # Sort by occurrence count and take top N
                sorted_keywords = sorted(
                    keywords.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:top_n]
                category_result[field_source] = dict(sorted_keywords)
            result[category] = category_result
            
        return result
    
    def extract_flat_rules(self):
        """Extract flat rules for each record: only label (main_category|subcategory) and keyword string list.
        Always return the same number of entries as input records, even if keywords cannot be parsed (return empty list).
        Structure: [{"Label": "main_category|subcategory", "Keywords": [str, ...]}, ...]
        """
        flat_rules = []
        for item in self.consolidated_data:
            if not isinstance(item, dict):
                continue
            label = None
            
            # First try to get label from "最终结果.类别" field
            if '最终结果' in item and isinstance(item['最终结果'], dict):
                category = item['最终结果'].get('类别')
                if isinstance(category, str) and category:
                    label = category
                else:
                    # If "类别" field does not exist, try to combine from "大类" and "小类" fields
                    current_category = item['最终结果'].get('大类')
                    current_subcategory = item['最终结果'].get('小类')
                    if isinstance(current_subcategory, dict) and '小类' in current_subcategory:
                        current_subcat_name = current_subcategory['小类']
                    else:
                        current_subcat_name = current_subcategory
                    if current_category and current_subcat_name:
                        label = f"{current_category}|{current_subcat_name}"
            
            # If still no label, use default value
            if not label:
                label = "Unknown Category|Unknown Subcategory"
            
            # Remove parentheses and their contents from label
            label = re.sub(r'[（(][^）)]*[）)]', '', label)
            keywords_only = []
            rule_raw = item.get('生成的规则')
            if isinstance(rule_raw, (str, dict)):
                # Allow string JSON or already parsed object
                try:
                    rule_obj = rule_raw if isinstance(rule_raw, dict) else json.loads(rule_raw)
                except Exception:
                    rule_obj = None
                    if self.auto_fix and isinstance(rule_raw, str):
                        try:
                            fixed_rule = self._attempt_fix_rule_json(rule_raw)
                            rule_obj = json.loads(fixed_rule)
                        except Exception:
                            rule_obj = None
                if isinstance(rule_obj, dict):
                    kws = rule_obj.get('关键词')
                    if isinstance(kws, list):
                        for kw in kws:
                            if isinstance(kw, dict):
                                val = kw.get('关键词')
                                if isinstance(val, str) and val:
                                    keywords_only.append(val)
                            elif isinstance(kw, str) and kw:
                                keywords_only.append(kw)
            flat_rules.append({
                'Label': label,
                'Keywords': keywords_only,
            })
        return flat_rules
    
    def deduplicate_rules(self, flat_rules):
        """Deduplicate flat rules, merge rules with the same label and keywords
        :param flat_rules: List of flat rules
        :return: Deduplicated list of rules
        """
        # Use dictionary to store deduplicated rules, key is (label, keyword_tuple)
        unique_rules = {}
        
        for rule in flat_rules:
            label = rule.get('Label', '')
            keywords = rule.get('Keywords', [])
            
            # Skip this rule if keyword list is empty
            if not keywords:
                continue
            
            # Sort keywords to ensure consistency
            keywords_sorted = sorted(keywords)
            
            # Create unique key
            key = (label, tuple(keywords_sorted))
            
            # Add rule if it doesn't exist
            if key not in unique_rules:
                unique_rules[key] = {
                    'Label': label,
                    'Keywords': keywords_sorted
                }
        
        # Convert to list and return
        return list(unique_rules.values())
    
    def analyze(self, output_file=None):
        """Perform complete analysis and print results, can specify output file"""
        # If output file is specified, write results to file; downgrade to console if no permission
        output = None
        if output_file:
            try:
                output = open(output_file, 'w', encoding='utf-8')
            except PermissionError:
                print(f"Warning: Cannot write to {output_file}, will print to console instead")
            except Exception as e:
                print(f"Warning: Failed to open output file ({e}), will print to console instead")
        
        def print_line(text=""):
            """Determine output location based on whether output file is specified"""
            if output:
                output.write(text + "\n")
            else:
                print(text)
        
        print_line("===== Device Classification Analysis Report =====")
        print_line(f"Total records: {len(self.consolidated_data)}\n")
        
        # Main category distribution
        print_line("1. Main Category Distribution:")
        category_dist = self.get_category_distribution()
        for cat, count in category_dist.items():
            percentage = (count / len(self.consolidated_data)) * 100
            print_line(f"   {cat}: {count} records ({percentage:.1f}%)")
        
        # Subcategory distribution
        print_line("\n2. Subcategory Distribution:")
        subcat_dist = self.get_subcategory_distribution()
        for cat, subcats in subcat_dist.items():
            print_line(f"   {cat}:")
            for subcat, count in subcats.items():
                percentage = (count / sum(subcats.values())) * 100
                print_line(f"      {subcat}: {count} records ({percentage:.1f}%)")
        
        # Member participation
        print_line("\n3. Member Cluster Participation Distribution:")
        member_dist = self.get_member_distribution()
        member_counts = defaultdict(int)
        for count in member_dist.values():
            member_counts[count] += 1
        for clusters, members in member_counts.items():
            print_line(f"   Members participating in {clusters} clusters: {members} people")
        
        # Rule extraction examples
        print_line("\n4. Rule Extraction Examples (first 2):")
        all_rules = self.extract_rules_by_category()
        sample_rules = []
        for cat_rules in all_rules.values():
            sample_rules.extend(cat_rules[:1])  # Take 1 from each category
            if len(sample_rules) >= 2:
                break
        
        for i, rule in enumerate(sample_rules[:2]):
            print_line(f"   Rule {i+1}:")
            if output:
                # Use pprint string output when writing to file
                output.write(pprint.pformat(rule, indent=6) + "\n")
            else:
                pprint.pprint(rule, indent=6)
        
        # Keyword frequency statistics
        print_line("\n5. Keyword Frequency Statistics (by category and field source):")
        keyword_freq = self.analyze_keyword_frequency(top_n=5)
        for category, field_sources in keyword_freq.items():
            print_line(f"   Category: {category}")
            for field_source, keywords in field_sources.items():
                print_line(f"      Field Source: {field_source}")
                for keyword, count in keywords.items():
                    print_line(f"         {keyword}: {count} times")
        
        # Close file
        if output:
            output.close()

if __name__ == "__main__":
    # Define file path (resolved based on script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path ='/home/zhangwj/my_model/res_result.json'
    
    # Create analyzer instance with auto-fix enabled
    analyzer = DeviceClassificationAnalyzer(
        file_path, 
        is_line_delimited_json=True,
        auto_fix=True  # Enable auto-fix
    )
    
   
    try:
        flat_rules = analyzer.extract_flat_rules()
        original_len = len(flat_rules)
        print(f"\nOriginal number of rules: {original_len}")
        
        # Do not filter rules with empty keywords, use all rules directly
        filtered_rules = [rule for rule in flat_rules if rule.get('Keywords')]
        filtered_len = len(filtered_rules)
        print(f"Number of rules after filtering empty keywords: {filtered_len}")
        
        export_data = filtered_rules
        export_path = os.path.join(script_dir, "./rules.json")
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"\nExported {filtered_len} filtered rules to: {export_path}")
        except PermissionError:
            # Fallback to HOME directory
            home_fallback = os.path.expanduser("./rules.json")
            with open(home_fallback, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"\nExported {filtered_len} filtered rules to: {home_fallback}")
    except Exception as e:
        print(f"\nFailed to export flat rules: {e}")