"""
Fast Indexer Wrapper - C++ Optimized Version
Maintains the same interface as ImprovedDynamicIndexer in the original index.py
"""

import json
import os
import shutil
from collections import defaultdict
from whoosh import index
from whoosh.analysis import StandardAnalyzer, RegexTokenizer, LowercaseFilter
from whoosh.fields import Schema, TEXT, ID, KEYWORD, STORED
from whoosh.query import Term, And
from whoosh.qparser import MultifieldParser, OrGroup, AndGroup
from tqdm import tqdm

# Attempt to import C++ optimized version, fall back to Python version if failed
try:
    import fast_indexer
    USE_CPP = True
except ImportError:
    USE_CPP = False
    print("Warning: Failed to import fast_indexer C++ module, using Python version")

# Attempt to import C++ index engine
try:
    import fast_index_engine
    USE_CPP_ENGINE = True
except ImportError:
    USE_CPP_ENGINE = False
    # Fail silently since Whoosh is the default solution


def load_all_json_files(data_path="/home/zhangwj/my_model/data"):
    """
    Load JSON data, supports folder and single file input, automatically skips invalid data
    Copied from index.py to maintain functional consistency
    """
    all_data = []
    file_count = 0
    error_count = 0
    skipped_docs = 0
    
    def load_single_json_file(file_path):
        """Helper function to load a single JSON file with enhanced error handling"""
        nonlocal all_data, file_count, error_count, skipped_docs
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            if not content:
                print(f"Warning: {file_path} is an empty file, skipped")
                return
            
            # Try to parse as a single JSON object
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    all_data.extend(data)
                    file_count += 1
                    print(f"Loaded file: {file_path} - {len(data)} documents")
                else:
                    print(f"Warning: {file_path} is not in list format, skipped")
            except json.JSONDecodeError as e:
                # If single JSON parsing fails, try parsing JSON objects line by line
                print(f"File {file_path} is not in standard JSON format, attempting line-by-line parsing...")
                try:
                    json_objects = []
                    line_errors = 0
                    valid_objects = 0
                    
                    lines = content.split('\n')
                    max_lines = 1000000
                    if len(lines) > max_lines:
                        print(f"Warning: File is too large ({len(lines)} lines), only processing first {max_lines} lines")
                        lines = lines[:max_lines]
                    
                    current_json = ""
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    
                    total_lines = len(lines)
                    for line_num, line in enumerate(lines, 1):
                        if line_num % 10000 == 0:
                            print(f"Processing progress: {line_num}/{total_lines} lines ({line_num/total_lines*100:.1f}%)")
                        
                        line = line.strip()
                        if not line:
                            continue
                            
                        if line in ['[', ']']:
                            continue
                        
                        for i, char in enumerate(line):
                            if escape_next:
                                escape_next = False
                                continue
                            if char == '\\':
                                escape_next = True
                                continue
                            if char == '"' and not escape_next:
                                in_string = not in_string
                            elif not in_string:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                        
                        current_json += line
                        
                        if brace_count == 0 and current_json.strip():
                            json_str = current_json.strip()
                            if json_str.endswith(','):
                                json_str = json_str[:-1]
                            
                            try:
                                doc = json.loads(json_str)
                                json_objects.append(doc)
                                valid_objects += 1
                            except json.JSONDecodeError:
                                line_errors += 1
                                if line_errors <= 5:
                                    print(f"Skipping invalid JSON object at line {line_num} in {file_path}")
                            
                            current_json = ""
                    
                    if json_objects:
                        all_data.extend(json_objects)
                        file_count += 1
                        print(f"Loaded file: {file_path} - {len(json_objects)} documents (multi-line parsing, skipped {line_errors} invalid objects)")
                    else:
                        print(f"Warning: No valid JSON data found in {file_path}")
                except Exception as e2:
                    print(f"Error: Failed to parse file {file_path}: {e2}")
                    error_count += 1
                    
        except Exception as e:
            print(f"Error: Failed to read file {file_path}: {e}")
            error_count += 1
    
    # Check if input is a file or folder
    if os.path.isfile(data_path):
        if data_path.endswith('.json'):
            load_single_json_file(data_path)
        else:
            print(f"Error: {data_path} is not a JSON file")
    elif os.path.isdir(data_path):
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    load_single_json_file(file_path)
    else:
        print(f"Error: {data_path} is neither a file nor a folder")
    
    print(f"\nLoading Statistics:")
    print(f"Successfully loaded files: {file_count}")
    print(f"Files with errors: {error_count}")
    print(f"Skipped invalid documents: {skipped_docs}")
    print(f"Total valid documents: {len(all_data)}")
    
    return all_data


def load_json_files_batch(data_path, batch_size=1000):
    """
    Load JSON data in batches, returns a generator to reduce memory usage
    Supports folder and single file input, automatically skips invalid data
    
    Args:
        data_path: Path to JSON file or folder
        batch_size: Number of documents to return per batch
    
    Yields:
        List of documents per batch (list of dict)
    """
    file_count = 0
    error_count = 0
    total_docs = 0
    current_batch = []
    
    def count_lines_fast(file_path):
        """Fast line count for files (may be slow for extremely large files)"""
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            # If file exceeds 500MB, line counting may be slow - return None to use unknown mode in tqdm
            if file_size_mb > 500:
                print(f"File is large ({file_size_mb:.1f}MB), skipping line count to save time...")
                return None
            with open(file_path, "rb") as f:
                count = sum(1 for _ in tqdm(f, desc="Counting lines", unit="lines", leave=False))
            return count
        except Exception:
            return None
    
    def parse_json_lines(file_path, f, total_lines=None):
        """Parse JSON objects line by line, returns generator with tqdm progress display"""
        current_json = ""
        brace_count = 0
        in_string = False
        escape_next = False
        line_num = 0
        
        # Use tqdm for progress display
        if total_lines:
            # Use unit scaling (K, M, etc.) if lines exceed 100,000
            use_scale = total_lines > 100000
            pbar = tqdm(total=total_lines, desc=f"Parsing {os.path.basename(file_path)}", unit="lines", unit_scale=use_scale)
        else:
            pbar = tqdm(desc=f"Parsing {os.path.basename(file_path)}", unit="lines", unit_scale=True)
        
        try:
            for line in f:
                line_num += 1
                pbar.update(1)
                
                line = line.strip()
                if not line or line in ['[', ']']:
                    continue
                
                for char in line:
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                
                current_json += line
                
                if brace_count == 0 and current_json.strip():
                    json_str = current_json.strip().rstrip(',')
                    try:
                        doc = json.loads(json_str)
                        yield doc
                    except json.JSONDecodeError:
                        pass
                    current_json = ""
        finally:
            pbar.close()
    
    def load_single_json_file_batch(file_path):
        """Load single JSON file in batches, prioritize line-by-line parsing to save memory"""
        nonlocal file_count, error_count, total_docs, current_batch
        
        try:
            # Check file size - use line-by-line parsing directly for large files
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # Count lines for tqdm progress
            print(f"Counting lines for file: {os.path.basename(file_path)}...")
            total_lines = count_lines_fast(file_path)
            if total_lines:
                print(f"File has {total_lines:,} lines")
            
            # Even for files >50MB, first try standard JSON parsing (if array format)
            # Fall back to line-by-line parsing if standard parsing fails (out of memory)
            if file_size_mb > 50:
                print(f"File {file_path} is large ({file_size_mb:.1f}MB), attempting standard JSON parsing first...")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        # Check if it's standard JSON array format
                        first_chars = f.read(100)
                        f.seek(0)
                        
                        if first_chars.strip().startswith('['):
                            # Attempt standard JSON parsing (even for large files)
                            try:
                                content = f.read()
                                data = json.loads(content)
                                if isinstance(data, list):
                                    doc_count = 0
                                    for doc in tqdm(data, desc=f"Loading {os.path.basename(file_path)}", unit="documents"):
                                        current_batch.append(doc)
                                        doc_count += 1
                                        total_docs += 1
                                        if len(current_batch) >= batch_size:
                                            yield current_batch
                                            current_batch = []
                                    file_count += 1
                                    print(f"Loaded file: {file_path} - {doc_count} documents (standard JSON parsing)")
                                    # Return final batch
                                    if current_batch:
                                        yield current_batch
                                        current_batch = []
                                    return
                            except MemoryError:
                                print(f"Out of memory, falling back to line-by-line parsing...")
                                f.seek(0)
                            except json.JSONDecodeError as e:
                                print(f"Standard JSON parsing failed: {e}, falling back to line-by-line parsing...")
                                f.seek(0)
                
                except Exception as e:
                    print(f"Error during standard JSON parsing attempt: {e}, falling back to line-by-line parsing...")
                
                # Fall back to line-by-line parsing
                print(f"Using line-by-line parsing to save memory...")
                doc_count = 0
                with open(file_path, "r", encoding="utf-8") as f:
                    for doc in parse_json_lines(file_path, f, total_lines=total_lines):
                        current_batch.append(doc)
                        doc_count += 1
                        total_docs += 1
                        if len(current_batch) >= batch_size:
                            yield current_batch
                            current_batch = []
                
                if doc_count > 0:
                    file_count += 1
                    print(f"Loaded file: {file_path} - {doc_count} documents (line-by-line parsing)")
                else:
                    print(f"Warning: No valid JSON data found in {file_path}")
                # Return final batch
                if current_batch:
                    yield current_batch
                    current_batch = []
                return
            
            # For small files, try standard JSON parsing first, fall back to line-by-line if failed
            try:
                doc_count = 0
                with open(file_path, "r", encoding="utf-8") as f:
                    # Check first line for standard JSON array
                    first_chars = f.read(100)
                    f.seek(0)
                    
                    if first_chars.strip().startswith('['):
                        # Attempt full parsing (only for small files)
                        try:
                            content = f.read()
                            data = json.loads(content)
                            if isinstance(data, list):
                                for doc in tqdm(data, desc=f"Loading {os.path.basename(file_path)}", unit="documents"):
                                    current_batch.append(doc)
                                    total_docs += 1
                                    if len(current_batch) >= batch_size:
                                        yield current_batch
                                        current_batch = []
                                file_count += 1
                                print(f"Loaded file: {file_path} - {len(data)} documents")
                                return
                        except (json.JSONDecodeError, MemoryError):
                            # Standard parsing failed, fall back to line-by-line
                            f.seek(0)
                    
                    # Line-by-line parsing
                    for doc in parse_json_lines(file_path, f, total_lines=total_lines):
                        current_batch.append(doc)
                        doc_count += 1
                        total_docs += 1
                        if len(current_batch) >= batch_size:
                            yield current_batch
                            current_batch = []
                    
                    if doc_count > 0:
                        file_count += 1
                        print(f"Loaded file: {file_path} - {doc_count} documents (line-by-line parsing)")
                    else:
                        print(f"Warning: No valid JSON data found in {file_path}")
                        
            except Exception as e2:
                print(f"Error: Failed to parse file {file_path}: {e2}")
                error_count += 1
                    
        except MemoryError:
            print(f"Error: Out of memory, cannot read file {file_path}")
            error_count += 1
        except Exception as e:
            print(f"Error: Failed to read file {file_path}: {e}")
            error_count += 1
    
    # Check if input is a file or folder
    if os.path.isfile(data_path):
        if data_path.endswith('.json'):
            for batch in load_single_json_file_batch(data_path):
                yield batch
        else:
            print(f"Error: {data_path} is not a JSON file")
    elif os.path.isdir(data_path):
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    for batch in load_single_json_file_batch(file_path):
                        yield batch
    else:
        print(f"Error: {data_path} is neither a file nor a folder")
    
    # Return final batch
    if current_batch:
        yield current_batch
    
    print(f"\nLoading Statistics:")
    print(f"Successfully loaded files: {file_count}")
    print(f"Files with errors: {error_count}")
    print(f"Total valid documents: {total_docs}")


def clean_json_record(record):
    """
    Clean JSON record by removing invalid fields
    Prioritize C++ version for better performance
    
    Args:
        record: Raw record
    
    Returns:
        Cleaned record
    """
    if not isinstance(record, dict):
        return None
    
    if USE_CPP:
        try:
            return fast_indexer.clean_json_record(record)
        except Exception as e:
            # Fall back to Python version if C++ version errors
            print(f"Warning: C++ clean_json_record error, using Python version: {e}")
    
    # Python version (original implementation)
    cleaned = {}
    basic_fields = ['ip', 'ip_str', 'country_code', 'country_name', 'city', 
                   'latitude', 'longitude', 'isp', 'org', 'asn', 'last_update',
                   'hostnames', 'domains', 'tags', 'data']
    
    for field in basic_fields:
        if field in record and record[field] is not None:
            value = record[field]
            
            if isinstance(value, str):
                if len(value) > 10000:
                    value = value[:10000] + "..."
                cleaned[field] = value
            elif isinstance(value, list):
                cleaned_list = []
                for item in value:
                    if isinstance(item, str) and len(item) > 1000:
                        cleaned_list.append(item[:1000] + "...")
                    elif isinstance(item, (str, int, float, bool)):
                        cleaned_list.append(item)
                if cleaned_list:
                    cleaned[field] = cleaned_list
            elif isinstance(value, (int, float, bool)):
                cleaned[field] = value
            elif isinstance(value, dict):
                cleaned[field] = value
    
    return cleaned


class ImprovedDynamicIndexer:
    """Dynamic indexer implementation with automatic field discovery and index building (C++ optimized version)"""
    
    def __init__(self, index_dir="improved_dynamic_index", create_new=True, max_fields=100, use_cpp_engine=True):
        self.index_dir = index_dir
        self.max_fields = max_fields
        self.use_cpp_engine = use_cpp_engine and USE_CPP_ENGINE
        self.schema = self._create_base_schema()
        
        # Initialize C++ engine (if available and enabled)
        self.cpp_engine = None
        if self.use_cpp_engine:
            try:
                self.cpp_engine = fast_index_engine.FastIndexEngine()
                print("✓ Using C++ index engine for acceleration")
            except Exception as e:
                print(f"Warning: Failed to initialize C++ engine, using Whoosh: {e}")
                self.use_cpp_engine = False

        if create_new:
            if os.path.exists(self.index_dir):
                shutil.rmtree(self.index_dir)
            os.mkdir(self.index_dir)

            # Whoosh for compatibility and persistence (if needed)
            if not self.use_cpp_engine:
                self.ix = index.create_in(self.index_dir, self.schema)
                self.writer = self.ix.writer()
            else:
                # In C++ engine mode, write to Whoosh simultaneously for persistence
                # This ensures Whoosh index can be read when create_new=False
                self.ix = index.create_in(self.index_dir, self.schema)
                self.writer = self.ix.writer()  # Keep writer for persistence support
            self.discovered_fields = set(self.schema.names())
            self.field_counter = defaultdict(int)
            self.document_ids = set()
            self.selected_fields = set()
        else:
            if not os.path.exists(self.index_dir) and not self.use_cpp_engine:
                raise ValueError(f"Index directory does not exist: {self.index_dir}")
            
            if not self.use_cpp_engine:
                self.ix = index.open_dir(self.index_dir)
                self.writer = None
                self.discovered_fields = set(self.ix.schema.names())
                self.field_counter = defaultdict(int)
                self.document_ids = set()
                self.selected_fields = set(self.discovered_fields)
            else:
                # In C++ engine mode, need to rebuild index or load from Whoosh
                # For compatibility (e.g., clean_rules_2.py needs access to ix.searcher()), still open Whoosh index
                if os.path.exists(self.index_dir):
                    try:
                        self.ix = index.open_dir(self.index_dir)
                        self.discovered_fields = set(self.ix.schema.names())
                        self.selected_fields = set(self.discovered_fields)
                    except:
                        self.ix = None
                        self.discovered_fields = set()
                        self.selected_fields = set()
                else:
                    self.ix = None
                    self.discovered_fields = set()
                    self.selected_fields = set()
                self.writer = None
                self.field_counter = defaultdict(int)
                self.document_ids = set()

    def _create_base_schema(self):
        content_exact_analyzer = RegexTokenizer(expression=r"[a-zA-Z0-9\-\._:/]+") | LowercaseFilter()
        return Schema(
            doc_id=ID(stored=True, unique=True),
            content=TEXT(analyzer=StandardAnalyzer(stoplist=None), stored=False),
            content_exact=TEXT(analyzer=content_exact_analyzer, stored=False),
            full_document=STORED()
        )

    def _clean_field_name(self, field_name):
        cleaned = field_name[1:] if field_name.startswith('_') else field_name
        return cleaned.replace(' ', '_')

    def _flatten_dict(self, d, parent_key='', sep='.', max_depth=3, current_depth=0):
        """
        Flatten dictionary with maximum depth limit of 3 levels
        Prioritize C++ version for better performance
        """
        if USE_CPP and current_depth == 0:
            try:
                # C++ version returns unordered_map<string, string>, need to convert to dict
                flat_dict_cpp = fast_indexer.flatten_dict(d, parent_key, sep, max_depth, current_depth)
                return dict(flat_dict_cpp)
            except Exception as e:
                # Fall back to Python version if C++ version errors
                print(f"Warning: C++ flatten_dict error, using Python version: {e}")
        
        # Python version (original implementation)
        items = []
        
        if current_depth >= max_depth:
            return {}
            
        for k, v in d.items():
            cleaned_key = self._clean_field_name(k)
            new_key = f"{parent_key}{sep}{cleaned_key}" if parent_key else cleaned_key
            
            if isinstance(v, dict):
                nested_items = self._flatten_dict(v, new_key, sep=sep, max_depth=max_depth, current_depth=current_depth + 1)
                items.extend(nested_items.items())
            elif isinstance(v, list):
                list_str = ','.join(map(str, v[:100]))
                if len(v) > 100:
                    list_str += "..."
                items.append((new_key, list_str))
            else:
                items.append((new_key, str(v) if v is not None else ''))
        return dict(items)

    def analyze_field_frequency(self, data_sample, sample_size=None):
        """
        Analyze field occurrence frequency and select most frequently used fields
        Supports iterator and list input
        
        Args:
            data_sample: List or iterator of documents
            sample_size: If specified, only analyze first N documents for field frequency (saves memory)
        """
        field_frequency = defaultdict(int)
        total_docs = 0
        
        # Handle iterator or list
        try:
            data_iter = iter(data_sample)
            if sample_size:
                print(f"Analyzing field frequency for first {sample_size} documents (for field selection)...")
            else:
                print("Analyzing field frequency for documents...")
        except TypeError:
            # If not iterator, treat as list
            data_iter = data_sample
            if sample_size:
                print(f"Analyzing field frequency for first {min(sample_size, len(data_sample))} documents (for field selection)...")
            else:
                print(f"Analyzing field frequency for {len(data_sample)} documents...")
        
        for doc_idx, doc in enumerate(tqdm(data_iter, desc="Analyzing field frequency")):
            if sample_size and doc_idx >= sample_size:
                break
            try:
                flat_doc = self._flatten_dict(doc)
                for field in flat_doc.keys():
                    field_frequency[field] += 1
                total_docs += 1
            except Exception as e:
                continue
        
        sorted_fields = sorted(field_frequency.items(), key=lambda x: x[1], reverse=True)
        
        selected_fields = set()
        for field, freq in sorted_fields[:self.max_fields]:
            selected_fields.add(field)
        
        print(f"Field frequency analysis completed:")
        print(f"Number of documents analyzed: {total_docs}")
        print(f"Total fields discovered: {len(field_frequency)}")
        print(f"Number of selected fields: {len(selected_fields)}")
        
        print(f"\nTop 100 most frequently used fields:")
        for i, (field, freq) in enumerate(sorted_fields[:100]):
            percentage = (freq / total_docs) * 100 if total_docs > 0 else 0
            print(f"{i+1:2d}. {field:<30} Occurrences: {freq:>6} ({percentage:5.1f}%)")
        
        return selected_fields

    def _build_schema(self, selected_fields):
        """Build schema containing only selected frequently used fields"""
        new_schema = self._create_base_schema()
        
        for field in selected_fields:
            if field not in new_schema:
                if field.endswith(('id', 'hash', 'code', 'ip', 'port')):
                    new_schema.add(field, KEYWORD(stored=True))
                else:
                    new_schema.add(field, TEXT(stored=True))
        
        if os.path.exists(self.index_dir):
            shutil.rmtree(self.index_dir)
        os.mkdir(self.index_dir)
        
        self.ix = index.create_in(self.index_dir, new_schema)
        self.writer = self.ix.writer()
        self.discovered_fields = set(new_schema.names())
        self.selected_fields = selected_fields
        
        print(f"Schema built successfully, contains {len(self.discovered_fields)} fields")

    def add_document(self, doc, doc_id):
        if self.use_cpp_engine and self.cpp_engine is not None:
            # Use C++ engine (write to Whoosh simultaneously for persistence)
            if doc_id in self.document_ids:
                return False
            self.document_ids.add(doc_id)
            
            # Use optimized flattening function
            flat_doc = self._flatten_dict(doc)
            
            filtered_doc = {}
            selected_fields_list = list(self.selected_fields)
            for field, value in flat_doc.items():
                if field in self.selected_fields:
                    filtered_doc[field] = value
                    self.field_counter[field] += 1
            
            content = ' '.join(filtered_doc.values())
            
            doc_data = {
                'doc_id': str(doc_id), 
                'full_document': doc, 
                'content': content, 
                'content_exact': content
            }
            doc_data.update(filtered_doc)
            
            # Add to both C++ engine and Whoosh (for persistence)
            py_doc_data = doc_data
            cpp_result = self.cpp_engine.add_document(doc_id, py_doc_data, selected_fields_list)
            
            # Write to Whoosh simultaneously for persistence
            if self.writer is not None:
                try:
                    self.writer.add_document(**doc_data)
                except Exception as e:
                    # Whoosh write failure does not affect C++ engine
                    print(f"Warning: Failed to write document {doc_id} to Whoosh: {e}")
            
            return cpp_result
        else:
            # Use Whoosh
            if self.writer is None:
                raise RuntimeError("Indexer is in read-only mode, cannot add documents")
            
            if doc_id in self.document_ids:
                return False
            self.document_ids.add(doc_id)
            
            # Use optimized flattening function
            flat_doc = self._flatten_dict(doc)
            
            filtered_doc = {}
            for field, value in flat_doc.items():
                if field in self.selected_fields:
                    filtered_doc[field] = value
                    self.field_counter[field] += 1
            
            content = ' '.join(filtered_doc.values())
            
            doc_data = {
                'doc_id': str(doc_id), 
                'full_document': doc, 
                'content': content, 
                'content_exact': content
            }
            doc_data.update(filtered_doc)
            
            self.writer.add_document(**doc_data)
            return True

    def commit(self):
        if self.use_cpp_engine:
            # C++ engine does not require commit (in-memory index), but need to commit Whoosh writes
            if self.writer is not None:
                try:
                    self.writer.commit()
                    self.writer = self.ix.writer()  # Re-acquire writer for future writes
                except Exception as e:
                    print(f"Warning: Whoosh commit failed: {e}")
            return
        if self.writer is None:
            raise RuntimeError("Indexer is in read-only mode, cannot commit changes")
        self.writer.commit()
        self.writer = self.ix.writer()

    def search(self, query_str, fields=None, limit=10, mode="OR", highlight=False, k=None):
        if self.use_cpp_engine and self.cpp_engine is not None:
            # Check if C++ engine has data, fall back to Whoosh if empty
            try:
                cpp_doc_count = self.cpp_engine.get_doc_count()
                # If C++ engine has 0 or very few documents, index is not loaded - fall back to Whoosh
                if cpp_doc_count == 0:
                    if self.ix is not None:
                        # Fall back to Whoosh
                        pass  # Continue to Whoosh search logic below
                    else:
                        return []
                else:
                    # Use C++ engine for search
                    k_filter = k if k is not None else -1
                    
                    if mode.upper() == "OR":
                        results = self.cpp_engine.search_or(query_str, limit, k_filter)
                    else:  # AND mode
                        results = self.cpp_engine.search_and(query_str, limit, k_filter)
                    
                    # Convert result format for compatibility
                    return results
            except Exception as e:
                # C++ engine error, fall back to Whoosh
                print(f"Warning: C++ engine search failed, falling back to Whoosh: {e}")
                # Continue to Whoosh search logic below
        
        # Use Whoosh search (fallback for empty/error C++ engine, or when C++ engine not enabled)
        if self.ix is None:
            return []
        
        if fields:
            fields = [self._clean_field_name(f) for f in fields]
        else:
            fields = [
                f for f in self.discovered_fields
                if f not in ['doc_id', 'full_document'] and
                isinstance(self.ix.schema[f], (TEXT, KEYWORD))
            ]
            if 'content' not in fields:
                fields.append('content')

        group = OrGroup if mode.upper() == "OR" else AndGroup
        parser = MultifieldParser(fields, schema=self.ix.schema, group=group)
        q = parser.parse(query_str)

        results = []
        seen_doc_ids = set()
        with self.ix.searcher() as searcher:
            hits = searcher.search(q, limit=limit)
            for hit in hits:
                doc_id = hit['doc_id']
                if k is not None:
                    try:
                        if int(doc_id) >= k:
                            continue
                    except Exception:
                        pass
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                match_info = {}
                for field in fields:
                    if field in hit and hit[field]:
                        match_info[field] = hit.highlights(field, top=3) if highlight else hit[field]
                results.append({
                    'doc_id': doc_id,
                    'score': hit.score,
                    'matches': match_info,
                    'document': hit['full_document']
                })
                if len(results) >= limit:
                    break
        return results

    def search_exact_terms(self, terms, limit=1000, k=None):
      
        use_cpp = False
        if self.use_cpp_engine and self.cpp_engine is not None:
            try:
                cpp_doc_count = self.cpp_engine.get_doc_count()
                if cpp_doc_count > 0:
         
                    k_filter = k if k is not None else -1
                    terms_list = [str(t) for t in terms]
                    try:
                        results = self.cpp_engine.search_exact_terms(terms_list, limit, k_filter)
                     
                        if results is not None:
                            return results
                  
                    except Exception as e2:
                    
                        pass
            except Exception as e:
     
                pass
        
   
        if not terms:
            return []
     
        import re
        tokenizer_pattern = r"[a-zA-Z0-9\-\._:/]+"
        
        query_components = []
        for term in terms:
            term_str = str(term).strip().lower()
            if not term_str:
                continue
            

            tokens = re.findall(tokenizer_pattern, term_str)
            if not tokens:
                continue
            
            if len(tokens) == 1:
   
                query_components.append(Term("content_exact", tokens[0]))
            else:
 
                query_components.append(And([Term("content_exact", token) for token in tokens]))
        
        if not query_components:
            return []
        

        query = And(query_components)

        results = []
        seen_doc_ids = set()
        if self.ix is None:
            return []
        with self.ix.searcher() as searcher:
            hits = searcher.search(query, limit=limit)
            for hit in hits:
                doc_id = hit['doc_id']
                if k is not None:
                    try:
                        if int(doc_id) >= k:
                            continue
                    except Exception:
                        pass
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                results.append({
                    'doc_id': doc_id,
                    'score': hit.score,
                    'matches': {},
                    'document': hit['full_document']
                })
                if len(results) >= limit:
                    break
        return results

    def get_stats(self):
        if self.use_cpp_engine and self.cpp_engine is not None:

            doc_count = self.cpp_engine.get_doc_count()
            field_stats = {f: self.field_counter[f] for f in sorted(self.discovered_fields)}
            return {
                'document_count': doc_count,
                'total_fields': len(self.discovered_fields),
                'index_size': 0,  # C++ engine uses in-memory index, size not tracked
                'field_distribution': field_stats
            }
        else:

            if self.ix is None:
                return {
                    'document_count': 0,
                    'total_fields': 0,
                    'index_size': 0,
                    'field_distribution': {}
                }
            with self.ix.searcher() as searcher:
                index_size = sum(
                    os.path.getsize(os.path.join(self.index_dir, f))
                    for f in os.listdir(self.index_dir)
                    if os.path.isfile(os.path.join(self.index_dir, f))
                )
                field_stats = {f: self.field_counter[f] for f in sorted(self.discovered_fields)}
                return {
                    'document_count': searcher.doc_count(),
                    'total_fields': len(self.discovered_fields),
                    'index_size': index_size,
                    'field_distribution': field_stats
                }
    
    def match_keywords_direct(self, keywords, doc_id_range=None, batch_size=1000):
        
        if not keywords:
            return set()
        
      
        keywords = [str(kw).strip() for kw in keywords if kw and str(kw).strip()]
        if not keywords:
            return set()
        
        matched_doc_ids = set()
        

        if self.ix is None:
            return set()
        
        with self.ix.searcher() as searcher:
            total_docs = searcher.doc_count()
            
       
            for doc_num in range(total_docs):
                try:
                    stored_doc = searcher.stored_fields(doc_num)
                    doc_id_str = stored_doc.get('doc_id')
                    if doc_id_str is None:
                        continue
                    
                    try:
                        doc_id = int(doc_id_str)
                    except (ValueError, TypeError):
                        continue
                    
                   
                    if doc_id_range is not None:
                        start_id, end_id = doc_id_range
                        if doc_id < start_id or doc_id >= end_id:
                            continue
                    
                   
                    full_doc = stored_doc.get('full_document', {})
                    if not full_doc:
                        continue
                    
                   
                    doc_text = self._document_to_text(full_doc)
                    
                   
                    all_keywords_found = True
                    for keyword in keywords:
                        if keyword not in doc_text:
                            all_keywords_found = False
                            break
                    
                    if all_keywords_found:
                        matched_doc_ids.add(doc_id)
                except (KeyError, ValueError, TypeError, AttributeError):
                    
                    continue
        
        return matched_doc_ids
    
    def _document_to_text(self, doc):

        if isinstance(doc, dict):
         
            text_parts = []
            for key, value in doc.items():
                if isinstance(value, dict):
                
                    text_parts.append(self._document_to_text(value))
                elif isinstance(value, list):
                    
                    for item in value:
                        if isinstance(item, dict):
                            text_parts.append(self._document_to_text(item))
                        else:
                            text_parts.append(str(item))
                else:
                    text_parts.append(str(value))
            return ' '.join(text_parts)
        elif isinstance(doc, list):
            text_parts = []
            for item in doc:
                if isinstance(item, dict):
                    text_parts.append(self._document_to_text(item))
                else:
                    text_parts.append(str(item))
            return ' '.join(text_parts)
        else:
            return str(doc)