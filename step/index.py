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

def clean_json_record(record):
    """
    Clean JSON record by removing invalid fields and truncating overlength values
    """
    if not isinstance(record, dict):
        return None
    
    cleaned = {}
    
    # Core fields to preserve
    basic_fields = ['ip', 'ip_str', 'country_code', 'country_name', 'city', 
                   'latitude', 'longitude', 'isp', 'org', 'asn', 'last_update',
                   'hostnames', 'domains', 'tags', 'data']
    
    for field in basic_fields:
        if field in record and record[field] is not None:
            value = record[field]
            
            # Truncate long string values
            if isinstance(value, str):
                if len(value) > 10000:  # Max string length limit
                    value = value[:10000] + "..."
                cleaned[field] = value
            
            # Process list values (truncate long items)
            elif isinstance(value, list):
                cleaned_list = []
                for item in value:
                    if isinstance(item, str) and len(item) > 1000:
                        cleaned_list.append(item[:1000] + "...")
                    elif isinstance(item, (str, int, float, bool)):
                        cleaned_list.append(item)
                if cleaned_list:
                    cleaned[field] = cleaned_list
            
            # Preserve numeric/boolean values as-is
            elif isinstance(value, (int, float, bool)):
                cleaned[field] = value
            
            # Preserve nested dictionaries
            elif isinstance(value, dict):
                cleaned[field] = value
    
    return cleaned

class ImprovedDynamicIndexer:
    """
    Dynamic indexer with automatic field discovery and optimized indexing
    Supports schema customization based on field frequency analysis
    """
    
    def __init__(self, index_dir="improved_dynamic_index", create_new=True, max_fields=100):
        self.index_dir = index_dir
        self.max_fields = max_fields  # Max number of fields to include in schema
        self.schema = self._create_base_schema()

        if create_new:
            # Create new index (delete existing if present)
            if os.path.exists(self.index_dir):
                shutil.rmtree(self.index_dir)
            os.mkdir(self.index_dir)

            self.ix = index.create_in(self.index_dir, self.schema)
            self.writer = self.ix.writer()
            self.discovered_fields = set(self.schema.names())
            self.field_counter = defaultdict(int)
            self.document_ids = set()
            self.selected_fields = set()  # Fields selected after frequency analysis
        else:
            # Open existing index (read-only mode)
            if not os.path.exists(self.index_dir):
                raise ValueError(f"Index directory does not exist: {self.index_dir}")
            
            self.ix = index.open_dir(self.index_dir)
            self.writer = None  # No writer in read-only mode
            self.discovered_fields = set(self.ix.schema.names())
            self.field_counter = defaultdict(int)  # Reset counter for existing index
            self.document_ids = set()  # Reset document ID tracking
            self.selected_fields = set(self.discovered_fields)  # Use all existing fields
    
    def _create_base_schema(self):
        """Create base schema with core fields (content, exact content, document ID)"""
        # Analyzer for exact matching (no stop words, preserves special characters)
        content_exact_analyzer = RegexTokenizer(expression=r"[a-zA-Z0-9\-\._:/]+") | LowercaseFilter()
        return Schema(
            doc_id=ID(stored=True, unique=True),
            content=TEXT(analyzer=StandardAnalyzer(stoplist=None), stored=False),
            content_exact=TEXT(analyzer=content_exact_analyzer, stored=False),
            full_document=STORED()
        )

    def _clean_field_name(self, field_name):
        """Clean field names to be compatible with Whoosh (remove underscores/prefixes)"""
        cleaned = field_name[1:] if field_name.startswith('_') else field_name
        return cleaned.replace(' ', '_')

    def _flatten_dict(self, d, parent_key='', sep='.', max_depth=3, current_depth=0):
        """
        Flatten nested dictionaries into flat key-value pairs (max 3 levels deep)
        Handles list values by converting to comma-separated strings
        """
        items = []
        
        # Stop recursion at max depth
        if current_depth >= max_depth:
            return {}
            
        for k, v in d.items():
            cleaned_key = self._clean_field_name(k)
            new_key = f"{parent_key}{sep}{cleaned_key}" if parent_key else cleaned_key
            
            if isinstance(v, dict):
                # Recursively flatten nested dictionaries
                nested_items = self._flatten_dict(v, new_key, sep=sep, max_depth=max_depth, current_depth=current_depth + 1)
                items.extend(nested_items.items())
            elif isinstance(v, list):
                # Convert lists to truncated string representation
                list_str = ','.join(map(str, v[:100]))  # Truncate lists to 100 items
                if len(v) > 100:
                    list_str += "..."
                items.append((new_key, list_str))
            else:
                # Convert all other values to string (handle None)
                items.append((new_key, str(v) if v is not None else ''))
        return dict(items)

    def analyze_field_frequency(self, data_sample):
        """
        Analyze field occurrence frequency from sample data
        Returns top N most frequent fields (N = max_fields)
        """
        field_frequency = defaultdict(int)
        total_docs = 0
        
        for doc in tqdm(data_sample, desc="Analyzing field frequency"):
            try:
                flat_doc = self._flatten_dict(doc)
                for field in flat_doc.keys():
                    field_frequency[field] += 1
                total_docs += 1
            except Exception as e:
                continue
        
        # Sort fields by frequency (descending)
        sorted_fields = sorted(field_frequency.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N fields
        selected_fields = set()
        for field, freq in sorted_fields[:self.max_fields]:
            selected_fields.add(field)

        # Print top 100 fields for reference
        print(f"\nTop 100 most frequent fields:")
        for i, (field, freq) in enumerate(sorted_fields[:100]):
            percentage = (freq / total_docs) * 100 if total_docs > 0 else 0
            print(f"{i+1:2d}. {field:<30} Occurrences: {freq:>6} ({percentage:5.1f}%)")
        
        return selected_fields

    def _build_schema(self, selected_fields):
        """
        Build custom schema with selected frequent fields
        Uses KEYWORD type for ID/IP/port fields, TEXT for others
        """
        new_schema = self._create_base_schema()
        
        # Add selected fields to schema
        for field in selected_fields:
            if field not in new_schema:
                # Use KEYWORD type for identifier fields (exact matching)
                if field.endswith(('id', 'hash', 'code', 'ip', 'port')):
                    new_schema.add(field, KEYWORD(stored=True))
                # Use TEXT type for general text fields (full-text search)
                else:
                    new_schema.add(field, TEXT(stored=True))
        
        # Recreate index directory with new schema
        if os.path.exists(self.index_dir):
            shutil.rmtree(self.index_dir)
        os.mkdir(self.index_dir)
        
        self.ix = index.create_in(self.index_dir, new_schema)
        self.writer = self.ix.writer()
        self.discovered_fields = set(new_schema.names())
        self.selected_fields = selected_fields
        
    def add_document(self, doc, doc_id):
        """
        Add single document to index
        Returns True if successful, False if duplicate document ID
        """
        if self.writer is None:
            raise RuntimeError("Index writer not available (read-only mode)")
        
        if doc_id in self.document_ids:
            return False
        self.document_ids.add(doc_id)
        
        # Flatten document structure for indexing
        flat_doc = self._flatten_dict(doc)
        
        # Filter to only selected fields
        filtered_doc = {}
        for field, value in flat_doc.items():
            if field in self.selected_fields:
                filtered_doc[field] = value
                self.field_counter[field] += 1
        
        # Create combined content field for full-text search
        content = ' '.join(filtered_doc.values())
        
        # Prepare document data for indexing
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
        """Commit pending writes to index (required to persist changes)"""
        if self.writer is None:
            raise RuntimeError("Index writer not available (read-only mode)")
        self.writer.commit()
        self.writer = self.ix.writer()  # Recreate writer for future writes

    def search(self, query_str, fields=None, limit=10, mode="OR", highlight=False, k=None):
        """
        Full-text search with multi-field support
        Args:
            query_str: Search query string
            fields: List of fields to search (None = all text/keyword fields)
            limit: Max number of results to return
            mode: "OR" (any term match) or "AND" (all terms required)
            highlight: Whether to return highlighted matches
            k: Filter results to only include doc_ids < k (optional)
        Returns:
            List of matching documents with scores and match info
        """
        if fields:
            fields = [self._clean_field_name(f) for f in fields]
        else:
            # Auto-select all text/keyword fields (exclude core fields)
            fields = [
                f for f in self.discovered_fields
                if f not in ['doc_id', 'full_document'] and
                isinstance(self.ix.schema[f], (TEXT, KEYWORD))
            ]
            if 'content' not in fields:
                fields.append('content')

        # Configure query group (OR/AND logic)
        group = OrGroup if mode.upper() == "OR" else AndGroup
        parser = MultifieldParser(fields, schema=self.ix.schema, group=group)
        q = parser.parse(query_str)

        results = []
        seen_doc_ids = set()
        with self.ix.searcher() as searcher:
            hits = searcher.search(q, limit=limit)
            for hit in hits:
                doc_id = hit['doc_id']
                # Filter by document ID threshold if specified
                if k is not None:
                    try:
                        if int(doc_id) >= k:
                            continue
                    except Exception:
                        pass
                # Avoid duplicate results
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                # Collect match information (with highlights if enabled)
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
        """
        Exact term search (matches exact tokens in content_exact field)
        Args:
            terms: List of terms to match exactly
            limit: Max number of results to return
            k: Filter results to only include doc_ids < k (optional)
        Returns:
            List of matching documents with exact term matches
        """
        # Validate input terms
        if not terms:
            return []
        
        # Normalize terms (lowercase, strip whitespace)
        normalized_terms = [str(t).strip().lower() for t in terms if str(t).strip()]
        if not normalized_terms:
            return []
        
        # Build exact match query (all terms required)
        query = And([Term("content_exact", t) for t in normalized_terms])

        results = []
        seen_doc_ids = set()
        with self.ix.searcher() as searcher:
            hits = searcher.search(query, limit=limit)
            for hit in hits:
                doc_id = hit['doc_id']
                # Filter by document ID threshold if specified
                if k is not None:
                    try:
                        if int(doc_id) >= k:
                            continue
                    except Exception:
                        pass
                # Avoid duplicate results
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
        """
        Get index statistics: document count, field distribution, index size
        Returns:
            Dictionary with index metrics
        """
        with self.ix.searcher() as searcher:
            # Calculate total index size (bytes)
            index_size = sum(
                os.path.getsize(os.path.join(self.index_dir, f))
                for f in os.listdir(self.index_dir)
                if os.path.isfile(os.path.join(self.index_dir, f))
            )
            # Field usage statistics
            field_stats = {f: self.field_counter[f] for f in sorted(self.discovered_fields)}
            return {
                'document_count': searcher.doc_count(),
                'total_fields': len(self.discovered_fields),
                'index_size': index_size,
                'field_distribution': field_stats
            }


def load_all_json_files(data_path="./data"):
    """
    Load all JSON files from specified path (file or directory)
    Supports:
    - Single JSON files (array format)
    - Line-delimited JSON files
    - Recursive directory scanning
    Automatically skips invalid JSON and handles large files
    """
    all_data = []
    file_count = 0
    error_count = 0
    skipped_docs = 0
    
    def load_single_json_file(file_path):
        """Helper function to load and parse single JSON file"""
        nonlocal all_data, file_count, error_count, skipped_docs
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            if not content:
                return
            
            # First try standard JSON array parsing
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    # Valid JSON array - add all documents
                    all_data.extend(data)
                    file_count += 1
                else:
                    print(f"Warning: {file_path} - Not a JSON array (skipping)")
            except json.JSONDecodeError as e:
                # Fallback to line-delimited JSON parsing
                print(f"Standard JSON parsing failed for {file_path} - trying line-by-line parsing...")
                try:
                    json_objects = []
                    line_errors = 0
                    valid_objects = 0

                    lines = content.split('\n')
                    
                    # Limit line processing for extremely large files
                    max_lines = 1000000  
                    if len(lines) > max_lines:
                        lines = lines[:max_lines]
                    
                    # State variables for JSON parsing
                    current_json = ""
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    
                    total_lines = len(lines)
                    for line_num, line in enumerate(lines, 1):
                        # Print progress every 10,000 lines
                        if line_num % 10000 == 0:
                            print(f"Processing progress: {line_num}/{total_lines} lines ({line_num/total_lines*100:.1f}%)")
                        
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Skip array delimiters
                        if line in ['[', ']']:
                            continue
                        
                        # Handle escaped characters and string boundaries
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
                        
                        # Parse when braces are balanced
                        if brace_count == 0 and current_json.strip():
                            # Remove trailing commas
                            json_str = current_json.strip()
                            if json_str.endswith(','):
                                json_str = json_str[:-1]
                            
                            try:
                                doc = json.loads(json_str)
                                json_objects.append(doc)
                                valid_objects += 1
                            except json.JSONDecodeError:
                                line_errors += 1
                                if line_errors <= 5:  # Only show first 5 errors
                                    print(f"Skipping invalid JSON object at line {line_num} in {file_path}")
                            
                            current_json = ""
                    
                    if json_objects:
                        # Add valid documents to dataset
                        all_data.extend(json_objects)
                        file_count += 1
                        print(f"Loaded file: {file_path} - {len(json_objects)} documents (line-by-line parsing, skipped {line_errors} invalid objects)")
                    else:
                        print(f"Warning: No valid JSON data found in {file_path}")
                except Exception as e2:
                    print(f"Error: Failed to parse file {file_path}: {e2}")
                    error_count += 1
                    
        except Exception as e:
            print(f"Error: Failed to read file {file_path}: {e}")
            error_count += 1
    
    # Check if input is file or directory
    if os.path.isfile(data_path):
        # Single file processing
        if data_path.endswith('.json'):
            load_single_json_file(data_path)
        else:
            print(f"Error: {data_path} is not a JSON file")
    elif os.path.isdir(data_path):
        # Recursive directory processing
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    load_single_json_file(file_path)
    else:
        print(f"Error: {data_path} is neither a file nor a directory")
    
    # Print loading statistics
    print(f"\nLoading Statistics:")
    print(f"Successfully loaded files: {file_count}")
    print(f"Files with errors: {error_count}")
    print(f"Skipped invalid documents: {skipped_docs}")
    print(f"Total valid documents: {len(all_data)}")
    
    return all_data


if __name__ == "__main__":
    import sys
    import traceback
    
    try:
        # Support command-line argument for data path
        if len(sys.argv) > 1:
            data_path = sys.argv[1]
        else:
            data_path = "/home/zhangwj/res/Dahua_DVR_search_results.json"
        
        print(f"Loading data from {data_path}...")
        print("=" * 50)
        
        # Load all JSON files
        data = load_all_json_files(data_path)
        
        if not data:
            print("No valid JSON data found - exiting program")
            exit(1)

        print(f"\nFound {len(data)} valid documents - analyzing field frequency...")
        print("=" * 50)
        
        # Create indexer instance (max 300 fields)
        indexer = ImprovedDynamicIndexer('./improved_dynamic_index', max_fields=300)
        
        # Analyze field frequency and select top fields
        selected_fields = indexer.analyze_field_frequency(data)
        
        print("Building index schema...")
        print("=" * 50)
        
        # Build custom schema with selected fields
        indexer._build_schema(selected_fields)

        print("Writing documents to index in batch...")
        write_errors = 0
        successful_writes = 0
        
        for doc_id, doc in enumerate(tqdm(data, desc="Writing documents")):
            try:
                if indexer.add_document(doc, doc_id):
                    successful_writes += 1
            except Exception as e:
                write_errors += 1
                if write_errors <= 10:  # Only show first 10 write errors
                    print(f"Warning: Failed to write document {doc_id}: {e}")
        
        if write_errors > 0:
            print(f"Write complete - skipped {write_errors} problematic documents")
        
        # Commit all changes to index
        indexer.commit()
        print("Index building completed!")
        print("=" * 50)

        # Get and print index statistics
        stats = indexer.get_stats()
        print(f"\nIndex Statistics:")
        print(f"Successfully written documents: {successful_writes}")
        print(f"Failed document writes: {write_errors}")
        print(f"Total documents in index: {stats['document_count']}")
        print(f"Index size: {stats['index_size']/1024/1024:.2f} MB")
        print(f"Total fields in index: {stats['total_fields']}")

        # Test search functionality
        print("\nSample Search Tests:")
        print("=" * 30)
        
        try:
            results_or = indexer.search("dahua", mode="OR", k=10)
            print("=== OR Mode (any keyword match) ===")
            for r in results_or:
                print(f"Document ID: {r['doc_id']} Score: {r['score']:.2f}")
        except Exception as e:
            print(f"OR mode search failed: {e}")
        
        try:
            results_and = indexer.search("dahua", mode="AND", k=10)
            print("\n=== AND Mode (all keywords required) ===")
            for r in results_and:
                print(f"Document ID: {r['doc_id']} Score: {r['score']:.2f}")
        except Exception as e:
            print(f"AND mode search failed: {e}")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        exit(1)
    except Exception as e:
        print(f"Program execution error: {e}")
        print("Detailed error information:")
        traceback.print_exc()
        exit(1)