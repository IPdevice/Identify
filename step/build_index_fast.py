#!/usr/bin/env python3
"""
Build index with C++ optimized version
Automatically use C++ engine to accelerate index building and querying
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))

from fast_index_wrapper import ImprovedDynamicIndexer, load_json_files_batch
from tqdm import tqdm
import traceback

def parse_args():
    parser = argparse.ArgumentParser(description="Build fast index with C++ acceleration")
    parser.add_argument("data_path", help="Path to json data file", nargs="?", default="../test_data.json")
    parser.add_argument("--batch_size", type=int, default=10000, help="Documents per batch")
    parser.add_argument("--sample_size", type=int, default=100000, help="Sample size for field frequency analysis")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        data_path = args.data_path
        batch_size = args.batch_size
        sample_size = args.sample_size
        
        print(f"Loading data in batches from {data_path}...")
        print(f"Batch size: {batch_size}, Field sampling count: {sample_size}")
        print("=" * 50)
        
        # Create indexer instance (automatically use C++ engine for acceleration)
        indexer = ImprovedDynamicIndexer('.improved_dynamic_index', max_fields=300, use_cpp_engine=True)
        
        # Step 1: Batch read and sample analyze field frequency
        print("Step 1: Analyzing field frequency...")
        print("=" * 50)
        
        def sample_documents():
            """Generator: Sample documents from batch reading for field analysis"""
            sample_count = 0
            for batch in load_json_files_batch(data_path, batch_size=batch_size):
                for doc in batch:
                    if sample_count < sample_size:
                        yield doc
                        sample_count += 1
                    else:
                        return
        
        # Analyze field frequency using sampled documents
        sample_data = sample_documents()
        selected_fields = indexer.analyze_field_frequency(sample_data, sample_size=sample_size)
        
        print("\nStep 2: Building index...")
        print("=" * 50)
        
        # Build schema with selected fields
        indexer._build_schema(selected_fields)
        
        # Step 2: Process all documents in batches and write to index
        print("\nStep 3: Writing documents in batches...")
        print("=" * 50)
        
        write_errors = 0
        successful_writes = 0
        doc_id = 0
        
        # Re-read all documents in batches and write to index
        for batch in tqdm(load_json_files_batch(data_path, batch_size=batch_size), desc="Processing batches"):
            for doc in batch:
                try:
                    if indexer.add_document(doc, doc_id):
                        successful_writes += 1
                    doc_id += 1
                except Exception as e:
                    write_errors += 1
                    doc_id += 1
                    if write_errors <= 10:  # Only show first 10 write errors
                        print(f"Warning: Failed to write document {doc_id}: {e}")
        
        if write_errors > 0:
            print(f"Writing completed, skipped {write_errors} problematic documents")
        
        indexer.commit()
        print("\nIndex building completed!")
        print("=" * 50)
        
        stats = indexer.get_stats()
        print(f"\nIndex statistics:")
        print(f"Successfully written documents: {successful_writes}")
        print(f"Failed to write documents: {write_errors}")
        print(f"Total documents in index: {stats['document_count']}")
        print(f"Index size: {stats['index_size']/1024/1024:.2f} MB" if stats['index_size'] > 0 else "Index size: In-memory index")
        print(f"Total fields: {stats['total_fields']}")
        
        # Example search test (exactly the same as index.py)
        print("\nExample search test:")
        print("=" * 30)
        
        try:
            results_or = indexer.search("dahua", mode="OR", k=10)
            print("=== OR mode (contains any keyword) ===")
            for r in results_or:
                print(f"Document ID: {r['doc_id']} Match score: {r['score']:.2f}")
        except Exception as e:
            print(f"OR mode search failed: {e}")
        
        try:
            results_and = indexer.search("dahua", mode="AND", k=10000)
            print("\n=== AND mode (must contain all) ===")
            for r in results_and:
                print(f"Document ID: {r['doc_id']} Match score: {r['score']:.2f}")
        except Exception as e:
            print(f"AND mode search failed: {e}")
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Program execution error: {e}")
        print("Detailed error information:")
        traceback.print_exc()
        sys.exit(1)