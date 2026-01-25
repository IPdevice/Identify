#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os
from datetime import datetime
from typing import Dict, Any, List


TRACKER_FILE = "./rules_growth_tracker.json"

def load_tracker_data() -> List[Dict[str, Any]]:
    """
    Load existing tracker data from JSON file
    Returns empty list if file doesn't exist or error occurs
    """
    if os.path.exists(TRACKER_FILE):
        try:
            with open(TRACKER_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading tracker data: {e}")
            return []
    return []

def save_tracker_data(data: List[Dict[str, Any]]):
    """
    Save tracker data to JSON file
    """
    try:
        with open(TRACKER_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving tracker data: {e}")

def record_batch_processing(batch_num: int, total_processed: int, batch_size: int, rules_count: int = 0, deleted_rules_count: int = 0):
    """
    Record batch processing statistics to tracker file
    
    Args:
        batch_num: Current batch number
        total_processed: Total number of devices processed so far
        batch_size: Number of devices processed in this batch
        rules_count: Total number of rules after this batch
        deleted_rules_count: Number of rules deleted in this batch
    """
    try:
        # Load existing tracker data
        tracker_data = load_tracker_data()
        
        # Create new record
        record = {
            "timestamp": datetime.now().isoformat(),
            "batch_num": batch_num,
            "total_processed": total_processed,
            "batch_size": batch_size,
            "rules_count": rules_count,
            "deleted_rules_count": deleted_rules_count,
            "progress_percentage": round((total_processed / 1000) * 100, 2) if total_processed <= 1000 else 100.0
        }
        
        # Add new record to tracker data
        tracker_data.append(record)
        
        # Save updated tracker data
        save_tracker_data(tracker_data)
        
        # Print summary stats every 10 batches
        if batch_num % 10 == 0:
            print_summary_stats(tracker_data)
            
    except Exception as e:
        print(f"Error recording batch processing: {e}")

def print_summary_stats(tracker_data: List[Dict[str, Any]]):
    """
    Print summary statistics from tracker data (called every 10 batches)
    """
    if not tracker_data:
        return
    
    total_batches = len(tracker_data)
    total_processed = tracker_data[-1]["total_processed"] if tracker_data else 0
    current_rules_count = tracker_data[-1].get("rules_count", 0) if tracker_data else 0
    total_deleted_rules = sum(record.get("deleted_rules_count", 0) for record in tracker_data)
    avg_batch_size = sum(record["batch_size"] for record in tracker_data) / total_batches if total_batches > 0 else 0
    
    # Note: Original function had no print statements - kept empty as in original code
   
def get_processing_stats() -> Dict[str, Any]:
    """
    Calculate and return consolidated processing statistics
    Returns:
        Dictionary with aggregated processing metrics
    """
    tracker_data = load_tracker_data()
    
    if not tracker_data:
        return {
            "total_batches": 0,
            "total_processed": 0,
            "current_rules_count": 0,
            "total_deleted_rules": 0,
            "avg_batch_size": 0,
            "progress_percentage": 0,
            "last_update": None
        }
    
    total_batches = len(tracker_data)
    total_processed = tracker_data[-1]["total_processed"]
    current_rules_count = tracker_data[-1].get("rules_count", 0)
    total_deleted_rules = sum(record.get("deleted_rules_count", 0) for record in tracker_data)
    avg_batch_size = sum(record["batch_size"] for record in tracker_data) / total_batches
    progress_percentage = tracker_data[-1]["progress_percentage"]
    last_update = tracker_data[-1]["timestamp"]
    
    return {
        "total_batches": total_batches,
        "total_processed": total_processed,
        "current_rules_count": current_rules_count,
        "total_deleted_rules": total_deleted_rules,
        "avg_batch_size": round(avg_batch_size, 1),
        "progress_percentage": progress_percentage,
        "last_update": last_update
    }

def clear_tracker_data():
    """
    Clear all tracker data by deleting the tracker file
    """
    try:
        if os.path.exists(TRACKER_FILE):
            os.remove(TRACKER_FILE)
            print("Tracker data cleared successfully")
    except Exception as e:
        print(f"Error clearing tracker data: {e}")

def export_tracker_report(output_file: str = "./rules_growth_tracker_report.txt"):
    """
    Export tracker data to a human-readable text report
    
    Args:
        output_file: Path to save the report file
    """
    try:
        tracker_data = load_tracker_data()
        
        if not tracker_data:
            print("No tracker data available to export")
            return
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("Rules Growth Tracking Report")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Statistics section
        stats = get_processing_stats()
        report_lines.append("【Processing Statistics】")
        report_lines.append(f"Total Batches: {stats['total_batches']}")
        report_lines.append(f"Total Devices Processed: {stats['total_processed']}")
        report_lines.append(f"Current Total Rules: {stats['current_rules_count']}")
        report_lines.append(f"Total Rules Deleted: {stats['total_deleted_rules']}")
        report_lines.append(f"Average Batch Size: {stats['avg_batch_size']}")
        report_lines.append(f"Processing Progress: {stats['progress_percentage']:.1f}%")
        report_lines.append(f"Last Updated: {stats['last_update']}")
        report_lines.append("")
        
        # Detailed records section
        report_lines.append("【Detailed Records】")
        for i, record in enumerate(tracker_data, 1):
            report_lines.append(f"Batch {record['batch_num']}: "
                              f"Processed {record['batch_size']} devices, "
                              f"Total {record['total_processed']} devices, "
                              f"Total Rules {record.get('rules_count', 0)}, "
                              f"Rules Deleted {record.get('deleted_rules_count', 0)}, "
                              f"Progress {record['progress_percentage']:.1f}% "
                              f"({record['timestamp']})")
        
        # Save report to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Tracking report exported to: {output_file}")
        
    except Exception as e:
        print(f"Failed to export tracking report: {e}")

if __name__ == "__main__":
    # Test the tracker functions
    record_batch_processing(1, 5, 5, 10)
    record_batch_processing(2, 10, 5, 12)
    record_batch_processing(3, 15, 5, 15)
    
    # Get and print statistics
    stats = get_processing_stats()
    print(f"\nStatistics: {stats}")
    
    # Export report
    export_tracker_report()