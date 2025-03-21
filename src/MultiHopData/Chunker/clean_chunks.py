import json
from collections import defaultdict, OrderedDict


def deduplicate_json(input_file, output_file):
    """
    Comprehensively detect and remove duplicate documents based on content.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
    
    Returns:
        tuple: (total_duplicates, duplicate_stats)
    """
    # Read JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Track duplicates
    content_seen = {}  # Store first occurrence of each content
    unique_documents = []
    duplicate_count = 0
    duplicate_stats = defaultdict(int)
    
    # Process each document
    for doc in data:
        # Create a hashable representation of the document's content
        # We'll exclude doc_id and original_uuid to identify true content duplicates
        doc_content_key = json.dumps({
            'content': doc['content'],
            'chunks': [
                {k: v for k, v in chunk.items() if k != 'chunk_id' and k != 'original_index'}
                for chunk in doc.get('chunks', [])
            ]
        }, sort_keys=True)
        
        # Check if this document content has been seen before
        if doc_content_key in content_seen:
            duplicate_count += 1
            duplicate_stats['total_document_duplicates'] += 1
            
            # Optionally, you can choose which document to keep based on some criteria
            # Here we're keeping the first occurrence
            continue
        
        # Mark this content as seen and add to unique documents
        content_seen[doc_content_key] = True
        unique_documents.append(doc)
    
    # Save cleaned data
    with open(output_file, 'w') as f:
        json.dump(unique_documents, f, indent=2)
    
    return duplicate_count, dict(duplicate_stats)

def main(input_file, output_file):
    try:
        total_dupes, stats = deduplicate_json(input_file, output_file)
        
        print(f"Deduplication complete!")
        print(f"Total duplicates removed: {total_dupes}")
        print("\nDuplicate statistics:")
        for level, count in stats.items():
            print(f"- {level}: {count} duplicates")
        print(f"\nCleaned data saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    
    for task in task_domains:
        input_file = f"auto-rag-eval/MultiHopData/{task}/chunks/docs_chunk_semantic_v7.json"
        output_file = f"auto-rag-eval/MultiHopData/{task}/chunks/docs_chunk_semantic_v7_cleaned.json"
        
        main(input_file, output_file)