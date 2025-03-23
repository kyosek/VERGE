import json
from typing import Dict, List, Tuple


def calculate_accuracy(results: List[Dict]) -> float:
    """
    Calculate the accuracy of exam results.
    
    Args:
        results (List[Dict]): List of exam result entries
        
    Returns:
        float: Accuracy score (0-1)
    """
    if not results:
        return 0.0
        
    correct_answers = sum(1 for entry in results if entry['is_correct'])
    return correct_answers / len(results)


def clean_json_results(input_path: str, output_path: str) -> Tuple[int, int, float, float]:
    """
    Clean JSON results by removing invalid questions and calculate accuracies.
    
    Args:
        input_path (str): Path to the input JSON file
        output_path (str): Path where the cleaned JSON will be saved
        
    Returns:
        Tuple[int, int, float, float]: Number of total entries, number of valid entries,
                                      original accuracy, and cleaned accuracy
    """
    # Read the input JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Count total entries and calculate original accuracy
    total_entries = len(results)
    original_accuracy = calculate_accuracy(results)
    
    # Filter out invalid questions
    valid_results = [
        entry for entry in results 
        if not entry['question'].strip().startswith('A)')
    ]
    
    # Count valid entries and calculate cleaned accuracy
    valid_entries = len(valid_results)
    cleaned_accuracy = calculate_accuracy(valid_results)
    
    # Save the cleaned results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(valid_results, f, indent=2, ensure_ascii=False)
        
    return total_entries, valid_entries, original_accuracy, cleaned_accuracy


def main(input_path: str, output_path: str):
    """
    Main function to process the exam results.
    
    Args:
        input_path (str): Path to the input JSON file
        output_path (str): Path where the cleaned JSON will be saved
    """
    total_entries, valid_entries, original_accuracy, cleaned_accuracy = clean_json_results(
        input_path, output_path
    )
    
    # Print summary with accuracy information
    print(f"\nProcessing: {input_path}")
    print(f"Total entries: {total_entries}")
    print(f"Valid entries: {valid_entries}")
    print(f"Removed entries: {total_entries - valid_entries}")
    print(f"Original accuracy: {original_accuracy:.2%}")
    print(f"Cleaned accuracy: {cleaned_accuracy:.2%}")
    print(f"Cleaned results saved to: {output_path}")
    print("-" * 80)


if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    llm_names = ["llama_3_1_8b", "gemma2-9b", "ministral-8b"]
    retriever_types = ["Dense", "Sparse", "Hybrid", "Rerank"]
    
    for task_domain in task_domains:
        print(f"\nProcessing task domain: {task_domain}")
        print("=" * 80)
        
        for llm_name in llm_names:
            for retriever_type in retriever_types:
                input_path = f"auto-rag-eval/MultiHopData/{task_domain}/exam_results/{llm_name}_{retriever_type}_exam_new_gemma2_9b_processed_v2.json_5_results.json"
                output_path = f"auto-rag-eval/MultiHopData/{task_domain}/exam_results/{llm_name}_{retriever_type}_exam_new_gemma2_9b_processed_v3.json_5_results.json"
                try:
                    main(input_path, output_path)
                except Exception as e:
                    print(f"Error: {e}")
                