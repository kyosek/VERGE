import json
from typing import List, Dict, Any
from pathlib import Path

def remove_entities(data: List[Dict[Any, Any]]) -> List[Dict[Any, Any]]:
    """
    Remove specified entities from each dictionary in the input list.
    
    Args:
        data (List[Dict[Any, Any]]): List of dictionaries containing question data
        
    Returns:
        List[Dict[Any, Any]]: Processed list with specified entities removed
    """
    entities_to_remove = {
        'direct', 'supporting', 'irrelevant',
        'direct_mean_position', 'direct_position_range', 'direct_position_std',
        'direct_count', 'supporting_mean_position', 'supporting_position_range',
        'supporting_position_std', 'supporting_count', 'irrelevant_mean_position',
        'irrelevant_position_range', 'irrelevant_position_std', 'irrelevant_count'
    }
    
    return [{k: v for k, v in item.items() if k not in entities_to_remove} 
            for item in data]

def process_json_file(input_path: str, output_path: str) -> None:
    """
    Process a single JSON file and save the cleaned version.
    
    Args:
        input_path (str): Path to the input JSON file
        output_path (str): Path where the processed file will be saved
        
    Raises:
        FileNotFoundError: If the input file does not exist
        json.JSONDecodeError: If the input file is not valid JSON
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = remove_entities(data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in input file: {str(e)}", e.doc, e.pos)
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")


if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    # task_domains = ["multifieldqa_en"]
    model_names = [
        'llama_3_1_8b',
        "ministral_8b",
        "gemma2_9b",
    ]
    exam_files = [
        "exam_new_llama_3_2_3b_processed_signal_ratio_v2.json",
        "exam_new_gemma2_9b_processed_signal_ratio_v2.json",
        "exam_new_ministral_8b_processed_signal_ratio_v2.json",
    ]
    # retriever_types = ["closed", "Dense", "Sparse", "Hybrid", "Rerank", "open"]
    retriever_types = ["open"]
    # n_documents = [5, 10]
    n_documents = [5]
    versions = ["v5", "v6", "v7", "v8"]
    # versions = ["v6"]
    
    for exam_file in exam_files:
        for model_name in model_names:
            for task_domain in task_domains:
                for retriever_type in retriever_types:
                    for n_doc in n_documents:
                        for version in versions:
                            print(f"\nProcessing:")
                            # print(f"Model: {model_name}")
                            # print(f"Exam: {exam_file}")
                            # print(f"Task: {task_domain}")
                            # print(f"Retriever: {retriever_type}")
                            # print(f"k = {n_doc}")
                            # print(f"version: {version}")
                            
                            try:
                                if version == "v3":
                                    exam_file_name = exam_file.replace("signal_ratio_v2.json", "v5_signal_ratio_v2.json")
                                else:
                                    exam_file_name = exam_file.replace("signal_ratio_v2.json", f"{version}_signal_ratio_v2.json")
                                if retriever_type in ["closed", "open"]:
                                    result_file = exam_file.replace("signal_ratio_v2.json", f"{version}.json.json")
                                else:
                                    result_file = exam_file.replace("signal_ratio_v2.json", f"{version}.json_{n_doc}_results.json")
                                results_path = f"auto-rag-eval/MultiHopData/{task_domain}/exam_results/{model_name}_{retriever_type}_{result_file}"

                                print(results_path)
                                process_json_file(results_path, results_path)
                                print(f"Successfully processed {results_path}")
                                print(f"Output saved to {results_path}")
                            except Exception as e:
                                print(f"Failed to remove entities: {e}")
