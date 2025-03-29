import json
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass

from LLMServer.llama_instant import ModelFactory, ModelType
from MultiHopData.Solver.solve_exam_rag import ExamQuestion, ExamSolver
from LLMServer.llama_instant import ModelFactory, ModelType

class ChunkAnalyser:
    def __init__(self):
        """Initialise chunk analysis."""
        self.llm = ModelFactory.create_model(ModelType.GEMMA2_9B)
    
    def categorise_error_type(self, question_data: dict, exam_taker_reasoning: str) -> Dict[str, bool]:
        """Categorize error type based on the question data and student reasoning."""
        options = [choice.split(') ')[1] for choice in question_data['choices']]
        
        chunk_dict = [{"text": question_data["documentation"]}]
        chunk_text = '\n'.join([f"Chunk{i+1}: {chunk['text']}" for i, chunk in enumerate(chunk_dict)])
        
        question = question_data['question']
        option_a = options[0]
        option_b = options[1]
        option_c = options[2]
        option_d = options[3]
        correct_answer = question_data['correct_answer']
        
        prompt = f"""<start_of_turn>user
        You are an expert evaluator analyzing errors in multi-hop question answering systems.
        Your task is to perform a detailed error analysis using a hierarchical classification system.

        Primary Error Categories:

        1. Information Processing Errors
        A. Integration Failures
            - Incorrect combination of retrieved facts
            - Failure to establish relationships between facts
            - Missing key connections between pieces of information
        
        B. Reasoning Failures
            - Invalid logical inference steps
            - Incorrect causality assumptions
            - Faulty deductive or inductive reasoning

        2. Knowledge Boundary Errors
        A. Context Utilization
            - Missing or overlooking relevant information
            - Using irrelevant or misleading information
            - Failure to identify key supporting evidence
        
        B. Knowledge Scope
            - Making assumptions beyond given information
            - Introducing external knowledge inappropriately
            - Misinterpreting information boundaries

        Analysis Instructions:
        1. Identify error chains by tracing the progression of mistakes
        2. If none of the categories are the right category, return "other"
        3. For each error identified, provide:
        - Evidence from the exam taker reasoning
        - Specific text from source documents
        - Explanation of how the error led to the incorrect answer

        Output Format:
        Return strictly only json format as specified below and do not return other explanations.
        {{
        "error_chain": {{
            "primary_error": {{
            "category": string,  // Main category from hierarchy
            "subcategory": string,  // Specific error type either A or B in the category
            "confidence": int,  // 1-5 scale
            "evidence": {{
                "exam_taker_reasoning": string,  // Relevant quote
                "source_document": string,  // Relevant quote
                "explanation": string
            }}
            }},
            "contributing_errors": [
            {{
                // Same structure as primary_error
            }}
            ]
        }},
        "quantitative_indicators": {{
            "information_overlap": float,  // Semantic similarity with source
            "reasoning_steps_identified": int,
            "external_knowledge_ratio": float  // Proportion of unsupported claims
        }}
        }}

        Input:
        Question: {question}
        Options:
        A) {option_a}
        B) {option_b}
        C) {option_c}
        D) {option_d}
        Correct Answer: {correct_answer}
        Documents: {chunk_text}
        Exam Taker Reasoning: {exam_taker_reasoning}
        <end_of_turn>
        <start_of_turn>model"""
        
        response = self.llm.invoke(prompt)
        
        try:
            return json.loads(response.replace('```json\n', '').replace('\n```', ''))
        except Exception as e:
            print(f"WARNING JSON load error: {e}")
            return {
                "classifications": False,
                "primary_error": False,
                "primary_error_confidence": False
            }

class ExamMistakeClassifier:
    def __init__(self, chunk_analyser: ChunkAnalyser):
        """Initialise the classifier with a ChunkAnalyser instance."""
        self.chunk_analyser = chunk_analyser
        self.solver = ExamSolver()
    
    def check_error_classification(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        has_error_classification = any(
            'error_classification' in item 
            for item in data 
            if item.get('is_correct') is False
        )
        
        return has_error_classification

    def process_exam_results(self, exam_path: str, result_path: str, model, rag_type, CoT, self_ask) -> List[Dict[str, Any]]:
        """Process exam results and classify mistakes for failed questions.
        
        Args:
            exam_path: Path to the original exam file
            result_path: Path to the exam results file
            
        Returns:
            Updated exam results with mistake classifications
        """
        # Load exam questions and results
        with open(exam_path, 'r', encoding='utf-8') as f:
            exam_data = json.load(f)
        
        with open(result_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        if not self.check_error_classification(result_path):
            # Create question mapping for easier lookup
            question_map = {q['question']: q for q in exam_data}
            
            # Process each failed question
            for result in tqdm(result_data):
                if not result.get('is_correct', True):
                    # Get original question data
                    question_data = question_map.get(result['question'])
                    if not question_data:
                        continue
                    
                    # Create ExamQuestion object
                    exam_question = ExamQuestion(
                        question=question_data['question'],
                        choices=question_data['choices'],
                        correct_answer=question_data['correct_answer'],
                        documentation=question_data.get('documentation', [])
                    )
                    
                    # Get reasoning for the failed question
                    classification_model = ModelFactory.create_model(ModelType.GEMMA2_9B)
                    if CoT:
                        reasoning = self.solver.solve_question(exam_question, model, rag_type, pos_swap=False, CoT=CoT, self_ask=self_ask, error_category=True)
                    elif self_ask:
                        reasoning = self.solver.solve_question(exam_question, model, rag_type, pos_swap=False, CoT=CoT, self_ask=self_ask, error_category=True)
                    else:
                        reasoning = self.solver.solve_question_with_reasoning(exam_question, model)
                    
                    # Get error classification
                    classification = self.chunk_analyser.categorise_error_type(question_data, reasoning)
                    
                    # Update result with classification
                    result['error_classification'] = classification
            
            return result_data
        else:
            return result_data

    def save_results(self, results: List[Dict[str, Any]], filepath: str):
        """Save the processed results to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

def main(exam_path: str, result_path: str, output_path: str, model_type: str, rag_type: str):
    """Main function to run the mistake classification process.
    
    Args:
        exam_path: Path to the original exam file
        result_path: Path to the exam results file
        output_path: Path where the updated results should be saved
    """
    # Initialise components
    chunk_analyser = ChunkAnalyser()
    classifier = ExamMistakeClassifier(chunk_analyser)
    
    if model_type == "gemini":
        model = GeminiGcp(model_name=model_name)
    elif model_type == "claude":
        model = ClaudeGcp(model_name=model_name)
    elif model_type == "cpp":
        model_mapping = {
            'llama_3_1_8b': ModelType.LLAMA_3_1_8B,
            'llama_3_2_3b': ModelType.LLAMA_3_2_3B,
            'ministral_8b': ModelType.MINISTRAL_8B,
            'mistral-small': ModelType.MISTRAL_SMALL,
            'mixtral-8-7b': ModelType.MIXTRAL_8_7B,
            "gemma2_9b": ModelType.GEMMA2_9B,
            "gemma2-27b": ModelType.GEMMA2_27B
        }
        print(f"Using {model_mapping[model_name]}")
        model = ModelFactory.create_model(model_mapping[model_name])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Process exam results
    updated_results = classifier.process_exam_results(exam_path, result_path, model, rag_type, CoT, self_ask)
    
    # Save updated results
    classifier.save_results(updated_results, output_path)
    print(f"Updated results saved to: {output_path}")


if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    model_names = [
        'llama_3_1_8b',
        "gemma2_9b",
        "ministral_8b",
    ]
    rag_types = ["Sparse"]
    exams = [
        # V3
        "exam_new_llama_3_2_3b_processed_v3.json",
        "exam_new_ministral_8b_processed_v3.json",
        "exam_new_gemma2_9b_processed_v3.json",
        # V5
        "exam_new_llama_3_2_3b_processed_v5.json",
        "exam_new_ministral_8b_processed_v5.json",
        "exam_new_gemma2_9b_processed_v5.json",
        # V8
        "exam_new_llama_3_2_3b_processed_v8.json",
        "exam_new_ministral_8b_processed_v8.json",
        "exam_new_gemma2_9b_processed_v8.json",
        # V6
        "exam_new_llama_3_2_3b_processed_v6.json",
        "exam_new_ministral_8b_processed_v6.json",
        "exam_new_gemma2_9b_processed_v6.json",
        # V7
        "exam_new_llama_3_2_3b_processed_v7.json",
        "exam_new_ministral_8b_processed_v7.json",
        "exam_new_gemma2_9b_processed_v7.json",
        V9
        "exam_new_llama_3_2_3b_processed_v9.json",
        "exam_new_ministral_8b_processed_v9.json",
        "exam_new_gemma2_9b_processed_v9.json",
        V10
        "exam_new_llama_3_2_3b_processed_v10.json",
        "exam_new_ministral_8b_processed_v10.json",
        "exam_new_gemma2_9b_processed_v10.json",
    ]
    internal_prevention = False
    CoT = True
    self_ask = False
    
    for task_domain in task_domains:
        for model_name in model_names:
            for exam in exams:
                for rag_type in rag_types:
                    # Construct paths
                    if internal_prevention:
                        exam_result_path = f"MultiHopData/{task_domain}/exam_results/{model_name}_open_{exam}_internal_prevention.json"
                    elif CoT:
                        exam_result_path = f"MultiHopData/{task_domain}/exam_results/{model_name}_{rag_type}_{exam}_5_results_CoT.json"
                    elif self_ask:
                        exam_result_path = f"MultiHopData/{task_domain}/exam_results/{model_name}_{rag_type}_{exam}_5_self_ask.json"
                    else:
                        exam_result_path = f"MultiHopData/{task_domain}/exam_results/{model_name}_{rag_type}_{exam}_5_results.json"
                    input_exam_path = f"MultiHopData/{task_domain}/exams/{exam}"
                    exam_output_name = exam_result_path
                    
                    print(f"Processing: {exam_result_path} - {model_name} - {exam}")
                    # Run classification
                    try:
                        main(input_exam_path, exam_result_path, exam_output_name, "cpp", rag_type)
                    except Exception as e:
                        print(f"WARNING a file does not exist: {e}")
