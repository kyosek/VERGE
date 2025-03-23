import json
from typing import Dict, List, Tuple
import pandas as pd
from collections import defaultdict


class VerificationAnalyzer:
    def __init__(self, weights: Dict[str, float]):
        """Initialize the analyzer with model weights.
        
        Args:
            weights: Dictionary mapping model names to their weights
                    e.g., {'gemma': 0.5, 'llama': 0.25, 'mistral': 0.25}
        """
        self.weights = weights
        self.threshold = 0.5
        
    def _extract_meets_requirement(self, question_data: Dict) -> bool:
        """Extract meets_requirement from different JSON structures with debugging.
        
        Args:
            question_data: Dictionary containing question verification data
            
        Returns:
            Boolean indicating if the question meets requirements
        """
        # First check metadata
        if "metadata" in question_data:
            metadata = question_data["metadata"]
            if isinstance(metadata, dict) and "meets_hop_requirement" in metadata:
                return bool(metadata["meets_hop_requirement"])
        
        # Then check final_verdict
        if "final_verdict" in question_data:
            final_verdict = question_data["final_verdict"]
            if isinstance(final_verdict, dict) and "meets_requirement" in final_verdict:
                return bool(final_verdict["meets_requirement"])
        
            
        # If not found in any location, log and return False
        print(f"Warning: Could not find meets_requirement in question: {question_data.get('question', '')[:100]}...")
        return False
    
    def load_verifications(self, file_paths: Dict[str, str]) -> Dict[str, List[Dict]]:
        """Load verification results from JSON files with better error handling.
        
        Args:
            file_paths: Dictionary mapping model names to file paths
        
        Returns:
            Dictionary mapping model names to their verification results
        """
        verifications = {}
        for model, path in file_paths.items():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    processed_data = []
                    for question in data:
                        meets_req = self._extract_meets_requirement(question)
                        processed_question = {
                            'question': question['question'],
                            'choices': question['choices'],
                            'correct_answer': question['correct_answer'],
                            'meets_requirement': meets_req
                        }
                        processed_data.append(processed_question)
                    verifications[model] = processed_data
                print(f"Processed {len(processed_data)} questions for {model}")
                print(f"Accepted questions: {sum(1 for q in processed_data if q['meets_requirement'])}")
            except Exception as e:
                print(f"Error processing {model} verification file: {str(e)}")
                raise
        return verifications
    
    def compute_weighted_vote(self, question_results: Dict[str, bool]) -> float:
        """Compute weighted vote for a question across models."""
        score = 0.0
        for model, meets_req in question_results.items():
            score += self.weights[model] * float(meets_req)
        return score
    
    def analyze_questions(self, verifications: Dict[str, List[Dict]]) -> Tuple[List[Dict], Dict]:
        """Analyze verification results across models.
        
        Returns:
            List of question analysis results and summary statistics
        """
        question_verdicts = defaultdict(dict)
        
        # Process each model's verifications
        for model, results in verifications.items():
            for question in results:
                q_id = question['question']
                question_verdicts[q_id][model] = question['meets_requirement']
        
        # Compute weighted votes and final decisions
        analysis_results = []
        for q_id, model_results in question_verdicts.items():
            weighted_vote = self.compute_weighted_vote(model_results)
            keep_question = weighted_vote >= self.threshold
            
            result = {
                'question': q_id,
                'weighted_vote': weighted_vote,
                'keep_question': keep_question,
                'model_verdicts': {
                    model: verdict for model, verdict in model_results.items()
                }
            }
            analysis_results.append(result)
        
        # Compute summary statistics
        summary = {
            'total_questions': len(analysis_results),
            'questions_kept': sum(1 for r in analysis_results if r['keep_question']),
            'questions_rejected': sum(1 for r in analysis_results if not r['keep_question']),
            'model_agreement_rate': self._compute_agreement_rate(analysis_results),
            'model_specific_stats': self._compute_model_stats(analysis_results)
        }
        
        return analysis_results, summary
    
    def _compute_agreement_rate(self, results: List[Dict]) -> float:
        """Compute the rate at which all models agree."""
        total_agreement = 0
        for result in results:
            verdicts = list(result['model_verdicts'].values())
            if all(verdicts) or not any(verdicts):
                total_agreement += 1
        return total_agreement / len(results) if results else 0
    
    def _compute_model_stats(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Compute statistics for each model."""
        model_counts = defaultdict(lambda: {'accepted': 0, 'total': 0})
        
        for result in results:
            for model, verdict in result['model_verdicts'].items():
                model_counts[model]['total'] += 1
                if verdict:
                    model_counts[model]['accepted'] += 1
        
        stats = {}
        for model, counts in model_counts.items():
            total = counts['total']
            accepted = counts['accepted']
            stats[model] = {
                'acceptance_rate': accepted / total if total > 0 else 0,
                'rejection_rate': (total - accepted) / total if total > 0 else 0
            }
        
        return stats


def main(gemma_path: str, llama_path: str, ministral_path: str, output_path: str):
    # Define model weights
    weights = {
        'gemma': 0.5,
        'llama': 0.25,
        'ministral': 0.25
    }
    
    # Define file paths
    file_paths = {
        'llama': llama_path,
        'gemma': gemma_path,
        'ministral': ministral_path
    }
    
    # Initialize analyzer
    analyzer = VerificationAnalyzer(weights)
    
    # Load and analyze verifications
    try:
        verifications = analyzer.load_verifications(file_paths)
        results, summary = analyzer.analyze_questions(verifications)
        
        output = {
            'summary': summary,
            'question_results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Print summary statistics
        print("\nVerification Analysis Summary:")
        print(f"Total Questions: {summary['total_questions']}")
        print(f"Questions Kept: {summary['questions_kept']} ({summary['questions_kept']/summary['total_questions']*100:.1f}%)")
        print(f"Questions Rejected: {summary['questions_rejected']} ({summary['questions_rejected']/summary['total_questions']*100:.1f}%)")
        print(f"Model Agreement Rate: {summary['model_agreement_rate']*100:.1f}%")
        
        print("\nModel-Specific Statistics:")
        for model, stats in summary['model_specific_stats'].items():
            print(f"\n{model.capitalize()}:")
            print(f"Acceptance Rate: {stats['acceptance_rate']*100:.1f}%")
            print(f"Rejection Rate: {stats['rejection_rate']*100:.1f}%")

        print(f"\nResults saved to '{output_path}'")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    exams = [
        # "llama_3_2_3b",
        "ministral_8b",
        # "gemma2_9b",
        ]
    
    for task in task_domains:
        for exam in exams:
            # gemma_path = f"auto-rag-eval/MultiHopData/{task}/exam_verification/gemma2-9b/exam_new_{exam}_processed_v2.json"
            # llama_path = f"auto-rag-eval/MultiHopData/{task}/exam_verification/llama_3_2_3b/exam_new_{exam}_processed_v2.json"
            # # llama_path = f"auto-rag-eval/MultiHopData/{task}/exams/exam_new_{exam}_processed_v2.json"
            # # ministral_path = f"auto-rag-eval/MultiHopData/{task}/exam_verification/ministral_8b/exam_new_{exam}_processed_v2.json"
            # ministral_path = f"auto-rag-eval/MultiHopData/{task}/exams/exam_new_{exam}_processed_v2.json"
            # output_path = f"auto-rag-eval/MultiHopData/{task}/exam_verification/new_exam_{exam}_verified.json"
            # v1 v2 diff
            gemma_path = f"auto-rag-eval/MultiHopData/{task}/exam_verification/gemma2-9b/exam_new_{exam}_v1_v2.json"
            llama_path = f"auto-rag-eval/MultiHopData/{task}/exam_verification/llama_3_2_3b/exam_new_{exam}_v1_v2.json"
            ministral_path = f"auto-rag-eval/MultiHopData/{task}/exam_verification/ministral_8b/exam_new_{exam}_v1_v2.json"
            output_path = f"auto-rag-eval/MultiHopData/{task}/exam_verification/new_exam_{exam}_v1_v2_diff_verified.json"
    
            main(gemma_path, llama_path, ministral_path, output_path)
