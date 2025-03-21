import json
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExamUpdater:
    def __init__(self, original_exam_path: str, verified_exam_path: str):
        self.original_exam_path = Path(original_exam_path)
        self.verified_exam_path = Path(verified_exam_path)
        
    def load_json_file(self, file_path: Path) -> List[Dict]:
        """Load and parse a JSON file."""
        try:
            with file_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
            
    def find_verified_questions(self, verified_exam: List[Dict]) -> set:
        """Extract questions that passed verification."""
        verified_questions = set()
        for entry in verified_exam:
            try:
                if entry.get('final_verdict', {}).get('meets_requirement'):
                    verified_questions.add(entry['question'])
            except KeyError as e:
                logger.warning(f"Missing expected key in verified exam entry: {e}")
        return verified_questions
        
    def filter_original_exam(self, original_exam: List[Dict], 
                           verified_questions: set) -> List[Dict]:
        """Filter original exam to keep only verified questions."""
        filtered_exam = []
        for entry in original_exam:
            if entry['question'] in verified_questions:
                filtered_exam.append(entry)
        return filtered_exam
        
    def save_updated_exam(self, updated_exam: List[Dict], output_path: str) -> None:
        """Save the updated exam to a new file."""
        try:
            with Path(output_path).open('w', encoding='utf-8') as f:
                json.dump(updated_exam, f, indent=2, ensure_ascii=False)
                logger.info(f"Updated exam saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving updated exam: {e}")
            raise
            
    def process(self, output_path: str) -> None:
        """Process the exam update pipeline."""
        logger.info("Loading exam files...")
        original_exam = self.load_json_file(self.original_exam_path)
        verified_exam = self.load_json_file(self.verified_exam_path)
        
        logger.info("Finding verified questions...")
        verified_questions = self.find_verified_questions(verified_exam)
        logger.info(f"Found {len(verified_questions)} verified questions")
        
        logger.info("Filtering original exam...")
        updated_exam = self.filter_original_exam(original_exam, verified_questions)
        logger.info(f"Filtered exam contains {len(updated_exam)} questions")
        
        logger.info("Saving updated exam...")
        self.save_updated_exam(updated_exam, output_path)
        


def main(task_domain, exam):
    original_exam_path = f"auto-rag-eval/MultiHopData/{task_domain}/exams/{exam}"
    verified_exam_path = f"auto-rag-eval/MultiHopData/{task_domain}/exam_verification/{task_domain}/{exam}"
    updated_exam_str = exam.replace("v2", "v3")
    output_path = f"auto-rag-eval/MultiHopData/{task_domain}/exams/{updated_exam_str}"

    updater = ExamUpdater(
        original_exam_path,
        verified_exam_path
    )
    try:
        updater.process(output_path)
        logger.info("Exam update completed successfully")
    except Exception as e:
        logger.error(f"Failed to update exam: {e}")
        raise


if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    exams = [
        "exam_new_llama_3_2_3b_processed_v2.json",
        "exam_new_ministral_8b_processed_v2.json",
        # "exam_new_gemma2_9b_processed_v2.json",
        ]
    
    for task_domain in task_domains:
        for exam in exams:
            main(task_domain, exam)
