import re
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from MultiHopData.prompt_template import PromptTemplate
from LLMServer.llama.llama_instant import ModelFactory, ModelType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class VerificationResult:
    """Data class to store verification results for a question"""
    required_chunks: List[int]
    reasoning: str
    confidence: float
    meets_requirement: bool


class ExamVerifier:
    def __init__(self, model_name: str):
        """
        Initialize the ExamVerifier with specified verification model.
        
        Args:
            model_name (str): Name of the model to use for verification
        """
        self.model_mapping = {
            'llama_3_1_8b': ModelType.LLAMA_3_1_8B,
            'llama_3_2_3b': ModelType.LLAMA_3_2_3B,
            'mistral_7b': ModelType.MISTRAL_7B,
            'ministral_8b': ModelType.MINISTRAL_8B,
            "gemma2_9b": ModelType.GEMMA2_9B,
        }
        
        # Convert to lowercase to handle case-insensitive input
        model_name = model_name.lower()
        
        if model_name not in self.model_mapping:
            raise ValueError(f"Unsupported model: {model_name}. "
                                f"Supported models: {list(self.model_mapping.keys())}")
            
        self.model_type = self.model_mapping[model_name]
        print(f"Using {self.model_type}")
        self.llm = ModelFactory.create_model(self.model_type)
        self.logger = logging.getLogger(__name__)

    def verify_exam(self, input_path: str, output_path: str) -> None:
        """
        Verify all questions in an exam file and save results.
        
        Args:
            input_path (str): Path to input exam JSON file
            output_path (str): Path to save verified exam JSON file
        """
        # Load exam data
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                exam_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load exam file: {e}")
            raise

        verified_questions = []
        for question in tqdm(exam_data):
            verification_result = self.verify_question(question)
            
            # Create verified question entry
            verified_question = {
                "question": question["question"],
                "choices": question["choices"],
                "correct_answer": question["correct_answer"],
                "documentation": question["documentation"],
                "final_verdict": {
                    "required_chunks": verification_result.required_chunks,
                    "reasoning": verification_result.reasoning,
                    "confidence": verification_result.confidence,
                    "meets_requirement": verification_result.meets_requirement
                }
            }
            verified_questions.append(verified_question)

        # Save verified exam
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(verified_questions, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Verified exam saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save verified exam: {e}")
            raise

    def verify_question(self, question_data: Dict) -> VerificationResult:
        """
        Verify a single question using the verification model.
        
        Args:
            question_data (Dict): Question data including documentation chunks
            
        Returns:
            VerificationResult: Verification results for the question
        """
        # Generate verification prompt
        prompt = self._create_verification_prompt(question_data)
        
        # Get verification response from model
        try:
            response = self.llm.invoke(prompt)
            verdict = self._extract_verdict(response)
            
            if verdict:
                return VerificationResult(
                    required_chunks=verdict["required_chunks"],
                    reasoning=verdict["reasoning"],
                    confidence=verdict.get("confidence", 0.0),
                    meets_requirement=len(verdict["required_chunks"]) >= len(question_data["documentation"])
                )
            else:
                self.logger.warning(f"Failed to extract verdict for question: {question_data['question'][:100]}...")
                return self._create_failed_verification()
        except Exception as e:
            self.logger.error(f"Error during verification: {e}")
            return self._create_failed_verification()

    def _create_verification_prompt(self, question_data: Dict) -> str:
        """Create verification prompt for the model."""
        chunk_dict = [
            {
                "text": question_data["documentation"],
            }
        ]
        
        verification_prompt = PromptTemplate.get_verification_prompt(self.model_type, question_data, chunk_dict)

        return verification_prompt

    def _create_failed_verification(self) -> VerificationResult:
        """Create a VerificationResult for failed verifications."""
        return VerificationResult(
            required_chunks=[],
            reasoning="Verification failed",
            confidence=0.0,
            meets_requirement=False
        )
    
    def extract_with_patterns(self, text: str, patterns: List) -> List[str]:

        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    return matches
            except re.error:
                continue
        return None
    
    def _extract_required_chunks(self, response: str) -> Optional[List[int]]:
        """Extract required_chunks from verdict response."""
        patterns = [
            r"\"required_chunks\":\s*\[([\d\s,]+)\]",
            r"required_chunks:\s*\[([\d\s,]+)\]",
            r"Required chunks:\s*\[([\d\s,]+)\]",
            r"Chunks needed:\s*\[([\d\s,]+)\]",
            r"Required chunks:[\s\n]*(\d+(?:,\s*\d+)*)",
            r"Chunks needed:[\s\n]*(\d+(?:,\s*\d+)*)",
        ]
        matches = self.extract_with_patterns(response, patterns)
        if matches:
            # Clean and parse the matched string
            chunk_str = matches[0].strip('[]').replace(' ', '')
            try:
                return [int(x) for x in chunk_str.split(',') if x]
            except ValueError:
                return None
        return None

    def _extract_reasoning(self, response: str) -> Optional[str]:
        """Extract reasoning from verdict response."""
        try:
        # First, try to parse as JSON
            try:
                # Try to parse the entire response as JSON
                json_data = json.loads(response)
                
                # Extract reasoning if it exists in a nested JSON structure
                if isinstance(json_data, dict):
                    # Check for 'reasoning' key at different levels
                    reasoning = json_data.get('reasoning')
                    if reasoning:
                        # If reasoning is a dictionary, convert to string
                        if isinstance(reasoning, dict):
                            return json.dumps(reasoning)
                        # If reasoning is already a string, return it
                        return str(reasoning)
            except json.JSONDecodeError:
                # If full JSON parsing fails, continue to regex methods
                pass

            # Define more comprehensive regex patterns
            patterns = [
                # JSON-style reasoning extraction
                r'"reasoning":\s*({[^}]+})',  # Capture full JSON object
                r'"reasoning":\s*"([^"]*)"',  # Quoted string reasoning
                r'"reasoning":\s*(\{[^}]+\})',  # Capture reasoning as JSON object
                
                # Plain text reasoning extraction
                r'Reasoning:\s*(.*?)(?=\n\s*[A-Z]|\n\s*\{|$)',
                r'reasoning:\s*(.*?)(?=\n\s*[a-z_"]+:|\n\s*\{|\n\s*\}|$)'
            ]
            
            # Try each pattern
            for pattern in patterns:
                matches = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
                if matches:
                    reasoning = matches[0]
                    
                    # Clean up the extracted reasoning
                    reasoning = reasoning.strip()
                    
                    # Remove surrounding quotes if present
                    reasoning = reasoning.strip('"')
                    
                    # Handle escaped characters
                    reasoning = reasoning.replace('\\"', '"').replace('\\\\', '\\')
                    
                    # If it looks like a JSON object, try to parse and reformat
                    try:
                        parsed_reasoning = json.loads(reasoning)
                        return json.dumps(parsed_reasoning, indent=2)
                    except (json.JSONDecodeError, TypeError):
                        # If not a valid JSON, return as-is
                        return reasoning

        except Exception as e:
            # Log the error or handle it as appropriate
            print(f"Error extracting reasoning: {e}")
        
        return None

    def _extract_confidence(self, response: str) -> Optional[int]:
        """Extract confidence score from verdict response."""
        patterns = [
            r"\"confidence\":\s*(\d+)",
            r"confidence:\s*(\d+)",
            r"Confidence:\s*(\d+)",
            r"Confidence level:\s*(\d+)",
        ]
        matches = self.extract_with_patterns(response, patterns)
        if matches:
            try:
                confidence = int(matches[0])
                return confidence if 1 <= confidence <= 5 else None
            except ValueError:
                return None
        return None

    def _extract_verdict(self, response: str) -> Dict:
        """Extract all verdict components using pattern matching."""
        verdict = {
            "required_chunks": self._extract_required_chunks(response),
            "reasoning": self._extract_reasoning(response),
            "confidence": self._extract_confidence(response)
        }
        
        # Validate that we have at least the critical fields
        if (verdict["required_chunks"] is not None and
            verdict["reasoning"] is not None):
            return verdict
        return None

def main():
    model_names = [
        'llama_3_2_3b',
        "ministral_8b",
        "gemma2_9b",
    ]
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    exams = [
        # "exam_new_llama_3_2_3b_processed_v2.json",
        # "exam_new_ministral_8b_processed_v2.json",
        # "exam_new_gemma2_9b_processed_v2.json",
        # v1 v2 diff
        "exam_new_llama_3_2_3b_v1_v2_diff.json",
        "exam_new_ministral_8b_v1_v2_diff.json",
        "exam_new_gemma2_9b_v1_v2_diff.json",
        ]
    
    for model_name in model_names:
        for task_domain in task_domains:
            for exam in exams:
                input_path = f"MultiHopData/{task_domain}/exams/{exam}"
                output_path = f"MultiHopData/{task_domain}/exam_verification/{model_name}/{exam}"

                verifier = ExamVerifier(model_name=model_name)
                try:
                    verifier.verify_exam(input_path, output_path)
                except Exception as e:
                    logging.error(f"Verification failed: {e}")

if __name__ == "__main__":
    main()
