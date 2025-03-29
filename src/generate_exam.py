import json
import re
import numpy as np
import faiss
import random
import os
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import ast
from tqdm import tqdm
from llama_cpp import Llama

from MultiHopData.prompt_template import PromptTemplate
from MultiHopData.retriever import Chunk, ChunkRetriever, HybridChunkRetriever
from LLMServer.llama_instant import ModelFactory, ModelType


class MCQGenerator:
    def __init__(self, model_name: str = None):
        """
        Initialise the MCQ Generator with a specific model.
        
        Args:
            model_name (str, optional): Name of the model to use.
            If None, uses a default model.
        """
        # Mapping of model names to ModelType enums
        self.model_mapping = {
            'llama_3_1_8b': ModelType.LLAMA_3_1_8B,
            'llama_3_2_3b': ModelType.LLAMA_3_2_3B,
            'mistral_7b': ModelType.MISTRAL_7B,
            'ministral_8b': ModelType.MINISTRAL_8B,
            "gemma2_9b": ModelType.GEMMA2_9B,
        }
        
        # Select model based on input or use default
        if model_name:
            # Convert to lowercase to handle case-insensitive input
            model_name = model_name.lower()
            
            if model_name not in self.model_mapping:
                raise ValueError(f"Unsupported model: {model_name}. "
                                 f"Supported models: {list(self.model_mapping.keys())}")
            
            self.model_type = self.model_mapping[model_name]
        else:
            # Default model if no name provided
            self.model_type = ModelType.LLAMA_3_2_3B
        
        print(f"Using {self.model_type}")
        self.llm = ModelFactory.create_model(self.model_type)
        self.verification_llm = ModelFactory.create_model(ModelType.GEMMA2_9B)
    
    def extract_with_patterns(self, text: str, patterns: List) -> List[str]:

        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    return matches
            except re.error:
                continue
        return None

    def _extract_question(self, response: str) -> Optional[str]:
        """Extract question from response with improved pattern matching."""
        question_patterns = [
            r"\*\*Question\*\*\n\n(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
            r"Question:\s*(.*?)(?=\nChoices:|\n[A-D]\)|\n\n[a-dA-D1-4]\))",
            r"Question:\s*(.*?)(?=\n\n|\n[A-D]\)|\Z)",
            r"Question:\s*(.+?)(?=\n|$)",
            r"Question:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
            r"Question 1:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
            r"question:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
            r"question 1:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
            r"documentation:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
            r"### Assistant: (.*?)\n",
        ]
        # Extract the question
        question_matches = self.extract_with_patterns(response, question_patterns)
        question = question_matches[0].strip() if question_matches else None
        return question

    def _extract_choices(self, response: str) -> Optional[List[str]]:
        """
        Extract and validate multiple choice answers with improved multi-line handling.
        
        Args:
            response (str): The raw response text containing the MCQ
            
        Returns:
            Optional[List[str]]: List of choices if found and valid, None otherwise
        """
        list_match = re.search(r"Choices:\s*(\[.*?\])", response, re.DOTALL)
    
        if list_match:
            try:
                # Use ast.literal_eval to safely parse the list string
                choices_list = ast.literal_eval(list_match.group(1))
                
                # Validate the choices
                if (len(choices_list) == 4 and 
                    all(isinstance(choice, str) and choice.startswith(letter+')') 
                        for choice, letter in zip(choices_list, ['A', 'B', 'C', 'D']))):
                    return choices_list
            except (ValueError, SyntaxError):
                pass

        # Try different patterns in order of specificity
        patterns = [
            # Basic pattern for lettered choices with parentheses
            r'(?:^|\n)([A-D]\))\s*(.*?)(?=\n[A-D]\)|$)',
            
            # Pattern for choices with newlines and content
            r'\n([A-D]\))\s*((?:(?!\n[A-D]\)).)*)',
            
            # Pattern for full answers with possible multiline content
            r'(?:^|\n)([A-D]\))\s*((?:(?!\n[A-D]\)|Correct Answer:).)*)',
            r'([A-D])\)\s*(.*?)(?=\n[A-D]\)|Correct Answer:|\Z)',
            r'([A-D])\)\s*((?:(?!\n[A-D]\)|Correct Answer:).)*)',
            
            # Fallback pattern for simple format
            r'([A-D]\))\s*([^\n]+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.MULTILINE | re.DOTALL)
            choices = []
            
            for match in matches:
                identifier = match.group(1)
                content = match.group(2).strip() if len(match.groups()) > 1 else ''
                full_choice = f"{identifier} {content}"
                # Clean up any extra whitespace
                full_choice = ' '.join(full_choice.split())
                choices.append(full_choice)
            
            choices = (
                choices[:4]
                if choices
                and len(choices) >= 4
                and len(set([choice[0] for choice in choices[:4]])) == 4
                else None
            )
            # Validate we have exactly 4 choices
            if len(choices) == 4 and len(set([c[0] for c in choices])) == 4:
                return choices
        
        return None

    def _extract_correct_answer(self, response: str) -> Optional[str]:
        """Extract correct answer with validation."""
        # Try first with full pattern including 'Correct Answer:'
        correct_answer_match = re.search(r"Correct Answer:\s*([A-D])\)?", response, re.IGNORECASE)
        
        # If first method fails, try a more lenient approach
        if not correct_answer_match:
            correct_answer_match = re.search(r"\*\*Correct Answer:\*\*\s*([A-D])\)?", response, re.IGNORECASE)
        
        # If still no match, try finding a lone capital letter at the end
        if not correct_answer_match:
            correct_answer_match = re.search(r"([A-D])$", response.split('\n')[-1].strip(), re.IGNORECASE)
        
        # Return the correct answer if found
        if correct_answer_match:
            return f"{correct_answer_match.group(1)})"
        
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
    
    def _make_enhanced_question_prompt(self, task_domain: str, chunks: List[Dict[str, str]]) -> str:
        """Create a prompt using the appropriate template for the specified model."""
        documentation = "\n\n".join([f"Chunk{i}: {chunk['text']}" for i, chunk in enumerate(chunks)])
        
        template = PromptTemplate.get_question_generation_prompt_template(self.model_type, chunks, task_domain, documentation)
        return template
        
    def generate_question(self, chunks: List[Dict[str, str]], task_domain: str) -> Optional[Dict]:
        """Generate a multiple-choice question with documentation included."""
        # Analyse chunks
        
        # Generate question with enhanced prompt
        prompt = self._make_enhanced_question_prompt(
            task_domain=task_domain,
            chunks=chunks,
        )
        
        response = self.llm.invoke(prompt)
        
        try:
            # Create the question dictionary with documentation
            parsed_question = {
                "question": self._extract_question(response),
                "choices": self._extract_choices(response),
                "correct_answer": self._extract_correct_answer(response),
                "documentation": [chunk["text"] for chunk in chunks],
                "metadata": {"num_chunks_used": len(chunks)}
            }
            
            return parsed_question
            
        except Exception as e:
            logging.error(f"Error parsing question format: {e}")
            return None

    def _regenerate_question_with_feedback(
        self,
        question_data: dict,
        feedback: str,
        ) -> Optional[Dict]:
        """Regenerate question using verification feedback."""
        prompt = PromptTemplate.get_regenerate_question_prompt(self.model_type, question_data, feedback)
        
        response = self.llm.invoke(prompt)
        
        try:
            return {
                "question": self._extract_question(response),
                "choices": self._extract_choices(response),
                "correct_answer": self._extract_correct_answer(response),
                "documentation": question_data["documentation"],
                "metadata": {
                    "num_chunks_used": len(question_data["documentation"])
                }
            }
        except Exception as e:
            logging.error(f"Error parsing regenerated question: {e}")
            return None

    def call_verification_agent(self, question_data: dict, chunks: List[Dict[str, str]], 
                    task_domain: str, target_hops: int, max_attempts: int = 3) -> Dict:
        """
        Verify and potentially regenerate the question to ensure it requires the target number of hops.
        
        Args:
            question_data: The generated question data
            chunks: List of document chunks
            task_domain: Domain of the task
            target_hops: Target number of hops required
            
        Returns:
            Dict containing the final question data with verification metadata
        """
        verification_attempts = 0
        current_question = question_data
        verdicts = []
        
        while verification_attempts < max_attempts:
            # Generate verification prompt
            verification_prompt = PromptTemplate.get_verification_prompt(self.model_type, current_question, chunks)
            
            # Get verdict from LLM
            verdict_response = self.verification_llm.invoke(verification_prompt)
            verdict = self._extract_verdict(verdict_response)
            if verdict:
                verdicts.append(verdict)
                
                # Check if the question meets the hop requirement
                if len(verdict['required_chunks']) >= target_hops:
                    break
                
                # Regenerate question with feedback
                verification_attempts += 1
                if verification_attempts < max_attempts:
                    regenerated_question = self._regenerate_question_with_feedback(
                        question_data=question_data,
                        feedback=verdict['reasoning']
                    )
                    if regenerated_question:
                        current_question = regenerated_question
            else:
                verification_attempts += 1
                logging.error("Failed to extract verdict from response")
        
        # Add verification metadata to the final question
        current_question['metadata'].update({
            'verification_attempts': verification_attempts,
            'verification_verdicts': verdicts,
            'final_verdict': verdicts[-1] if verdicts else None,
            'meets_hop_requirement': (len(verdicts[-1]['required_chunks']) >= target_hops) & (verdicts[-1]["reasoning"]["shortcut_reasoning_risk"]) & (verdicts[-1]["if_solvable"]) if verdicts else False
        })
        
        return current_question

def generate_exam(
    data: List[Dict[str, str]],
    task_domain: str,
    model_name: str,
    retriever: ChunkRetriever,
    target_hop_number: int = 250,
) -> List[Dict[str, str]]:
    """
    Generate an exam with multiple-choice questions from the given data.
    """
    mcq_generator = MCQGenerator(model_name)
    num_questions = len(data)
    exam = []
    hop_counts = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
    }

    for ith_question in tqdm(range(0, num_questions)):
        # Get the current chunk and its similar chunks
        current_chunk = data[ith_question]
        chunk_data = Chunk(
            chunk_id=current_chunk.chunk_id,
            doc_id=current_chunk.doc_id,
            content=current_chunk.content,
            original_index=ith_question,
        )
        
        hop_try_count = 0
        while True:
            num_hops = random.randint(1, 4)
            if hop_counts[str(num_hops)] < target_hop_number:
                break
            hop_try_count += 1
            if hop_try_count > 3:
                break
        
        try:
            similar_chunks = retriever.find_similar_chunks(
                chunk_data, k=num_hops, similarity_threshold=0.01, exclude_same_doc=False
            )
            
            hop_counts[str(len(similar_chunks))] += 1

            chunk_dict = [
                {
                    "chunk_id": current_chunk.chunk_id,
                    "doc_id": current_chunk.doc_id,
                    "text": current_chunk.content,
                }
            ]
            chunk_dict += [
                {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.content}
                for c, _ in similar_chunks
            ]

            try:
                question_data = mcq_generator.generate_question(chunk_dict, task_domain)
                if question_data:
                    question_data = mcq_generator.call_verification_agent(
                        question_data=question_data,
                        chunks=chunk_dict,
                        task_domain=task_domain,
                        target_hops=len(similar_chunks) + 1  # +1 because we include the original chunk
                        )
                    exam.append(question_data)
            except Exception as e:
                logging.error(f"Error generating question: {e}")
                continue
        except Exception as e:
            logging.error(f"No similar chunks: {e}")
            continue

    return exam


def main(
    data_path: str,
    output_path: str,
    model_name: str,
    task_domain: str,
    sample_size: int,
    version: str,
    target_hop_number: int = 250
):
    logging.info("Start processing")
    retriever = HybridChunkRetriever(task_domain, random_seed=42)

    if not os.path.exists(f"MultiHopData/{task_domain}/chunk_database_{version}"):
        logging.info("Load documents")
        retriever.load_documents(data_path)
        logging.info(f"Save the database to 'MultiHopData/{task_domain}/chunk_database_{version}'")
        retriever.save_database(f"MultiHopData/{task_domain}/chunk_database_{version}")
    else:
        logging.info("Loading database from file")
        retriever = HybridChunkRetriever.load_database(
            f"MultiHopData/{task_domain}/chunk_database_{version}", task_domain
        )

    # Sample chunks with a specific seed
    sampled_chunks = retriever.sample_chunks(sample_size, seed=42)

    # Generate the exam
    print("Start generating the exam")
    exam = generate_exam(
        sampled_chunks,
        task_domain,
        model_name,
        retriever,
        target_hop_number
    )

    # Save the exam to a JSON file
    with open(output_path, "w") as f:
        json.dump(exam, f, indent=2)


if __name__ == "__main__":
    sample_size = 700
    target_hop_number = 176
    
    assert sample_size < target_hop_number * 4
    
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    # task_domains = ["multifieldqa_en"]
    
    model_names = ["llama_3_2_3b", "gemma2_9b", 'ministral_8b']
    
    versions = ["v1"]
    
    # task_domain = "gov_report"
    for model_name in model_names:
        for task_domain in task_domains:
            for version in versions:
                data_path = f"MultiHopData/{task_domain}/chunks/docs_chunk_semantic_{version}_cleaned.json"
                output_path = f"MultiHopData/{task_domain}/exams/exam_new_{model_name}_{version}.json"

                main(
                    data_path,
                    output_path,
                    model_name,
                    task_domain,
                    sample_size,
                    version,
                    target_hop_number=target_hop_number
                )
