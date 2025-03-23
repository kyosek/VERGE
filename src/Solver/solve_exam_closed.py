import re
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import json
import torch
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize

from MultiHopData.retriever import Chunk, ChunkRetriever
from LLMServer.llama_gcp.llama_gcp_instant import LlamaGcpModel
from LLMServer.gcp.claude_instant import ClaudeGcp
from LLMServer.gcp.gemini_instant import GeminiGcp
from LLMServer.llama_instant import ModelFactory, ModelType


@dataclass
class ExamQuestion:
    question: str
    choices: List[str]
    correct_answer: str
    documentation: List[str]


class BaseRetriever(ABC):
    """Abstract base class for different retrieval methods."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve relevant documents for a query."""
        pass


class FAISSRetriever(BaseRetriever):
    """Dense retrieval using FAISS."""

    def __init__(self, chunk_retriever: "ChunkRetriever"):
        self.chunk_retriever = chunk_retriever

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        # Create a temporary chunk for the query
        query_chunk = Chunk(chunk_id="query", doc_id="query", content=query, original_index=-1)

        # Use the existing chunk retriever to find similar chunks
        similar_chunks = self.chunk_retriever.find_similar_chunks(
            query_chunk, k=k, exclude_same_doc=False
        )

        return [(chunk.content, score) for chunk, score in similar_chunks]


class BM25Retriever(BaseRetriever):
    """Sparse retrieval using BM25."""

    def __init__(self, documents: List[str]):
        # Tokenize documents
        tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[-k:][::-1]

        return [(self.documents[i], scores[i]) for i in top_k_indices]


class HybridRetriever(BaseRetriever):
    """Combines multiple retrievers with optional weights."""

    def __init__(self, retrievers: List[Tuple[BaseRetriever, float]]):
        self.retrievers = retrievers  # List of (retriever, weight) tuples

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        all_results = []

        # Get results from each retriever
        for retriever, weight in self.retrievers:
            results = retriever.retrieve(query, k=k)
            weighted_results = [(doc, score * weight) for doc, score in results]
            all_results.extend(weighted_results)

        # Combine and deduplicate results
        unique_results = {}
        for doc, score in all_results:
            if doc in unique_results:
                unique_results[doc] = max(unique_results[doc], score)
            else:
                unique_results[doc] = score

        # Sort by score and return top k
        sorted_results = sorted(unique_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]


class ExamSolver:
    def load_exam(self, exam_file: str) -> List[ExamQuestion]:
        """Load exam questions from JSON file."""
        with open(exam_file, "r") as f:
            data = json.load(f)

        questions = []
        for item in data:
            question = ExamQuestion(
                question=item["question"],
                choices=item["choices"],
                correct_answer=item["correct_answer"],
                documentation=item.get("documentation", []),
            )
            questions.append(question)
        return questions

    def solve_question(self, question: ExamQuestion, model, CoT, self_ask) -> str:
        """Solve a single exam question with LLM."""

        formatted_choices = "\n".join(
            f"{chr(65+i)}. {choice}" for i, choice in enumerate(question.choices)
        )

        # Construct a more structured prompt with system and user roles
        if CoT:
            prompt = f"""<s>[INST] <<SYS>>
            As an assistant, your task is to answer the question after <Question> by using the given supporting documents.
            You should first think step by step about the question and give your thought, then answer the <Question>. 
            Your answer should be after <Answer> in JSON format with key "thought" and "answer" and their values should be string.
            The "answer" must be one of "A", "B", "C", or "D".
            Do not output anything else.

            There are some examples for you to refer to:

            <Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
            <Answer>:
            {{"thought":"Modern Record is a big R&B label with artists including Etta James, Joe Houston, Little Richard, Ike, Tina Turner and John Lee Hooker in the 1950s and 1960s.
            Little Richard is an American musician, signer actor and songwriter, born in December 5 1932. So the answer is Little Richard.","answer": "Little Richard"}}
            
            <Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
            <Answer>:
            {{"thought":"Chinua Achebe was a Nigerian novelist, poet, professor, and critic. Rachel Carson was an American marine biologist, author, and conservationist.
            Chinua Achebe has 4 jobs while Rachel Carson has 3 jobs. So the answer is Chinua Achebe.",
            "answer": "Chinua Achebe"}}
            
            <Question>: Remember Me Ballin' is a CD single by Indo G that features an American rapper born in what year?
            <Answer>:
            {{"thought":"Remember Me Ballin' is the CD singer by Indo G that features Gangsta Boo, who is named Lola Mitchell, an American rapper born in 1979. So the answer is 1979.",
            "answer": "1979"}}
            <</SYS>>[/INST]
            
            Now your supporting document:
            {question.documentation}
            
            Question is
            <Question>: {question.question}
            <Choices>: {formatted_choices}
            <Answer>:
            </s>
            """
        
        elif self_ask:
            prompt = f"""<s>[INST] <<SYS>>
            Solve the question with the given knowledge.
            Each line should start with either "Intermediate_answer", "Follow_up", "final_answer", or "Are_follow_up_questions_needed_here:".
            Your answer should be in JSON format with key "Intermediate_answer", "Follow_up", and "final_answer" and their values should be string.
            The "final_answer" must be one of "A", "B", "C", or "D".
            Do not output anything else.

            Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
            Are_follow_up_questions_needed_here: Yes.
            Follow_up: Who worked with Modern Records?
            Intermediate_answer: Artists worked with Modern Records include Etta James, Little Richard, Joe Houston, Ike and Tina Turner and John Lee Hooker.
            Follow_up: Is Little Richard an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?
            Intermediate_answer: Yes, Little Richard, born in December 5, 1932, is an American musician, singer, actor, comedian and songwriter.
            final_answer: Little Richard

            Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
            Are_follow_up_questions_needed_here: Yes.
            Follow_up: What jobs did Chinua Achebe have?
            Intermediate_answer: Chinua Achebe was a Nigerian (1) novelist, (2) poet, (3) professor, and (4) critic, so Chinua Achebe had 4 jobs.
            Follow_up: What jobs did Rachel Carson have?
            Intermediate_answer: Rachel Carson was an American (1) marine biologist, (2) author, and (3) conservationist, so Rachel Carson had 3 jobs.
            Follow_up: Did Chinua Achebe have more jobs than Rachel Carson?
            Intermediate_answer: Chinua Achebe had 4 jobs, while Rachel Carson had 3 jobs. 4 is greater than 3, so yes, Chinua Achebe had more jobs.
            final_answer: Chinua Achebe

            Question: Remember Me Ballin' is a CD single by Indo G that features an American rapper born in what year?
            Are_follow_up_questions_needed_here: Yes.
            Follow_up: Which American rapper is featured by Remember Me Ballin', a CD single by Indo G?
            Intermediate_answer: Gangsta Boo
            Follow_up: In which year was Gangsta Boo born?
            Intermediate_answer: Gangsta Boo was born in August 7, 1979, so the answer is 1979.
            final_answer: 1979
            <</SYS>>[/INST]
            
            Knowledge:
            {question.documentation}

            Question: {question.question}
            
            Choices:
            {formatted_choices}
            Are_follow_up_questions_needed_here:
            </s>
            """
        else:
            prompt = f"""<s>[INST] <<SYS>>
            You are an AI assistant taking a multiple choice exam. Your task is to:
            1. Read the question and choices carefully
            2. Analyze the choices
            3. Select the most appropriate answer
            4. Respond with ONLY the letter (A, B, C, or D) of the correct answer
            <</SYS>>

            Question: {question.question}

            Choices:
            {formatted_choices}

            Instructions:
            - You must respond with exactly one letter: A, B, C, or D
            - Do not include any explanation, period, or additional text
            - Just the letter of the correct answer

            Examples of valid responses:
            A
            B
            C
            D

            Your answer (one letter only): [/INST]</s>"""

        # Get model response
        try:
            response = model.invoke(prompt)

            if CoT:
                response = self.extract_json_with_fallbacks(response)["answer"]
            
            elif self_ask:
                response = self.extract_json_with_fallbacks(response)["final_answer"]

            # Extract just the letter from the response
            # Look for first occurrence of A, B, C, or D
            valid_answers = {"A", "B", "C", "D"}
            for char in response:
                if char in valid_answers:
                    return char

            # If no valid letter found, return the last character as fallback
            return response.strip()[-1]
        except:
            return "NA"
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize the model name for use in filenames."""
        # Replace forward slashes with underscores
        # Remove or replace other potentially problematic characters
        sanitized = re.sub(r'[/\\:*?"<>|]', '_', filename)
        return sanitized
    
    def extract_json_from_llm_output(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Robustly extract JSON from LLM output that may contain markdown code blocks and additional text.
        
        Args:
            response (str): Raw LLM output potentially containing JSON in markdown code blocks
            
        Returns:
            Optional[Dict[str, Any]]: Extracted JSON dictionary if found and valid, None otherwise
            
        Examples:
            >>> text = '```json\n{"key": "value"}\n```\nExtra text'
            >>> extract_json_from_llm_output(text)
            {'key': 'value'}
        """
        # Pattern to match JSON content within markdown code blocks
        json_pattern = r'```(?:json)?\n(.*?)\n```'
        
        try:
            # Try to find JSON block using regex
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            if matches:
                # Take the first JSON block found
                json_str = matches[0].strip()
                return json.loads(json_str)
            
            # If no markdown blocks found, try parsing the entire response
            return json.loads(response.strip())
        
        except json.JSONDecodeError as e:
            # Handle malformed JSON
            print(f"Failed to parse JSON: {str(e)}")
            return None
        except Exception as e:
            # Handle any other unexpected errors
            print(f"Unexpected error during JSON extraction: {str(e)}")
            return None
    
    def extract_json_with_fallbacks(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON using multiple fallback strategies.
        
        Args:
            response (str): Raw LLM output potentially containing JSON
            
        Returns:
            Optional[Dict[str, Any]]: Extracted JSON dictionary if found and valid, None otherwise
        """
        # Strategy 1: Try the regex-based extraction first
        result = self.extract_json_from_llm_output(response)
        if result:
            return result
        
        # Strategy 2: Try finding the first { and last } if regex failed
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx + 1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Clean up common issues and try again
        try:
            cleaned = response.replace('```json\n', '').replace('\n```', '')
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
            cleaned = cleaned.strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    def evaluate_performance(
        self, questions: List[ExamQuestion], model, task_domain, model_name, exam_file, CoT, self_ask
    ) -> Dict[str, float]:
        """Evaluate the solver's performance on a set of questions."""
        correct = 0
        total = len(questions)
        results = []

        for question in tqdm(questions):
            predicted_answer = self.solve_question(question, model, CoT, self_ask)

            question_result = {
                "question": question.question,
                "model_answer": predicted_answer,
                "correct_answer": question.correct_answer,
                "is_correct": predicted_answer == question.correct_answer,
                "number_of_hops": len(question.documentation)
            }

            # Add the question result to the list
            results.append(question_result)

            if predicted_answer == question.correct_answer:
                correct += 1

        metrics = {"accuracy": correct / total, "correct": correct, "total": total}

        results_dir = f"MultiHopData/{task_domain}/exam_results/"
        os.makedirs(results_dir, exist_ok=True)

        if CoT:
            results_file = os.path.join(
                results_dir,
                f"{model_name}_{os.path.basename(exam_file)}_results_CoT.json"
            )
        elif self_ask:
            results_file = os.path.join(
                results_dir,
                f"{model_name}_{os.path.basename(exam_file)}_results_self_ask.json"
            )
        else:
            results_file = os.path.join(
                results_dir,
                f"{model_name}_{os.path.basename(exam_file)}_results.json"
            )
        with open(results_file, "w") as json_file:
            json.dump(results, json_file, indent=2)

        return metrics


def main(task_domain: str, model_type: str, model_name: str, exam_file: str, CoT: bool, self_ask: bool):
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
            "gemma2_9b": ModelType.GEMMA2_9B,
            "gemma2_27b": ModelType.GEMMA2_27B,
        }
        
        print(f"Using {model_mapping[model_name]}")
        model = ModelFactory.create_model(model_mapping[model_name])
    else:
        print("Invalid model name")

    print("Solving the exam")
    solver = ExamSolver()
    questions = solver.load_exam(f"MultiHopData/{task_domain}/exams/{exam_file}")
    metrics = solver.evaluate_performance(questions, model, task_domain, model_name, exam_file, CoT, self_ask)

    print(f"Exam Performance:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")


if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    # task_domains = ["gov_report"]
    # model_type = "claude"
    # model_type = "gemini"
    model_type = "cpp"
    # model_name = "claude-3-5-haiku@20241022"
    # model_name = "claude-3-5-sonnet@20240620"
    # model_name = "gemini-1.5-pro-002"
    # model_name = "gemini-1.5-flash-002"

    # model_names = ["MINISTRAL_8B"]
    model_names = [
        'llama_3_1_8b',
        "ministral_8b",
        "gemma2_9b",
        ]
    
    exam_files = [
        # "llama_3_2_3b_single_hop_exam_processed.json",
        # "gemma2_9b_single_hop_exam_processed.json",
        # "ministral_8b_single_hop_exam_processed.json",
        "exam_new_ministral_8b_processed_v3.json",
        "exam_new_llama_3_2_3b_processed_v3.json",
        "exam_new_gemma2_9b_processed_v3.json",
        # V8
        # "exam_new_ministral_8b_processed_v8.json",
        # "exam_new_llama_3_2_3b_processed_v8.json",
        # "exam_new_gemma2_9b_processed_v8.json",
        ]
    
    CoT = False
    self_ask = True

    for exam_file in exam_files:
        for model_name in model_names:
            for task_domain in task_domains:
                print(f"Using {model_name}")
                print(f"Solving {exam_file} on {task_domain}")
                main(task_domain, model_type, model_name, exam_file, CoT, self_ask)
