import re
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union
import json
import faiss
import numpy as np
from tqdm import tqdm

from MultiHopData.retriever import BaseRetriever, BM25Retriever, Chunk, ChunkRetriever, FAISSRetriever, HybridRetriever, RerankingRetriever
from LLMServer.gcp.claude_instant import ClaudeGcp
from LLMServer.gcp.gemini_instant import GeminiGcp
from LLMServer.llama_instant import ModelFactory, ModelType


@dataclass
class ExamQuestion:
    question: str
    choices: List[str]
    correct_answer: str
    documentation: List[str]
    retrieved_chunks: Optional[Dict[str, List[Dict[str, Union[str, float]]]]] = None


class ExamSolver:
    def __init__(self, retriever: Optional[BaseRetriever] = None, n_documents: int = 15):
        self.retriever = retriever
        self.n_documents = n_documents

    def load_exam(self, exam_file: str) -> List[ExamQuestion]:
        """Load exam questions from JSON file with pre-retrieved chunks."""
        with open(exam_file, "r") as f:
            data = json.load(f)

        questions = []
        for item in data:
            question = ExamQuestion(
                question=item["question"],
                choices=item["choices"],
                correct_answer=item["correct_answer"],
                documentation=item.get("documentation", []),
                retrieved_chunks=item.get("retrieved_chunks", None)
            )
            questions.append(question)
        return questions
    
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

    def solve_question(self, question: ExamQuestion, model, retriever_method: str, pos_swap, CoT, self_ask, error_category: bool = False) -> str:
        """Solve a single exam question using either live retrieval or pre-retrieved chunks."""
        if question.retrieved_chunks and retriever_method in question.retrieved_chunks:
            # Use pre-retrieved chunks
            retrieved_docs = question.retrieved_chunks[retriever_method]
            # Sort by score in descending order and take top n_documents
            sorted_docs = sorted(retrieved_docs, key=lambda x: x['score'], reverse=True)[:self.n_documents]
            context = "\n".join([f"{i+1}) {doc['content']}" for i, doc in enumerate(sorted_docs)])
        elif self.retriever:
            # Fallback to live retrieval if no pre-retrieved chunks available
            retrieved_docs = self.retriever.retrieve(question.question, k=self.n_documents)
            context = "\n".join([f"{i+1}) {doc}" for i, (doc, _) in enumerate(retrieved_docs)])
        else:
            context = "No supporting documents available."

        formatted_choices = "\n".join(f"{choice}" for choice in question.choices)
        
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
            {context}
            
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
            {context}

            Question: {question.question}
            
            Choices:
            {formatted_choices}
            Are_follow_up_questions_needed_here:
            </s>
            """
            

        elif pos_swap:
            prompt = f"""<s>[INST] <<SYS>>
            You are an AI assistant taking a multiple choice exam.
            Your task is to:
            1. Read the given question, choices and supporting document carefully
            2. Select the most appropriate answer
            3. Respond with ONLY one letter (A, B, C, or D) of the correct answer
            
            Instructions:
            - You must respond with exactly one letter: A, B, C, or D
            - Do not include any explanation, period, or additional text
            - Just the letter of the correct answer

            Examples of valid responses:
            A
            B
            C
            D

            <</SYS>>[/INST]
            
            Supporting documents:
            {context}
            
            Question: {question.question}

            Choices:
            {formatted_choices}
            
            Your answer (one letter only):
            </s>
            """
        else:
            prompt = f"""<s>[INST] <<SYS>>
            You are an AI assistant taking a multiple choice exam.
            Your task is to:
            1. Read the given question, choices and supporting document carefully
            2. Select the most appropriate answer
            3. Respond with ONLY one letter (A, B, C, or D) of the correct answer
            
            Instructions:
            - You must respond with exactly one letter: A, B, C, or D
            - Do not include any explanation, period, or additional text
            - Just the letter of the correct answer

            Examples of valid responses:
            A
            B
            C
            D

            <</SYS>>[/INST]

            Question: {question.question}

            Choices:
            {formatted_choices}
            
            Supporting documents:
            {context}
            
            Your answer (one letter only):
            </s>
            """

        try:
            response = model.invoke(prompt)
            
            if error_category:
                return response
            
            valid_answers = {"A", "B", "C", "D"}
            # if CoT or self_ask:
            #     return response
            if CoT:
                response = self.extract_json_with_fallbacks(response)["answer"]
                # response = json.loads(response.replace('```json\n', '').replace('\n```', ''))["answer"]
                # response = self._extract_final_answer(response)
            
            elif self_ask:
                response = self.extract_json_with_fallbacks(response)["final_answer"]
                # response = json.loads(response.replace('```json\n', '').replace('\n```', ''))["final_answer"]

            for char in response:
                if char in valid_answers:
                    return char
            return response.strip()[-1]
        except:
            return "NA"

    def evaluate_performance(
        self, questions: List[ExamQuestion], model, task_domain, retriever_type, model_name, exam_file, n_documents, pos_swap, CoT, self_ask
    ) -> Dict[str, float]:
        """Evaluate the solver's performance on a set of questions."""
        correct = 0
        total = len(questions)
        results = []

        print(f"Solving the exam using {retriever_type} retriever")
        for question in tqdm(questions):
            # Map retriever type to the corresponding key in retrieved_chunks
            if retriever_type == "Rerank":
                retriever_method = retriever_type
            else:
                retriever_method = retriever_type.lower()  # 'Dense' -> 'dense', etc.
            predicted_answer = self.solve_question(question, model, retriever_method, pos_swap, CoT, self_ask)

            # try:
            #     if self_ask:
            #         solver = ExamSolver()
            #         response_dict = solver.extract_json_with_fallbacks(predicted_answer)
            #         question_result = {
            #             "question": question.question,
            #             "model_answer": response_dict["final_answer"],
            #             "correct_answer": question.correct_answer,
            #             "is_correct": predicted_answer == question.correct_answer,
            #             "number_of_hops": len(question.documentation),
            #             "reasoning": predicted_answer
            #         }
            #     elif CoT:
            #         solver = ExamSolver()
            #         response_dict = solver.extract_json_with_fallbacks(predicted_answer)
            #         question_result = {
            #             "question": question.question,
            #             "model_answer": response_dict["answer"],
            #             "correct_answer": question.correct_answer,
            #             "is_correct": predicted_answer == question.correct_answer,
            #             "number_of_hops": len(question.documentation),
            #             "reasoning": predicted_answer
            #         }
            #     else:        
            #         question_result = {
            #             "question": question.question,
            #             "model_answer": predicted_answer,
            #             "correct_answer": question.correct_answer,
            #             "is_correct": predicted_answer == question.correct_answer,
            #             "number_of_hops": len(question.documentation),
            #         }
            # except Exception:
            #     question_result = {
            #         "question": question.question,
            #         "model_answer": "NA",
            #         "correct_answer": question.correct_answer,
            #         "is_correct": predicted_answer == question.correct_answer,
            #         "number_of_hops": len(question.documentation)
            #     }
            question_result = {
                    "question": question.question,
                    "model_answer": predicted_answer,
                    "correct_answer": question.correct_answer,
                    "is_correct": predicted_answer == question.correct_answer,
                    "number_of_hops": len(question.documentation),
                }
            

            results.append(question_result)

            if predicted_answer == question.correct_answer:
                correct += 1

        metrics = {"accuracy": correct / total, "correct": correct, "total": total}

        # Save results
        results_dir = f"MultiHopData/{task_domain}/exam_results"
        os.makedirs(results_dir, exist_ok=True)
        
        if pos_swap:
            results_path = os.path.join(
                results_dir,
                f"{model_name}_{retriever_type}_{os.path.basename(exam_file)}_{n_documents}_results_chunk_pos_swap.json"
            )
        elif CoT:
            results_path = os.path.join(
                results_dir,
                f"{model_name}_{retriever_type}_{os.path.basename(exam_file)}_{n_documents}_results_CoT.json"
            )
        elif self_ask:
            results_path = os.path.join(
                results_dir,
                f"{model_name}_{retriever_type}_{os.path.basename(exam_file)}_{n_documents}_results_self_ask.json"
            )
        else:
            results_path = os.path.join(
                results_dir,
                f"{model_name}_{retriever_type}_{os.path.basename(exam_file)}_results.json"
            )
        
        with open(results_path, "w") as json_file:
            json.dump(results, json_file, indent=2)

        return metrics


def main(
    task_domain: str,
    retriever_type: str,
    model_type: str,
    model_name: str,
    exam_file: str,
    n_documents: int,
    pos_swap: bool,
    CoT: bool,
    self_ask: bool
) -> None:
    """
    Main function to solve exams using pre-retrieved chunks.
    
    Args:
        task_domain: Domain of the task (e.g., "gov_report")
        retriever_type: Type of retriever to use ("Dense", "Sparse", "Hybrid")
        model_type: Type of model to use ("gemini", "claude", "cpp")
        model_name: Name of the specific model
        exam_file: Name of the exam file to solve
        n_documents: Number of supporting documents to use
    """
    # Initialise exam solver without retriever since we're using pre-retrieved chunks
    solver = ExamSolver(n_documents=n_documents)

    # Initialise the appropriate model
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

    # Construct the path to the exam file with retrieved chunks
    # exam_with_chunks = exam_file.replace('.json', '_with_retrievals.json')
    exam_path = f"MultiHopData/{task_domain}/exams/{exam_file}"
    
    try:
        # Load and solve exam using pre-retrieved chunks
        questions = solver.load_exam(exam_path)
        metrics = solver.evaluate_performance(
            questions=questions,
            model=model,
            task_domain=task_domain,
            retriever_type=retriever_type,
            model_name=model_name,
            exam_file=exam_file,
            n_documents=n_documents,
            pos_swap=pos_swap,
            CoT=CoT,
            self_ask=self_ask,
        )

        print(f"\nExam Performance Summary:")
        print(f"Model: {model_name}")
        print(f"Task: {task_domain}")
        print(f"Retriever: {retriever_type}")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Correct: {metrics['correct']}/{metrics['total']}")
        print("-" * 50)
        
    except FileNotFoundError:
        print(f"Error: Could not find exam file with pre-retrieved chunks at {exam_path}")
        print("Please ensure you have run the retrieval preparation step first.")
        return


if __name__ == "__main__":
    # Configuration
    model_type = "cpp"
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    # task_domains = ["SecFilings"]
    # retriever_types = ["Dense", "Sparse", "Hybrid", "Rerank"]
    retriever_types = ["Dense"]
    model_names = [
        'llama_3_1_8b',
        "ministral_8b",
        "gemma2_9b",
    ]
    exam_files = [
        # V3 (ChunkRetriever)
        # "llama_3_2_3b_single_hop_exam_processed.json",
        # "gemma2_9b_single_hop_exam_processed.json",
        # "ministral_8b_single_hop_exam_processed.json",
        "exam_new_llama_3_2_3b_processed_v3.json",
        "exam_new_gemma2_9b_processed_v3.json",
        "exam_new_ministral_8b_processed_v3.json",
        # V4 (HybridChunk)
        # "llama_3_2_3b_single_hop_exam_processed_v4.json",
        # "gemma2_9b_single_hop_exam_processed_v4.json",
        # "ministral_8b_single_hop_exam_processed_v4.json",
        # "exam_new_llama_3_2_3b_processed_v4.json",
        # "exam_new_gemma2_9b_processed_v4.json",
        # "exam_new_ministral_8b_processed_v4.json",
        # V5 (chunk: 512)
        # "llama_3_2_3b_single_hop_exam_processed_v5.json",
        # "gemma2_9b_single_hop_exam_processed_v5.json",
        # "ministral_8b_single_hop_exam_processed_v5.json",
        # "exam_new_llama_3_2_3b_processed_v5.json",
        # "exam_new_gemma2_9b_processed_v5.json",
        # "exam_new_ministral_8b_processed_v5.json",
        # V9 (chunk: 1024)
        # "exam_new_gemma2_9b_processed_v9.json",
        # "exam_new_llama_3_2_3b_processed_v9.json",
        # "exam_new_ministral_8b_processed_v9.json",
        # V10 (chunk: 2048)
        # "exam_new_llama_3_2_3b_processed_v10.json",
        # "exam_new_gemma2_9b_processed_v10.json",
        # "exam_new_ministral_8b_processed_v10.json",
    ]
    n_documents = 5
    CoT = False
    self_ask = True

    # Process all combinations
    for exam_file in exam_files:
        for model_name in model_names:
            for task_domain in task_domains:
                for retriever_type in retriever_types:
                    print(f"\nProcessing:")
                    print(f"Model: {model_name}")
                    print(f"Exam: {exam_file}")
                    print(f"Task: {task_domain}")
                    print(f"Retriever: {retriever_type}")
                    
                    main(
                        task_domain=task_domain,
                        retriever_type=retriever_type,
                        model_type=model_type,
                        model_name=model_name,
                        exam_file=exam_file,
                        n_documents=n_documents,
                        pos_swap=False,
                        CoT=CoT,
                        self_ask=self_ask
                    )