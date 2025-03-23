from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import json
import faiss
import numpy as np
from tqdm import tqdm
import re

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
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """
        Extracts the final answer (A, B, C, or D) from a reasoning chain text.
        Handles various formats including those with choice descriptions and parentheses.
        
        Args:
            text (str): The input text containing the reasoning chain and final answer
            
        Returns:
            Optional[str]: The extracted answer (A, B, C, or D) or None if no valid answer is found
        """
        # Clean the input text
        text = text.strip()
        
        # List of possible answer patterns (ordered from most specific to least specific)
        patterns = [
            # Pattern for "The answer is: A) Description"
            r'(?:the answer is:|therefore, the answer is:|final answer:)\s*([ABCD])\).*?(?=\n|$)',
            
            # Pattern for "The answer is: Choice A"
            r'(?:the answer is:|therefore, the answer is:|final answer:)\s*(?:choice\s+)?([ABCD])(?:\s|$)',
            
            # Standard patterns
            r'## Answer:\s*\n\s*([ABCD])\s*$',
            r'## Final Answer:\s*([ABCD])\s*$',
            r'Therefore,\s*the answer is:\s*\*\*([ABCD])\*\*\s*$',
            r'Therefore, the answer is:\s*([ABCD])\s*$',
            r'The answer is\s*([ABCD])\s*$',
            r'\*\*([ABCD])\*\*\s*$',
            
            # Last resort pattern - find the last occurrence of A, B, C, or D
            r'(?:^|\s)([ABCD])(?:\)|\.|\s|$)'
        ]
        
        # Try each pattern (case-insensitive)
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            if matches:
                # Get the last match (in case there are multiple)
                return matches[-1].group(1).upper()
        
        # If no pattern matches, try to find the last occurrence after specific keywords
        keywords = ['answer', 'Answer', 'therefore', 'Therefore', 'conclusion', 'Conclusion']
        last_position = -1
        final_answer = None
        
        for keyword in keywords:
            pos = text.rfind(keyword)
            if pos > last_position:
                # Search for A, B, C, or D after this keyword
                substring = text[pos:]
                match = re.search(r'(?:^|\s)([ABCD])(?:\)|\.|\s|$)', substring, re.IGNORECASE)
                if match:
                    final_answer = match.group(1).upper()
                    last_position = pos
        
        return final_answer
    
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

    def validate_answer(self, answer: Optional[str]) -> bool:
        """
        Validates if the extracted answer is one of A, B, C, or D.
        
        Args:
            answer (Optional[str]): The extracted answer to validate
            
        Returns:
            bool: True if the answer is valid, False otherwise
        """
        if answer is None:
            return False
        return answer.upper() in {'A', 'B', 'C', 'D'}

    def solve_question(self, question: ExamQuestion, model, internal_prevention: bool, CoT: bool, position_swap: bool, self_ask: bool) -> str:
        """Solve a single exam question with LLM."""

        formatted_choices = "\n".join(
            f"{choice}" for i, choice in enumerate(question.choices)
        )

        # Construct a more structured prompt with system and user roles
        if internal_prevention:
            prompt = f"""<s>[INST] <<SYS>>
                You are an AI assistant taking a multiple choice exam. Your task is to:

                1. Read the question, provided choices, and documents carefully
                2. Analyze them following these critical guidelines:
                - Base your analysis STRICTLY on explicit information in the documents
                - Do not infer causal relationships unless explicitly stated
                - Do not assume relationships or connections without direct textual evidence
                - Do not generalize from specific instances
                - Do not assume one factor is primary unless explicitly ranked
                - Do not assume that temporal proximity implies causation
                - Be especially careful with words like "because," "due to," "led to," or "resulted in"
                3. Rewrite the given supporting documents in order to select the most appropriate choice
                4. Re-assess the question, choices and rewritten documents and select the answer choice that:
                - Is directly supported by the documents
                - Makes the fewest assumptions beyond the given facts
                - Remains valid even if unstated possibilities exist
                5. Respond with ONLY the letter (A, B, C, or D) of the correct answer

                Question: {question.question}

                Choices:
                {formatted_choices}

                Documents:
                {question.documentation}

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
        
        elif CoT:
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
        
        elif position_swap:
            prompt = f"""<s>[INST] <<SYS>>
                You are an AI assistant taking a multiple choice exam. Your task is to:
                1. Read the question, provided choices and documents carefully
                2. Analyze the choices
                3. Select the most appropriate answer
                4. Respond with ONLY the letter (A, B, C, or D) of the correct answer
                
                Instructions:
                - You must respond with exactly one letter: A, B, C, or D
                - Do not include any explanation, period, or additional text
                - Just the letter of the correct answer

                Examples of valid responses:
                A
                B
                C
                D

                Your answer (one letter only): <</SYS>>[/INST]</s>
                
                Documents:
                {question.documentation}
                
                Question: {question.question}

                Choices:
                {formatted_choices}
                """
        
        else:
            prompt = f"""<s>[INST] <<SYS>>
                You are an AI assistant taking a multiple choice exam. Your task is to:
                1. Read the question, provided choices and documents carefully
                2. Analyze the choices
                3. Select the most appropriate answer
                4. Respond with ONLY the letter (A, B, C, or D) of the correct answer
                <</SYS>>

                Question: {question.question}

                Choices:
                {formatted_choices}
                
                Documents:
                {question.documentation}

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
    
    def solve_question_with_reasoning(self, question: ExamQuestion, model) -> str:
        """Solve a single exam question with LLM."""

        formatted_choices = "\n".join(
            f"{choice}" for i, choice in enumerate(question.choices)
        )

        # Construct a more structured prompt with system and user roles
        prompt = f"""<start_of_turn>user
        You are an AI assistant taking a multiple choice exam. Your tasks are:
        1. Read the question, provided choices and documents carefully
        2. Analyse the choices
        3. Select the most appropriate answer
        4. Respond the correct answer with your reasoning
        
        Instruction:
        Output format:
        Your answer: One of the following choices - A, B, C, D
        Reasoning: Explanation of why you choose your answer

        Question: {question.question}

        Choices:
        {formatted_choices}
        
        Documents:
        {question.documentation}

        <end_of_turn>
        <start_of_turn>model
        Your answer:
        """

        try:
            response = model.invoke(prompt)
            
            valid_answers = {"A", "B", "C", "D"}
            if CoT:
                response = self.extract_json_with_fallbacks(response)["answer"]
            
            elif self_ask:
                response = self.extract_json_with_fallbacks(response)["final_answer"]

            for char in response:
                if char in valid_answers:
                    return char
            return response.strip()[-1]

        except Exception as e:
            print(f"WARNING reasoning generation error: {e}")
            return "NA"

    def evaluate_performance(
        self, questions: List[ExamQuestion], model, task_domain, model_name, exam_file, internal_prevention, CoT, position_swap, self_ask
    ) -> Dict[str, float]:
        """Evaluate the solver's performance on a set of questions."""
        correct = 0
        total = len(questions)
        results = []

        for question in tqdm(questions):
            predicted_answer = self.solve_question(question, model, internal_prevention, CoT, position_swap, self_ask)

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

        if internal_prevention:
            with open(f"MultiHopData/{task_domain}/exam_results/{model_name}_open_{exam_file}_internal_prevention_v2.json", "w") as json_file:
                json.dump(results, json_file, indent=2)
        elif CoT:
            with open(f"MultiHopData/{task_domain}/exam_results/{model_name}_open_{exam_file}_CoT.json", "w") as json_file:
                json.dump(results, json_file, indent=2)
        elif self_ask:
            with open(f"MultiHopData/{task_domain}/exam_results/{model_name}_open_{exam_file}_self_ask.json", "w") as json_file:
                json.dump(results, json_file, indent=2)
        elif position_swap:
            with open(f"MultiHopData/{task_domain}/exam_results/{model_name}_open_{exam_file}_chunk_pos_swap.json", "w") as json_file:
                json.dump(results, json_file, indent=2)
        else:
            with open(f"MultiHopData/{task_domain}/exam_results/{model_name}_open_{exam_file}.json", "w") as json_file:
                json.dump(results, json_file, indent=2)

        return metrics


def main(task_domain: str, model_type: str, model_name: str, exam_file: str, internal_prevention: bool, CoT: bool, position_swap: bool, self_ask: bool):
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
            "gemma2_27b": ModelType.GEMMA2_27B,
            "deepseek_qwen_1_5b": ModelType.DEEPSEEK_R1_QWEN_1_5B,
            "deepseek_qwen_7b": ModelType.DEEPSEEK_R1_QWEN_7B,
            "deepseek_llama_8b": ModelType.DEEPSEEK_R1_LLAMA_8B
        }
        
        print(f"Using {model_mapping[model_name]}")
        model = ModelFactory.create_model(model_mapping[model_name])
    else:
        print("Using Llama-cpp")
        # model = LlamaModel(model_path=model_path)

    print("Solving the exam")
    solver = ExamSolver()
    questions = solver.load_exam(f"MultiHopData/{task_domain}/exams/{exam_file}")
    metrics = solver.evaluate_performance(questions, model, task_domain, model_name, exam_file, internal_prevention, CoT, position_swap, self_ask)

    print(f"Exam Performance:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")


if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    # task_domains = ["multifieldqa_en"]
    # model_type = "claude"
    # model_type = "gemini"
    model_type = "cpp"
    # model_name = "claude-3-5-haiku@20241022"
    # model_name = "claude-3-5-sonnet@20240620"
    # model_name = "gemini-1.5-pro-002"
    # model_name = "gemini-1.5-flash-002"

    # model_names = ["gemini-1.5-pro-002", "gemini-1.5-flash-002"]
    # model_names = ["claude-3-5-sonnet@20240620", "claude-3-5-haiku@20241022"]
    # model_names = ["claude-3-5-sonnet@20240620"]
    model_names = [
        'llama_3_1_8b',
        "ministral_8b",
        "gemma2_9b",
        # "deepseek_llama_8b",
        # "deepseek_qwen_1_5b",
        # "deepseek_qwen_7b",
        ]
    
    exam_files = [
        # V8 chunk_size = 512
        # "exam_new_llama_3_2_3b_processed_v8.json",
        # "exam_new_gemma2_9b_processed_v8.json",
        # "exam_new_ministral_8b_processed_v8.json",
        # # V6 chunk_size = 1024
        # "exam_new_llama_3_2_3b_processed_v6.json",
        # "exam_new_gemma2_9b_processed_v6.json",
        # "exam_new_ministral_8b_processed_v6.json",
        # # V7 chunk_size = 2048
        # "exam_new_llama_3_2_3b_processed_v7.json",
        # "exam_new_gemma2_9b_processed_v7.json",
        # "exam_new_ministral_8b_processed_v7.json",
        # V5 chunk_size = 512
        # "llama_3_2_3b_single_hop_exam_processed.json",
        # "gemma2_9b_single_hop_exam_processed.json",
        # "ministral_8b_single_hop_exam_processed.json",
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
        # V3 (chunk size: 4000)
        "exam_new_llama_3_2_3b_processed_v3.json",
        "exam_new_gemma2_9b_processed_v3.json",
        "exam_new_ministral_8b_processed_v3.json",
        ]
    internal_prevention = False
    CoT = False
    self_ask = True
    position_swap = False

    for exam_file in exam_files:
        for model_name in model_names:
            for task_domain in task_domains:
                print(f"Using {model_name}")
                print(f"Solving {exam_file} on {task_domain}")
                try:
                    main(task_domain, model_type, model_name, exam_file, internal_prevention, CoT, position_swap, self_ask)
                except Exception as e:
                    print(f"The exam file does not exist: {e}")
