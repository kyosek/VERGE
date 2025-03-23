from typing import List, Dict

from LLMServer.llama_instant import ModelType


class PromptTemplate:
    @staticmethod
    def get_question_generation_prompt_template(model_type: ModelType, chunks: List[Dict[str, str]], task_domain, documentation) -> str:
        """Return the appropriate prompt template based on model type."""
        num_chunks = len(chunks)
        prompts = {
            ModelType.LLAMA_3_2_3B: f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Advanced Multi-Hop Question Generation Methodology

            Objective:
            Generate an exceptionally challenging multiple-choice question (1 correct answer and 3 distractors) that demands sophisticated, interdependent reasoning across multiple document chunks.

            Comprehensive Question Design Criteria:

            1. Multi-Hop Reasoning Architecture
            - Construct questions that necessitate intricate information synthesis
            - Enforce mandatory cross-chunk logical dependencies
            - Eliminate single-chunk solution pathways
            - Create a complex reasoning landscape that requires:
                * Temporal reasoning
                * Contextual cross-referencing
                * Nuanced inference mechanisms
                * Sophisticated information integration

            2. Chunk Interdependency Strategies
            - Mandate that NO single document chunk provides a complete answer
            - Implement strategic information distribution across chunks
            - Create deliberate cognitive "gaps" requiring active reasoning
            - Establish intricate logical bridges between disparate information sources

            3. Distractor Engineering Principles
            Three distractors must be meticulously crafted to exploit specific reasoning vulnerabilities:

            a) Partial Information Distractor
                - Leverages incomplete chunk understanding
                - Appears superficially plausible
                - Represents a common cognitive shortcut

            b) Contextual Misalignment Distractor
                - Based on seemingly logical but fundamentally flawed reasoning
                - Requires subtle comprehension of inter-chunk relationships
                - Exploits potential misinterpretation patterns

            c) Critical Dependency Distractor
                - Appears correct if crucial inter-chunk dependencies are overlooked
                - Demands highest-order reasoning to distinguish from correct answer
                - Represents a sophisticated reasoning challenge

            4. Reasoning Complexity Calibration
            - Questions should require:
                * Minimum {num_chunks}-step reasoning process
                * Explicit integration of ALL provided chunks
                * Ability to distinguish nuanced information relationships
            - Difficulty Level: Advanced (Graduate/Professional Examination Standard)

            5. Linguistic Precision Requirements
            - Use sophisticated, academically rigorous language
            - Avoid direct chunk referencing
            - Construct questions requiring active cognitive engagement
            - Ensure questions are solvable exclusively through provided documentation

            Output Instruction: 
            CRITICAL: Your entire response MUST start with "Question:" followed by the question text. Do NOT include any prefacing text or headers before the question.
            CRITICAL: All the options MUST start with the corresponding letter such as "A)", "B)", "C)" or "D)".
            CRITICAL: The Correct Answer MUST start with "Correct Answer".
            
            Output Format:
            Question: [Sophisticated multi-hop reasoning challenge]
            A) [Option A]
            B) [Option B]
            C) [Option C]
            D) [Option D]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            * Do not generate anything else other than the above output format

            Fundamental Constraints:
            - Zero single-chunk solution possibilities
            - Mandatory full documentation utilisation
            - Rigorous reasoning complexity
            - Linguistically sophisticated presentation

            Evaluation Heuristic:
            A high-quality multi-hop question successfully prevents:
            - Surface-level information scanning
            - Premature answer selection
            - Incomplete reasoning processes

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Domain: {task_domain}
            Documentation: {documentation}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            Generate an advanced multi-hop reasoning question
            """,
            ModelType.MINISTRAL_8B: f"""
            <s>[INST]
            Advanced Multi-Hop Question Generation Methodology

            Objective:
            Generate an exceptionally challenging multiple-choice question (1 correct answer and 3 distractors) that demands sophisticated, interdependent reasoning across multiple document chunks.

            Comprehensive Question Design Criteria:

            1. Multi-Hop Reasoning Architecture
            - Construct questions that necessitate intricate information synthesis
            - Enforce mandatory cross-chunk logical dependencies
            - Eliminate single-chunk solution pathways
            - Create a complex reasoning landscape that requires:
                * Temporal reasoning
                * Contextual cross-referencing
                * Nuanced inference mechanisms
                * Sophisticated information integration

            2. Chunk Interdependency Strategies
            - Mandate that NO single document chunk provides a complete answer
            - Implement strategic information distribution across chunks
            - Create deliberate cognitive "gaps" requiring active reasoning
            - Establish intricate logical bridges between disparate information sources

            3. Distractor Engineering Principles
            Three distractors must be meticulously crafted to exploit specific reasoning vulnerabilities:

            a) Partial Information Distractor
                - Leverages incomplete chunk understanding
                - Appears superficially plausible
                - Represents a common cognitive shortcut

            b) Contextual Misalignment Distractor
                - Based on seemingly logical but fundamentally flawed reasoning
                - Requires subtle comprehension of inter-chunk relationships
                - Exploits potential misinterpretation patterns

            c) Critical Dependency Distractor
                - Appears correct if crucial inter-chunk dependencies are overlooked
                - Demands highest-order reasoning to distinguish from correct answer
                - Represents a sophisticated reasoning challenge

            4. Reasoning Complexity Calibration
            - Questions should require:
                * Minimum {num_chunks}-step reasoning process
                * Explicit integration of ALL provided chunks
                * Ability to distinguish nuanced information relationships
            - Difficulty Level: Advanced (Graduate/Professional Examination Standard)

            5. Linguistic Precision Requirements
            - Use sophisticated, academically rigorous language
            - Avoid direct chunk referencing
            - Construct questions requiring active cognitive engagement
            - Ensure questions are solvable exclusively through provided documentation

            Output Format:
            Question: [Sophisticated multi-hop reasoning challenge]
            A) [Option A]
            B) [Option B]
            C) [Option C]
            D) [Option D]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            * Do not generate anything else other than the above output format

            Fundamental Constraints:
            - Zero single-chunk solution possibilities
            - Mandatory full documentation utilisation
            - Rigorous reasoning complexity
            - Linguistically sophisticated presentation

            Evaluation Heuristic:
            A high-quality multi-hop question successfully prevents:
            - Surface-level information scanning
            - Premature answer selection
            - Incomplete reasoning processes
            [/INST]
            
            Domain: {task_domain}
            Documentation: {documentation}
            
            Generate an advanced multi-hop reasoning question
            </s>
            """,
            ModelType.GEMMA2_9B: f"""
            <start_of_turn>user
            Advanced Multi-Hop Question Generation Methodology

            Objective:
            Generate an exceptionally challenging multiple-choice question (1 correct answer and 3 distractors) that demands sophisticated, interdependent reasoning across multiple document chunks.

            Comprehensive Question Design Criteria:

            1. Multi-Hop Reasoning Architecture
            - Construct questions that necessitate intricate information synthesis
            - Enforce mandatory cross-chunk logical dependencies
            - Eliminate single-chunk solution pathways
            - Create a complex reasoning landscape that requires:
                * Temporal reasoning
                * Contextual cross-referencing
                * Nuanced inference mechanisms
                * Sophisticated information integration

            2. Chunk Interdependency Strategies
            - Mandate that NO single document chunk provides a complete answer
            - Implement strategic information distribution across chunks
            - Create deliberate cognitive "gaps" requiring active reasoning
            - Establish intricate logical bridges between disparate information sources

            3. Distractor Engineering Principles
            Three distractors must be meticulously crafted to exploit specific reasoning vulnerabilities:

            a) Partial Information Distractor
                - Leverages incomplete chunk understanding
                - Appears superficially plausible
                - Represents a common cognitive shortcut

            b) Contextual Misalignment Distractor
                - Based on seemingly logical but fundamentally flawed reasoning
                - Requires subtle comprehension of inter-chunk relationships
                - Exploits potential misinterpretation patterns

            c) Critical Dependency Distractor
                - Appears correct if crucial inter-chunk dependencies are overlooked
                - Demands highest-order reasoning to distinguish from correct answer
                - Represents a sophisticated reasoning challenge

            4. Reasoning Complexity Calibration
            - Questions should require:
                * Minimum {num_chunks}-step reasoning process
                * Explicit integration of ALL provided chunks
                * Ability to distinguish nuanced information relationships
            - Difficulty Level: Advanced (Graduate/Professional Examination Standard)

            5. Linguistic Precision Requirements
            - Use sophisticated, academically rigorous language
            - Avoid direct chunk referencing
            - Construct questions requiring active cognitive engagement
            - Ensure questions are solvable exclusively through provided documentation

            Output Format:
            Question: [Sophisticated multi-hop reasoning challenge]
            A) [Option A]
            B) [Option B]
            C) [Option C]
            D) [Option D]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            * Do not generate anything else other than the above output format

            Fundamental Constraints:
            - Zero single-chunk solution possibilities
            - Mandatory full documentation utilisation
            - Rigorous reasoning complexity
            - Linguistically sophisticated presentation

            Evaluation Heuristic:
            A high-quality multi-hop question successfully prevents:
            - Surface-level information scanning
            - Premature answer selection
            - Incomplete reasoning processes

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Domain: {task_domain}
            Documentation: {documentation}
            <end_of_turn>
            <start_of_turn>model
            Generate an advanced multi-hop reasoning question
            """,
    }
        return prompts.get(model_type, prompts[ModelType.LLAMA_3_2_3B])
    
    @staticmethod
    def get_single_hop_question_generation_prompt_template(model_type: ModelType, chunks: List[Dict[str, str]], task_domain, documentation) -> str:
        prompts = {
            ModelType.LLAMA_3_2_3B: f"""
            <s>[INST]
            You are an expert exam question generator specialising in creating challenging multiple-choice questions (1 correct answer and 3 distractors) that require complex reasoning across multiple pieces of information.

            Core requirements:
            1. Question MUST require synthesizing information from the given chunk
            2. Distractors must be highly plausible and based on common misconceptions or partial understanding
            3. The correct answer should not be obvious without carefully analysing the given chunk
            4. Each distractor should represent a different type of reasoning error
            5. As students do not have an access to the chunk information, do not mention chunks in the question

            Question Design Principles:
            1. Require careful analysis of conditional statements
            2. Include scenarios where surface-level reading might lead to wrong conclusions
            3. Design distractors that would be chosen if key information from certain chunks is missed
                3.1. One distractor should incorporate some but not all key information
                3.2. One distractor should be based on common misinterpretation
                3.3. One distractor would be correct if one crucial detail is missed
            
            Format Requirements:
            - Question text should be clear but complex
            - Each option must start with A), B), C), or D)

            Generate a question following this format:
            Question: <Question>
            A) [Choice A]
            B) [Choice B]
            C) [Choice C]
            D) [Choice D]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            [/INST]

            Domain: {task_domain}
            Documentation: {documentation}
            </s>
            """,
            ModelType.LLAMA_3_1_8B: f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert exam question generator specialising in creating challenging multiple-choice questions (1 correct answer and 3 distractors) that require complex reasoning across multiple pieces of information.

            Core requirements:
            1. Question MUST require synthesizing information from the given chunk
            2. Distractors must be highly plausible and based on common misconceptions or partial understanding
            3. The correct answer should not be obvious without carefully analysing the given chunk
            4. Each distractor should represent a different type of reasoning error
            5. As students do not have an access to the chunk information, do not mention chunks in the question

            Question Design Principles:
            1. Require careful analysis of conditional statements
            2. Include scenarios where surface-level reading might lead to wrong conclusions
            3. Design distractors that would be chosen if key information from certain chunks is missed
                3.1. One distractor should incorporate some but not all key information
                3.2. One distractor should be based on common misinterpretation
                3.3. One distractor would be correct if one crucial detail is missed

            Format Requirements:
            - Question text should be clear but complex
            - Each option must start with A), B), C), or D)

            Domain: {task_domain}
            Documentation: {documentation}

            Generate a question following this format:
            Question: <Question>
            A) [Choice A]
            B) [Choice B]
            C) [Choice C]
            D) [Choice D]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Generate a question.
            <|eot_id|>
            """,
    }
        return prompts.get(model_type, prompts[ModelType.LLAMA_3_2_3B])
    
    @staticmethod
    def get_verification_prompt(model_type: ModelType, question_data: dict, chunks: List[Dict[str, str]]) -> str:
        """Format the verification prompt with the question data and chunks."""
        # Extract options from choices
        options = [choice.split(') ')[1] for choice in question_data['choices']]
        
        # Format chunk text
        chunk_text = '\n'.join([f"Chunk{i+1}: {chunk['text']}" 
                               for i, chunk in enumerate(chunks)])
        
        return PromptTemplate.verification_prompt_template(model_type, question_data, options, chunk_text)

    @staticmethod
    def verification_prompt_template(model_type: ModelType, question_data: dict, options: List[str], chunk_text: List[str]):
        question=question_data['question'],
        option_a=options[0],
        option_b=options[1],
        option_c=options[2],
        option_d=options[3],
        correct_answer=question_data['correct_answer'],
        chunk_text=chunk_text
            
        prompts = {
            ModelType.LLAMA_3_2_3B: f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an advanced exam question verifier specializing in multi-hop reasoning and comprehensive document analysis.

            Verification Objectives:
            1. Multi-Hop Reasoning Verification
            - Determine if the question requires synthesizing information across multiple document chunks
            - Identify all necessary chunks for a comprehensive answer
            - Detect and prevent "shortcut reasoning" that bypasses critical information

            2. Question-Answer Integrity Assessment
            - Evaluate the coherence and correctness of the question and all options
            - Verify that the correct answer is truly supported by the given documents
            - Assess the depth and complexity of reasoning required

            3. Chunk Utilization Analysis
            - Confirm whether all provided document chunks are essential
            - Identify any unused or potentially relevant chunks
            - Recommend strategies to incorporate underutilized information

            Output Instructions:
            Generate a comprehensive JSON response with the following mandatory entities:
            - required_chunks: List of chunk indices essential for answering
            - reasoning:
                - shortcut_reasoning_risk: Indicates if the question can be answered via shortcuts
                - if_solvable: Indicates if the question can be answered by only using the given documents
                - unused_chunks: List of chunks not utilized in the answer
                - improvement_suggestions: Recommendations for exam enhancement
            - confidence: 1-5 confidence scale in the assessment
            You do not need to add anything else other than the JSON response

            {{
                "required_chunks": [int],
                "reasoning": dict {{
                    "shortcut_reasoning_risk": bool,
                    "unused_chunks": [int],
                    "improvement_suggestions": str
                }},
                "confidence": int
            }}

            Specific Verification Criteria:
            - Multi-Hop Complexity: Assess if answering requires integrating information from multiple sources
            - Information Completeness: Evaluate whether all relevant document information is meaningfully incorporated
            - Reasoning Depth: Determine the level of analytical thinking needed to derive the correct answer

            Guidance for Analysis:
            - Critically examine each document chunk's contribution
            - Highlight potential ambiguities or gaps in the question-answer construct
            - Provide constructive feedback for improving exam quality and multi-hop reasoning challenge
            <|eot_id|><|start_header_id|>user<|end_header_id|>

            Question: {question}
            Options:
            A) {option_a}
            B) {option_b}
            C) {option_c}
            D) {option_d}
            Correct Answer: {correct_answer}
            Documents: {chunk_text}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            Perform comprehensive multi-hop reasoning verification
            """,
            ModelType.GEMMA2_9B: f"""<start_of_turn>user
            You are an advanced exam question verifier specializing in multi-hop reasoning and comprehensive document analysis.

            Verification Objectives:
            1. Multi-Hop Reasoning Verification
            - Determine if the question requires synthesizing information across multiple document chunks
            - Identify all necessary chunks for a comprehensive answer
            - Detect and prevent "shortcut reasoning" that bypasses critical information

            2. Question-Answer Integrity Assessment
            - Evaluate the coherence and correctness of the question and all options
            - Verify that the correct answer is truly supported by the given documents
            - Assess the depth and complexity of reasoning required

            3. Chunk Utilization Analysis
            - Confirm whether all provided document chunks are essential
            - Identify any unused or potentially relevant chunks
            - Recommend strategies to incorporate underutilized information

            Output Instructions:
            Generate a comprehensive JSON response with the following mandatory entities:
            - required_chunks: List of chunk indices essential for answering
            - reasoning:
                - shortcut_reasoning_risk: Indicates if the question can be answered via shortcuts
                - if_solvable: Indicates if the question can be answered by only using the given documents
                - unused_chunks: List of chunks not utilized in the answer
                - improvement_suggestions: Recommendations for exam enhancement
            - confidence: 1-5 confidence scale in the assessment
            You do not need to add anything else other than the JSON response

            {{
                "required_chunks": [int],
                "reasoning": dict {{
                    "if_solvable": bool,
                    "improvement_suggestions": str
                }},
                "confidence": int
            }}

            Specific Verification Criteria:
            - Multi-Hop Complexity: Assess if answering requires integrating information from multiple sources
            - Information Completeness: Evaluate whether all relevant document information is meaningfully incorporated
            - Reasoning Depth: Determine the level of analytical thinking needed to derive the correct answer

            Guidance for Analysis:
            - Critically examine each document chunk's contribution
            - Highlight potential ambiguities or gaps in the question-answer construct
            - Provide constructive feedback for improving exam quality and multi-hop reasoning challenge
            
            Question: {question}
            Options:
            A) {option_a}
            B) {option_b}
            C) {option_c}
            D) {option_d}
            Correct Answer: {correct_answer}
            Documents: {chunk_text}
            
            <end_of_turn>
            <start_of_turn>model
            Perform comprehensive multi-hop reasoning verification
            """,
            ModelType.MINISTRAL_8B: f"""<s>[INST]
            You are an advanced exam question verifier specializing in multi-hop reasoning and comprehensive document analysis.

            Verification Objectives:
            1. Multi-Hop Reasoning Verification
            - Determine if the question requires synthesizing information across multiple document chunks
            - Identify all necessary chunks for a comprehensive answer
            - Detect and prevent "shortcut reasoning" that bypasses critical information

            2. Question-Answer Integrity Assessment
            - Evaluate the coherence and correctness of the question and all options
            - Verify that the correct answer is truly supported by the given documents
            - Assess the depth and complexity of reasoning required

            3. Chunk Utilization Analysis
            - Confirm whether all provided document chunks are essential
            - Identify any unused or potentially relevant chunks
            - Recommend strategies to incorporate underutilized information

            Output Instructions:
            Generate a comprehensive JSON response with the following mandatory entities:
            - required_chunks: List of chunk indices essential for answering
            - reasoning:
                - shortcut_reasoning_risk: Indicates if the question can be answered via shortcuts
                - "unused_chunks": List of chunks not utilized in the answer
                - improvement_suggestions: Recommendations for including all the chunks' information
            - confidence: 1-5 confidence scale in the assessment
            You do not need to add anything else other than the JSON response

            {{
                "required_chunks": [int],
                "reasoning": dict {{
                    "shortcut_reasoning_risk": bool,
                    "unused_chunks": [int],
                    "improvement_suggestions": str
                }},
                "confidence": int
            }}

            Specific Verification Criteria:
            - Multi-Hop Complexity: Assess if answering requires integrating information from multiple sources
            - Information Completeness: Evaluate whether all relevant document information is meaningfully incorporated
            - Reasoning Depth: Determine the level of analytical thinking needed to derive the correct answer

            Guidance for Analysis:
            - Critically examine each document chunk's contribution
            - Highlight potential ambiguities or gaps in the question-answer construct
            - Provide constructive feedback for improving exam quality and multi-hop reasoning challenge
            [/INST]
            
            Question: {question}
            Options:
            A) {option_a}
            B) {option_b}
            C) {option_c}
            D) {option_d}
            Correct Answer: {correct_answer}
            Documents: {chunk_text}
            
            Perform comprehensive multi-hop reasoning verification
            </s>
            """,
        }
        return prompts.get(model_type, prompts[ModelType.LLAMA_3_2_3B])
    
    @staticmethod
    def get_regenerate_question_prompt(model_type: ModelType, question_data: dict, feedback: str):
        question = question_data["question"]
        choices = question_data["choices"]
        correct_answer = question_data["correct_answer"]
        chunks = question_data["documentation"]
        
        prompts = {
            ModelType.LLAMA_3_2_3B: f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Advanced Multi-Hop Question Iteration Methodology

            Objective:
            Systematically enhance the generated multi-hop multiple-choice question to ensure:
            - Comprehensive information synthesis across ALL document chunks
            - Elimination of potential reasoning shortcuts
            - Increased cognitive complexity and challenge

            Iteration Strategy:
            1. Comprehensive Chunk Integration Analysis
            - Rigorously examine the feedback from the verification process
            - Identify specific gaps in chunk utilization
            - Develop a strategic approach to incorporate underutilized information

            2. Reasoning Complexity Enhancement
            - Restructure the question to mandate full chunk integration
            - Create more sophisticated logical dependencies
            - Introduce additional layers of cognitive challenge

            3. Distractor Refinement
            - Redesign distractors to exploit more nuanced reasoning vulnerabilities
            - Ensure distractors require careful analysis of all chunks
            - Eliminate options that can be resolved through partial information

            Iteration Principles:
            - Mandatory full documentation utilization
            - Zero single-chunk solution pathways
            - Maximized cognitive engagement
            - Maintained academic rigor

            Output Instruction:
            CRITICAL: Your entire response MUST start with "Question:" followed by the completely redesigned question. Ensure the new question:
            CRITICAL: All the options MUST start with the corresponding letter such as "A)", "B)", "C)" or "D)".
            CRITICAL: The Correct Answer MUST start with "Correct Answer".
            - Addresses all verification feedback
            - Requires synthesising information from ALL chunks
            - Presents a more challenging reasoning scenario

            Output Format:
            Question: [Sophisticated multi-hop reasoning challenge]
            A) [Option A]
            B) [Option B]
            C) [Option C]
            D) [Option D]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            * Do not generate anything else other than the above output format

            Guidance for Iteration:
            - Deconstruct the original question's reasoning framework
            - Identify and eliminate any identified shortcut reasoning paths
            - Reconstruct the question to create a more intricate reasoning challenge
            - Ensure the new question is substantially more complex than the original

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Generated Exam Question:
            Question: {question}
            Choices: {choices}
            Correct Answer: {correct_answer}

            Original Chunks: {chunks}

            Verification Feedback: {feedback}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            Generate an enhanced multi-hop reasoning question
            """,
            ModelType.GEMMA2_9B: f"""
            <start_of_turn>user
            Advanced Multi-Hop Question Iteration Methodology

            Objective:
            Systematically enhance the generated multi-hop multiple-choice question to ensure:
            - Comprehensive information synthesis across ALL document chunks
            - Elimination of potential reasoning shortcuts
            - Increased cognitive complexity and challenge

            Iteration Strategy:
            1. Comprehensive Chunk Integration Analysis
            - Rigorously examine the feedback from the verification process
            - Identify specific gaps in chunk utilization
            - Develop a strategic approach to incorporate underutilized information

            2. Reasoning Complexity Enhancement
            - Restructure the question to mandate full chunk integration
            - Create more sophisticated logical dependencies
            - Introduce additional layers of cognitive challenge

            3. Distractor Refinement
            - Redesign distractors to exploit more nuanced reasoning vulnerabilities
            - Ensure distractors require careful analysis of all chunks
            - Eliminate options that can be resolved through partial information

            Iteration Principles:
            - Mandatory full documentation utilization
            - Zero single-chunk solution pathways
            - Maximized cognitive engagement
            - Maintained academic rigor

            Output Instruction:
            CRITICAL: Your entire response MUST start with "Question:" followed by the completely redesigned question. Ensure the new question:
            CRITICAL: All the options MUST start with the corresponding letter such as "A)", "B)", "C)" or "D)".
            CRITICAL: The Correct Answer MUST start with "Correct Answer".
            - Addresses all verification feedback
            - Requires synthesising information from ALL chunks
            - Presents a more challenging reasoning scenario

            Output Format:
            Question: [Sophisticated multi-hop reasoning challenge]
            A) [Option A]
            B) [Option B]
            C) [Option C]
            D) [Option D]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            * Do not generate anything else other than the above output format

            Guidance for Iteration:
            - Deconstruct the original question's reasoning framework
            - Identify and eliminate any identified shortcut reasoning paths
            - Reconstruct the question to create a more intricate reasoning challenge
            - Ensure the new question is substantially more complex than the original

            <end_of_turn>
            <start_of_turn>user
            
            Generated Exam Question:
            Question: {question}
            Choices: {choices}
            Correct Answer: {correct_answer}

            Original Chunks: {chunks}

            Verification Feedback: {feedback}
            
            <end_of_turn>
            <start_of_turn>model
            Generate an enhanced multi-hop reasoning question
            """,
            ModelType.MINISTRAL_8B: f"""
            <s>[INST]
            Advanced Multi-Hop Question Iteration Methodology

            Objective:
            Systematically enhance the generated multi-hop multiple-choice question to ensure:
            - Comprehensive information synthesis across ALL document chunks
            - Elimination of potential reasoning shortcuts
            - Increased cognitive complexity and challenge

            Iteration Strategy:
            1. Comprehensive Chunk Integration Analysis
            - Rigorously examine the feedback from the verification process
            - Identify specific gaps in chunk utilization
            - Develop a strategic approach to incorporate underutilized information

            2. Reasoning Complexity Enhancement
            - Restructure the question to mandate full chunk integration
            - Create more sophisticated logical dependencies
            - Introduce additional layers of cognitive challenge

            3. Distractor Refinement
            - Redesign distractors to exploit more nuanced reasoning vulnerabilities
            - Ensure distractors require careful analysis of all chunks
            - Eliminate options that can be resolved through partial information

            Iteration Principles:
            - Mandatory full documentation utilization
            - Zero single-chunk solution pathways
            - Maximized cognitive engagement
            - Maintained academic rigor

            Output Instruction:
            CRITICAL: Your entire response MUST start with "Question:" followed by the completely redesigned question. Ensure the new question:
            CRITICAL: All the options MUST start with the corresponding letter such as "A)", "B)", "C)" or "D)".
            CRITICAL: The Correct Answer MUST start with "Correct Answer".
            - Addresses all verification feedback
            - Requires synthesising information from ALL chunks
            - Presents a more challenging reasoning scenario

            Output Format:
            Question: [Sophisticated multi-hop reasoning challenge]
            A) [Option A]
            B) [Option B]
            C) [Option C]
            D) [Option D]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            * Do not generate anything else other than the above output format

            Guidance for Iteration:
            - Deconstruct the original question's reasoning framework
            - Identify and eliminate any identified shortcut reasoning paths
            - Reconstruct the question to create a more intricate reasoning challenge
            - Ensure the new question is substantially more complex than the original

            [/INST]
            
            Generated Exam Question:
            Question: {question}
            Choices: {choices}
            Correct Answer: {correct_answer}

            Original Chunks: {chunks}

            Verification Feedback: {feedback}
            
            Generate an enhanced multi-hop reasoning question
            </s>
            """,
        }
        
        return prompts.get(model_type, prompts[ModelType.LLAMA_3_2_3B])
