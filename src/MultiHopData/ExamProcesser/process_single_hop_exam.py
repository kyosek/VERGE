import json
import random
from copy import deepcopy


def has_null_values(question):
    """
    Check if a question dictionary has any null values in its main fields.

    Args:
        question (dict): Question dictionary

    Returns:
        bool: True if any main field has a null value, False otherwise
    """
    if not isinstance(question, dict):
        return True
    
    main_fields = ["question", "choices", "correct_answer"]
    return any(question.get(field) is None for field in main_fields)


def needs_cleaning(question):
    """
    Check if a question needs cleaning based on its choices.

    Args:
        question (dict): Question dictionary

    Returns:
        bool: True if question needs cleaning, False otherwise
    """
    # Check if question is None or empty
    if not question:
        return True

    if "choices" not in question:
        return True

    choices = question["choices"]
    valid_options = ["A) ", "B) ", "C) ", "D) "]

    # Check if exactly 4 choices
    if len(choices) != 4:
        return True

    # Check if all choices start with correct prefixes
    for i, choice in enumerate(choices):
        if not choice or not isinstance(choice, str):  # Check for None or non-string values
            return True
        if not choice.startswith(valid_options[i]):
            return True

    # Check if correct_answer needs cleaning
    if "correct_answer" in question:
        if question["correct_answer"] is None:  # Check for None value
            return True
        answer = question["correct_answer"].strip().upper()
        if len(answer) > 1 or answer not in ["A", "B", "C", "D"]:
            return True

    return False


def clean_question_choices(data):
    """
    Clean question data while maintaining the original format.
    Remove any invalid questions that cannot be cleaned or contain null values.

    Args:
        data (list): List of question dictionaries

    Returns:
        tuple: (List of cleaned questions, Number of questions cleaned, Number of questions removed)
    """
    cleaned_data = []
    questions_cleaned = 0
    questions_removed = 0

    for question in data:
        # Skip None, empty questions, or questions with null values
        if not question or has_null_values(question):
            questions_removed += 1
            continue

        try:
            if needs_cleaning(question):
                # Create a clean copy of the question
                cleaned_question = deepcopy(question)

                # Ensure choices list exists
                if "choices" not in cleaned_question:
                    cleaned_question["choices"] = []

                # Get current choices
                choices = cleaned_question["choices"]

                # Clean and standardize the choices
                cleaned_choices = []
                valid_options = ["A) ", "B) ", "C) ", "D) "]

                # Keep only the first 4 choices and ensure they match the format
                for i, choice in enumerate(choices[:4]):
                    if i < len(valid_options):
                        # Skip None or non-string choices
                        if choice is None or not isinstance(choice, str):
                            continue
                        # If the choice doesn't start with the correct prefix, add it
                        if not choice.startswith(valid_options[i]):
                            choice = valid_options[i] + choice.lstrip("ABCD) ")
                        cleaned_choices.append(choice)

                # If we have fewer than 4 choices, add empty ones
                while len(cleaned_choices) < 4:
                    cleaned_choices.append(valid_options[len(cleaned_choices)])

                # Update the choices in the cleaned question
                cleaned_question["choices"] = cleaned_choices

                # Clean the correct answer format if needed
                if "correct_answer" in cleaned_question:
                    # Remove questions with None correct_answer
                    if cleaned_question["correct_answer"] is None:
                        questions_removed += 1
                        continue
                    
                    answer = cleaned_question["correct_answer"].strip().upper()
                    # Extract just the letter if it includes more
                    answer = answer[0] if answer else ""
                    if answer not in ["A", "B", "C", "D"]:
                        questions_removed += 1
                        continue
                    cleaned_question["correct_answer"] = answer

                cleaned_data.append(cleaned_question)
                questions_cleaned += 1
            else:
                # Keep the original question unchanged
                cleaned_data.append(question)
        except Exception as e:
            # If any error occurs while cleaning a question, skip it
            questions_removed += 1
            continue

    return cleaned_data, questions_cleaned, questions_removed


def process_json_file(input_path, output_path, sample_size=1000, random_seed=42):
    """
    Read JSON file, clean the data, and save it back to a new JSON file.

    Args:
        input_path (str): Path to the input JSON file
        output_path (str): Path where the cleaned JSON file will be saved
    """
    random.seed(random_seed)

    try:
        # Read the input JSON file
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Ensure the sample size isn't larger than the dataset
        if sample_size > len(data):
            print(f"Warning: Sample size ({sample_size}) is larger than the dataset size ({len(data)})")
            sample_size = len(data)

        # Clean the data and get statistics
        cleaned_data, questions_cleaned, questions_removed = clean_question_choices(data)

        # Randomly sample entries
        sampled_data = random.sample(cleaned_data, sample_size)

        # Save the cleaned data to a new JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sampled_data, f, indent=2, ensure_ascii=False)

        print(f"Successfully processed {input_path}")
        print(f"- Total questions: {len(data)}")
        print(f"- Questions cleaned: {questions_cleaned}")
        print(f"- Questions removed: {questions_removed}")
        print(f"- Questions unmodified: {len(data) - questions_cleaned - questions_removed}")
        print(f"Results saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: Could not find the input file {input_path}")
    except json.JSONDecodeError:
        print(f"Error: The file {input_path} is not valid JSON")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    exam_file_names = ["llama_3_2_3b_single_hop_exam", "gemma2_9b_single_hop_exam", "ministral_8b_single_hop_exam"]
    
    for exam_file_name in exam_file_names:
        for task_domain in task_domains:
            input_file = f"auto-rag-eval/MultiHopData/{task_domain}/exams/{exam_file_name}.json"
            output_file = f"auto-rag-eval/MultiHopData/{task_domain}/exams/{exam_file_name}_processed.json"

            process_json_file(input_file, output_file, sample_size=150)
    