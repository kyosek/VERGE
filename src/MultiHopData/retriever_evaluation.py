import json
from typing import List, Dict, Any
from tqdm import tqdm
from MultiHopData.retriever import BaseRetriever, Chunk, ChunkRetriever
from MultiHopData.Solver.solve_exam_rag import BM25Retriever, ExamQuestion, FAISSRetriever, HybridRetriever


def load_exam(exam_file: str) -> List[ExamQuestion]:
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


def evaluate_retrieval(
    data: List[Dict[str, Any]], retriever: BaseRetriever, k: int = 5
) -> Dict[str, float]:
    total_queries = len(data)
    total_score = 0
    pass_at_n = 0

    for item in tqdm(data, desc="Evaluating retrieval"):
        query = item.question
        ground_truth = set(item.documentation)

        retrieved_docs = retriever.retrieve(query, k=k)
        retrieved_set = set(doc for doc, _ in retrieved_docs)

        # Calculate overlap between retrieved and ground truth
        overlap = len(ground_truth.intersection(retrieved_set))

        # Calculate score for this query
        score = overlap / len(ground_truth) if ground_truth else 0
        total_score += score

        # Check if at least one relevant document was retrieved
        if overlap > 0:
            pass_at_n += 1

    results = {
        "average_score": total_score / total_queries,
        "pass_at_n": (pass_at_n / total_queries) * 100,
        "total_queries": total_queries,
    }

    return results


def main(task_domain: str, retriever_type: str):
    # Load the chunk database
    chunk_retriever = ChunkRetriever(task_domain)
    chunk_retriever = chunk_retriever.load_database(
        f"MultiHopData/{task_domain}/chunk_database", task_domain
    )

    # Initialize the appropriate retriever
    if retriever_type == "Dense":
        retriever = FAISSRetriever(chunk_retriever)
    elif retriever_type == "Sparse":
        retriever = BM25Retriever([chunk.content for chunk in chunk_retriever.chunks])
    elif retriever_type == "Hybrid":
        faiss_retriever = FAISSRetriever(chunk_retriever)
        bm25_retriever = BM25Retriever([chunk.content for chunk in chunk_retriever.chunks])
        retriever = HybridRetriever([(faiss_retriever, 0.5), (bm25_retriever, 0.5)])
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    # Load the original JSONL data for queries and ground truth
    original_data = load_exam(f"MultiHopData/{task_domain}/exam_cleaned_1000_42.json")

    # Evaluate retrieval
    k_values = [1, 3, 5, 10, 20]
    for k in k_values:
        print(f"\nEvaluating retrieval for k={k}")
        results = evaluate_retrieval(original_data, retriever, k)
        print(f"Pass@{k}: {results['pass_at_n']:.2f}%")
        print(f"Average Score: {results['average_score']:.4f}")
        print(f"Total queries: {results['total_queries']}")


if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    retriever_types = ["Dense", "Sparse", "Hybrid"]

    for task_domain in task_domains:
        for retriever_type in retriever_types:
            print(f"\nEvaluating {task_domain}")
            print(f"Retriever: {retriever_type}")
            main(task_domain, retriever_type)
