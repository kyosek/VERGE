from datasets import load_dataset
import os


def save_contexts_to_files(datasets, output_dir="output"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    try:
        data = load_dataset("THUDM/LongBench", dataset_name, split="test")

        # Process each entry in the dataset
        for entry in data:
            # Extract the first 4 characters of the _id
            short_id = str(entry["_id"])[:4]

            # Create filename
            filename = f"{dataset_name}_{short_id}.txt"
            filepath = os.path.join(output_dir, filename)

            # Save the context to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(entry["context"])

        print(f"Completed saving contexts for {dataset_name}")

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        pass


if __name__ == "__main__":
    # List of datasets to process
    datasets = ["2wikimqa"]
    datasets = ["2wikimqa", "qasper", "multifieldqa_en", "hotpotqa", "gov_report"]

    # Process each dataset
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        save_contexts_to_files(dataset_name, f"MultiHopData/{dataset_name}/")
