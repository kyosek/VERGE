import json
import logging
from datetime import datetime
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SecFillingsData:
    def __init__(self, max_char_length: int, min_char_length: int):
        self.max_char_length = max_char_length
        self.min_char_length = min_char_length
        self.sections = [
            "section_2",
        ]

    def process_section(self, row: Dict, section: str) -> Dict:
        content = row[section]
        if (
            not content
            or len(content) < self.min_char_length
            or len(content) > self.max_char_length
        ):
            return None

        return {
            "filename": row["filename"],
            "content": content,
        }

    def preprocess(self, text: str) -> str:
        # Add any specific preprocessing steps here if needed
        return text.strip()

    def load_save_data(self) -> None:
        dataset = load_dataset("eloukas/edgar-corpus", split="train")

        processed_data = []

        for row in tqdm(dataset):
            for section in self.sections:
                processed_section = self.process_section(row, section)
                if processed_section:
                    processed_data.append(processed_section)

        logger.info(f"Processed {len(processed_data)} sections from {len(dataset)} documents.")

        os.makedirs("output", exist_ok=True)

        for item in processed_data:
            filename = f"{item['filename']}.txt"
            file_path = os.path.join("output", filename)
            with open(file_path, "w") as f:
                f.write(item["content"])

        logger.info(f"Saved processed data to the 'output' directory")


if __name__ == "__main__":
    edgar_data = SecFillingsData(max_char_length=200_000, min_char_length=20_000)
    edgar_data.load_save_data()
