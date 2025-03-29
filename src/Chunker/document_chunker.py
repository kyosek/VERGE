import os
import json
from abc import ABC, abstractmethod
import spacy
import nltk
import uuid
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any, Optional
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SpacyTextSplitter,
    LatexTextSplitter,
)
from tqdm import tqdm
from langchain_community.document_loaders import TextLoader
from transformers import AutoTokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class DocumentChunker:
    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200):
        """Initialise the DocumentChunker with configurable parameters.

        Args:
            chunk_size: The size of each chunk in characters
            chunk_overlap: The overlap between chunks in characters
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single text file and return its chunks in the desired format."""
        # Generate a doc_id from the filename
        doc_id = os.path.splitext(os.path.basename(file_path))[0]

        # Load and process the document
        loader = TextLoader(file_path)
        document = loader.load()[0]

        # Get the full content
        full_content = document.page_content

        # Generate chunks
        chunk_texts = self.text_splitter.split_text(full_content)

        # Create chunks with metadata
        chunks = []
        for idx, chunk_text in enumerate(chunk_texts):
            chunk = {
                "chunk_id": f"{doc_id}_chunk_{idx}",
                "original_index": idx,
                "content": chunk_text,
            }
            chunks.append(chunk)

        # Create the document entry
        doc_entry = {
            "doc_id": doc_id,
            "original_uuid": uuid.uuid4().hex[:4],
            "content": full_content,
            "chunks": chunks,
        }

        return doc_entry

    def process_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """Process all text files in a directory and return a list of processed documents."""
        processed_docs = []

        # Process each .txt file in the directory
        for filename in tqdm(os.listdir(dir_path)):
            if filename.endswith(".txt"):
                file_path = os.path.join(dir_path, filename)
                doc_entry = self.process_file(file_path)
                processed_docs.append(doc_entry)

        return processed_docs

    def save_to_json(self, processed_docs: List[Dict[str, Any]], output_path: str):
        """Save the processed documents to a JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_docs, f, indent=2, ensure_ascii=False)


class ChunkStrategy(ABC):
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass

class RecursiveChunkStrategy(ChunkStrategy):
    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def split_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

class SemanticChunkStrategy(ChunkStrategy):
    """Chunks text based on semantic similarity and natural boundaries."""
    
    def __init__(
        self, 
        target_chunk_size: int = 2000,
        similarity_threshold: float = 0.5,
    ):
        self.target_chunk_size = target_chunk_size
        self.similarity_threshold = similarity_threshold
        self.nlp = spacy.load("en_core_web_sm")
        
    def split_text(self, text: str) -> List[str]:
        # First split into sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Calculate sentence embeddings using TF-IDF for efficiency
        vectorizer = TfidfVectorizer()
        sentence_vectors = vectorizer.fit_transform(sentences)
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.target_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Check semantic similarity with previous sentences in chunk
            if current_chunk:
                current_vec = sentence_vectors[i]
                prev_vec = sentence_vectors[i-1]
                similarity = (current_vec * prev_vec.T).toarray()[0][0]
                
                if similarity < self.similarity_threshold and current_length > self.target_chunk_size * 0.5:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

class TopicBasedChunkStrategy(ChunkStrategy):
    """Chunks text based on topic changes and section boundaries."""
    
    def __init__(
        self, 
        min_chunk_size: int = 2000,
        max_chunk_size: int = 6000,
        topic_threshold: float = 0.3
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.topic_threshold = topic_threshold
        self.nlp = spacy.load("en_core_web_sm")
        
    def _detect_section_boundary(self, text: str) -> bool:
        """Detect if text contains section boundary markers."""
        boundary_patterns = [
            "\n\n",
            "\nSection",
            "\nCHAPTER",
            "\n\d+\.",
            "\n[A-Z][A-Z\s]+\n"
        ]
        return any(pattern in text for pattern in boundary_patterns)
    
    def _get_topic_signature(self, text: str) -> np.ndarray:
        """Generate topic signature using key terms."""
        doc = self.nlp(text)
        return np.array([
            token.vector 
            for token in doc 
            if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and not token.is_stop
        ]).mean(axis=0)
    
    def split_text(self, text: str) -> List[str]:
        # First split into paragraphs
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = []
        current_length = 0
        prev_topic_sig = None
        
        for para in paragraphs:
            para_length = len(para)
            
            # Check if adding this paragraph exceeds max chunk size
            if current_length + para_length > self.max_chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
                prev_topic_sig = None
            
            # Check for topic change if we have enough content
            if current_length > self.min_chunk_size:
                topic_sig = self._get_topic_signature(para)
                
                if prev_topic_sig is not None:
                    similarity = np.dot(topic_sig, prev_topic_sig) / (
                        np.linalg.norm(topic_sig) * np.linalg.norm(prev_topic_sig)
                    )
                    
                    if similarity < self.topic_threshold or self._detect_section_boundary(para):
                        if current_chunk:
                            chunks.append("\n\n".join(current_chunk))
                            current_chunk = []
                            current_length = 0
                
                prev_topic_sig = topic_sig
            
            current_chunk.append(para)
            current_length += para_length
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks

class EnhancedDocumentChunker:
    def __init__(self, strategy: ChunkStrategy):
        """Initialise the DocumentChunker with a specific chunking strategy."""
        self.strategy = strategy
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single text file using the selected chunking strategy."""
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        loader = TextLoader(file_path)
        document = loader.load()[0]
        full_content = document.page_content
        
        # Generate chunks using the selected strategy
        chunk_texts = self.strategy.split_text(full_content)
        
        chunks = [
            {
                "chunk_id": f"{doc_id}_chunk_{idx}",
                "original_index": idx,
                "content": chunk_text,
            }
            for idx, chunk_text in enumerate(chunk_texts)
        ]
        
        return {
            "doc_id": doc_id,
            "original_uuid": uuid.uuid4().hex[:4],
            "content": full_content,
            "chunks": chunks,
        }
    
    def process_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """Process all text files in a directory."""
        processed_docs = []
        
        for filename in tqdm(os.listdir(dir_path)):
            if filename.endswith(".txt"):
                file_path = os.path.join(dir_path, filename)
                doc_entry = self.process_file(file_path)
                processed_docs.append(doc_entry)
        
        return processed_docs
    
    def save_to_json(self, processed_docs: List[Dict[str, Any]], output_path: str):
        """Save the processed documents to a JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_docs, f, indent=2, ensure_ascii=False)


def main(task_name: str, version: dict, chunk_size: int, strategy_name: str = "recursive"):
    # Choose chunking strategy
    strategies = {
        "recursive": RecursiveChunkStrategy(chunk_size=4000, chunk_overlap=200),
        "semantic": SemanticChunkStrategy(target_chunk_size=chunk_size),
        "topic": TopicBasedChunkStrategy(min_chunk_size=2000, max_chunk_size=6000)
    }
    
    strategy = strategies[strategy_name]
    chunker = EnhancedDocumentChunker(strategy)
    
    # Process documents
    docs_dir = f"MultiHopData/{task_name}/raw_texts"
    output_path = f"MultiHopData/{task_name}/chunks/docs_chunk_{strategy_name}_{version}.json"
    os.makedirs(docs_dir, exist_ok=True)
    
    processed_docs = chunker.process_directory(docs_dir)
    chunker.save_to_json(processed_docs, output_path)
    
    print(f"Processed {len(processed_docs)} documents using {strategy_name} strategy")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    task_names = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    chunk_strategy = "semantic"
    version_configs = [{"v3": 4000}, {"v5": 512}, {"v6": 1024}, {"v7": 2048}]
    
    for task_name in task_names:
        for version_config in version_configs:
            version = list(version_config.keys())[0]
            chunk_size = version_config[str(version)]
            print(f"Chunking {task_name}")
            print(f"Version: {version} - {chunk_size}")
            main(task_name, version, chunk_size, chunk_strategy)
