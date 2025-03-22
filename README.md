# VERGE: Verification-Augmented Generation of Multi-Hop Datasets for Evaluating Task-Specific RAG

<p align="center">
<a href="...">VERGE Paper</a>
</p>

## Overview

This repository contains the implementation of VERGE, a verification-augmented methodology for generating multi-hop datasets to evaluate Retrieval-Augmented Generation (RAG) systems. VERGE addresses significant methodological gaps in existing RAG evaluation frameworks by generating task-specific, multi-hop reasoning dataset.

## 🌟 Key Features

- **VERGE**: Implements a novel verification agent that ensures generated questions necessitate genuine multi-hop reasoning and maintain factual consistency
- **Multiple-Choice Format**: Isolates retrieval and reasoning capabilities from variability in answer generation
- **Hierarchical Error Taxonomy**: Provides structured analysis of RAG system failure patterns specifically in multi-hop reasoning contexts

## Repository Structure

- `dataset_generation/`: Scripts for generating verification-augmented multi-hop questions
- `verification_agent/`: Implementation of the verification agent for question validation and refinement
<!-- - `rag_evaluation/`: Evaluation frameworks for assessing RAG performance across configurations -->
- `error_analysis/`: Tools for hierarchical error categorization and analysis
- `prompts/`: Prompting templates for question generation, verification, and evaluation
- `configs/`: Configuration files for experiment reproduction

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### Download data (if necessary)

```bash
python data/
```

#### Chunking, Embedding and Storing

```bash
python data/
```

#### Generating Multi-hop Datasets with Verification Agent

```bash
python dataset_generation/generate_dataset.py --task_domain <domain> --output_path <path>
```

### Analyzing Error Patterns

```bash
python error_analysis/categorize_errors.py --results <results_path> --output <output_path>
```

## License

[MIT License](LICENSE)