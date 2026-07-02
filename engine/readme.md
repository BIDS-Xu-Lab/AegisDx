# Multi-Agent System for Medical Diagnosis

## Overview
This project implements a multi-agent system that processes cases stored in `test.json` and includes a RAG (Retrieval-Augmented Generation) agent for medical guideline retrieval.

### Prerequisites
- Python 3.6 or higher
- OpenAI API key

## Components

### 1. Multi-Agent System
The main system processes medical cases for diagnosis using multiple AI agents.

### 2. RAG Agent for Medical Guidelines
A simple RAG agent that can retrieve relevant medical guidelines from a guideline memory based on keywords using LlamaIndex with BAAI/bge-small-en-v1.5 embeddings.

## Usage

### Multi-Agent System
Run the multi-agent system with your OpenAI API key:

```bash
python main.py --api $OPENAI_KEY
```

Replace `$OPENAI_KEY` with your actual OpenAI API key or use the environment variable.

### RAG Agent

#### Setup
First, install the required dependencies:
```bash
python setup.py
```

Or manually install:
```bash
pip install -r requirements.txt
```

#### Basic Usage
```bash
python rag_agent.py --api-key $OPENAI_KEY --keywords "heart failure"
```

#### Interactive Mode
```bash
python test_rag_agent.py --interactive
```

#### Example Usage
```bash
python example_usage.py
```

#### Test the RAG Agent
```bash
python test_rag_agent.py
```

## Configuration
The test cases are configured in the `test.json` file. Modify this file to adjust the test scenarios for the multi-agent system.

The medical guidelines are stored in `data/mdm_eval.json` and contain real medical cases with diagnoses, AI predictions, and review comments.

## Features

### RAG Agent Features
- **Vector Search**: Uses LlamaIndex with BAAI/bge-small-en-v1.5 embeddings for semantic search
- **Efficient Retrieval**: Local embedding model reduces API costs and improves speed
- **Keyword Extraction**: Automatically extracts medical terms from cases
- **Fallback Search**: Includes keyword-based search as a fallback
- **Rich Output**: Provides detailed summaries including patient presentation, medical decisions, AI predictions, and review comments
- **Interactive Mode**: Allows interactive querying of the guideline database

### Supported Medical Domains
- Cardiovascular diseases
- Infectious diseases
- Cancer and malignancies
- Renal diseases
- Pulmonary conditions
- Pediatric conditions
- And many more medical specialties
