# Medical AI Assistant

This project is a **Medical AI Assistant** that leverages advanced language models and retrieval-augmented generation (RAG) to answer medical-related queries. The assistant uses **Phi-3.5**, a large language model (LLM) fine-tuned on **PubMed-QA** using LoRA (Low-Rank Adaptation) for medical QA tasks, combined with a RAG system created by **LangChain** that retrieves information from multiple medical books. The project is accessible through a GUI built with **Streamlit**.

## Features

- **LLM**: `phi-3.5`, fine-tuned on PubMed-QA for answering medical queries.
- **RAG System**: Retrieval-augmented generation using LangChain, retrieving answers from a collection of books related to the medical field.
- **Streamlit GUI**: User-friendly web application interface for easy interaction with the AI assistant.

## Project Structure

```bash
├── notebooks/
│   ├── phi-3.5-qa-finetuning.ipynb        # Fine-tuning phi-3.5 on PubMed-QA using LoRA
│   ├── deployment.ipynb                   # Code and instructions for deploying the assistant
│   ├── books_retrieval.ipynb              # Downloads books for the RAG system
│   ├── RAG.ipynb                          # RAG system creation using LangChain
├── vectorstore/                           # Contains the vector index for document retrieval
├── app.py                                 # Streamlit app for the GUI
├── utils.py                               # Helper functions (loading LLM, chain setup, etc.)
```

## Usage

Once the application is up and running, you can interact with the AI assistant through the Streamlit interface. Simply type your medical-related questions, and the assistant will retrieve information from the medical books using the RAG system and answer using the fine-tuned `phi-3.5` model.

## 📖 References

- **Phi-3.5 (Base Model)**: [Hugging Face Model Card](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- **Fine-Tuned Adapter**: [Hugging Face Model Card](https://huggingface.co/oyounis/Phi-3.5-instruct-pubmedQA)
- **PubMed-QA Dataset**: [Hugging Face Card](https://huggingface.co/datasets/qiaojin/PubMedQA)
