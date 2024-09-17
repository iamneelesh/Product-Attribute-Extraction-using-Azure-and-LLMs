
# Product Attribute Extraction using Azure and LLMs

This repository contains scripts and modules that process product images and extract relevant attributes like quantity and unit using large language models (LLMs) and Azure Inference API.

## Overview

The project processes a CSV file of product images and names, sending the images and prompts to the Azure-based LLM (such as the Phi model) to extract relevant product attributes. It uses:
- Azure Inference API to query an LLM for responses based on image and text prompts.
- Pandas to handle CSV processing and storage of results.
- Regular expressions to extract JSON-like responses from the LLM.

The processing is resumable, allowing it to pick up from where it left off in case of interruptions.

## Setup

### Prerequisites

- Python 3.7+
- Azure API subscription (access to Azure Inference API)
- Install necessary libraries using `pip`:



# Environment Variables
Make sure to set up the following environment variables in your system:

AZURE_API_KEY: Your Azure Inference API key
AZURE_ENDPOINT: Your Azure model endpoint
# Files in the Repository
generate_submission.py: Main script to process images and CSV data, calling the get_phi_response function to extract product attributes.
llm_processor.py: Handles prompt generation logic for entity names and other LLM processing tasks.
phi_model.py: Interfaces with the Azure Inference API to send text and image-based queries.
# How It Works
CSV Input: The script reads a CSV file (test.csv) containing product images and entity names.
LLM Query: For each row, it generates a prompt based on the entity name and sends the image URL and prompt to the Azure-based LLM.
Response Processing: The response is processed to extract relevant details like quantity and unit from a JSON-like structure.
Output: The results are saved to submission.csv, and the script keeps track of its progress, so it can resume from the last processed index in case of interruptions.
