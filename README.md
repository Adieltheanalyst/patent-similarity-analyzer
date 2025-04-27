# Patent Similarity Analyzer

## Overview
This Patent Similarity Analyzer is a tool for analyzing and comparing patent documents. It extracts text from uploaded PDF files, processes the text, and performs various analyses to assess the similarity between two patents. The tool provides insights through exploratory data analysis (EDA) and similarity analysis, including word frequency, bigram analysis, and different types of similarity metrics (TF-IDF, Jaccard, BERT). Additionally, it offers a summarization feature for long patent texts.

## Features
- **PDF Text Extraction:** Extracts content from uploaded patent PDF files.
- **Text Preprocessing:** Cleans and preprocesses the extracted text by removing unwanted characters, OCR errors, and document-specific text.
- **Domain-Specific Token Filtering:** Removes stopwords while retaining domain-specific terms.

### Exploratory Data Analysis (EDA)
- Word frequency analysis
- Word cloud generation
- Word frequency plots
- Bigram co-occurrence analysis

### Similarity Analysis:
- TF-IDF similarity
- Jaccard similarity
- BERT-based contextual similarity

- **Text Summarization:** Summarizes lengthy patent texts.
- **Downloadable Summaries:** Provides download buttons for text summaries.

## Requirements
To run this project, ensure you have the following Python packages installed:
- `streamlit`
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `fitz (PyMuPDF)`
- `nltk`
- `spacy`
- `transformers`
- `sentence-transformers`
- `wordcloud`

You can install the required libraries with:

```bash
pip install streamlit pandas numpy seaborn matplotlib fitz nltk spacy transformers sentence-transformers wordcloud
