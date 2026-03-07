# Sentiment and Bias Analysis of Cyber Threat Intelligence Reports

A Python-based NLP pipeline that analyzes cyber threat intelligence (CTI) documents for **sentiment**, **bias**, and **latent topics**. The tool ingests reports from diverse sources — government agencies, security vendors, mainstream media, independent researchers, and international outlets — and quantifies how each source frames cybersecurity threats.

## Motivation

Cyber threat reports are rarely neutral. A vendor advisory may inflate risk to sell products, a government bulletin may attribute attacks to foreign adversaries for political reasons, and a news article may sensationalize events for clicks. This project provides a reproducible, data-driven framework to surface those patterns.

## Features

- **Multi-format ingestion** — reads `.txt`, `.pdf`, and `.docx` files from a directory or a `.zip` archive.
- **Language detection & translation** — automatically detects non-English documents (via `langdetect`) and translates them to English (via `deep_translator` / Google Translate) before analysis.
- **Text preprocessing** — cleans raw text (URL/email removal, lowercasing, punctuation filtering) and produces lemmatized token lists ready for NLP.
- **Source classification** — heuristically labels each document as *government*, *vendor*, *media*, *research*, *international*, or *unknown* based on filename/path keywords.
- **VADER sentiment analysis** — rule-based sentiment scoring at both the document and sentence level.
- **Transformer sentiment analysis** (optional) — deep learning sentiment via `distilbert-base-uncased-finetuned-sst-2-english` with automatic chunking for long documents.
- **Bias detection** — keyword-lexicon scanning across four bias categories:
  - **Commercial** — promotional language, sales CTAs, fear appeals, product mentions.
  - **Political** — state attribution, nationalistic framing, blame language, policy advocacy.
  - **Geopolitical** — named threat actors, cyber-warfare framing, alliance language, escalation rhetoric.
  - **Sensationalism** — hyperbole, fear words, artificial urgency, speculation hedging.
- **Context-aware bias analysis** — overlays sentence-level sentiment on flagged bias sentences to distinguish positive spin from fear-driven framing.
- **LDA topic modeling** — discovers latent themes across the corpus using Gensim's LDA implementation, with coherence scoring and optional grid search for optimal topic count.
- **Visualizations** — generates four publication-ready charts saved as PNGs:
  1. Sentiment distribution by source type (box plot + stacked bar).
  2. Bias vs. sentiment correlation (scatter plot + grouped bar).
  3. Bias pattern dashboard (heatmap, bar chart, dominant-bias counts, score histogram).
  4. Topic distribution (horizontal bar + topic-by-source heatmap).
- **CSV export** — all per-document results (sentiment scores, bias scores, dominant topic, etc.) are exported to a single CSV for further analysis.

## Project Structure

```
├── Sentiment_and_Bias_Analysis_of_Cyber_Threat_Media.py   # Main script
├── Project 1 Files.zip    # (optional) Zipped input documents
├── output/                # Auto-created at runtime
│   ├── analysis_results.csv
│   ├── viz1_sentiment_distribution.png
│   ├── viz2_bias_sentiment.png
│   ├── viz3_bias_patterns.png
│   └── viz4_topics.png
└── README.md
```

## Requirements

**Python 3.8+**

### Core dependencies

| Package | Purpose |
|---|---|
| `pandas` | Tabular data handling |
| `numpy` | Numerical operations |
| `nltk` | Tokenization, stopwords, lemmatization |
| `vaderSentiment` | Rule-based sentiment analysis |
| `gensim` | LDA topic modeling |
| `matplotlib` | Plotting |
| `seaborn` | Heatmap visualization |
| `langdetect` | Language detection |
| `deep_translator` | Google Translate wrapper |

### Optional dependencies

| Package | Purpose |
|---|---|
| `transformers` | Transformer-based sentiment (DistilBERT) |
| `pdfplumber` | PDF text extraction |
| `python-docx` | Word document extraction |

### Install

```bash
pip install pandas numpy nltk vaderSentiment gensim matplotlib seaborn langdetect deep_translator
```

For full functionality (PDF/DOCX support and transformer sentiment):

```bash
pip install pdfplumber python-docx transformers torch
```

NLTK data is downloaded automatically on first run.

## Usage

### 1. Prepare your data

Place your cyber threat documents (`.txt`, `.pdf`, `.docx`) in one of two locations:

- **Zip archive** — name it `Project 1 Files.zip` and place it in the same directory as the script.
- **Flat directory** — place files directly alongside the script (or configure `data_dir` in Section 2).

### 2. Run the analysis

```bash
python Sentiment_and_Bias_Analysis_of_Cyber_Threat_Media.py
```

The script will print progress to the console as it moves through five phases: ingestion, processing (sentiment + bias + topics), console reporting, visualization generation, and CSV export.

### 3. Review outputs

All results are written to the `output/` directory:

- `analysis_results.csv` — one row per document with columns for filename, source type, language, word count, VADER scores, bias scores per category, dominant bias, dominant topic, and topic probability.
- `viz1–viz4` PNG files — ready to embed in reports or presentations.

## Configuration

Key settings can be adjusted near the top of the script (Section 2):

- `data_dir` — path to the folder containing input documents.
- `output_dir` — path where results and charts are saved.
- `num_topics` — number of LDA topics (default: 5). Use `find_optimal_topics()` for data-driven selection.
- `use_transformer` — set to `True` in `run_analysis_pipeline()` to enable DistilBERT sentiment alongside VADER.

## Pipeline Overview

```
Input Documents
      │
      ▼
  Extract Text (PDF / DOCX / TXT)
      │
      ▼
  Detect Language → Translate if non-English
      │
      ▼
  Clean Text → Tokenize → Lemmatize
      │
      ▼
  ┌─────────────────┬──────────────────┬──────────────────┐
  │  VADER Sentiment │  Bias Detection  │  LDA Topics      │
  │  (+ Transformer) │  (4 categories)  │  (Gensim)        │
  └────────┬────────┴────────┬─────────┴────────┬─────────┘
           │                 │                   │
           ▼                 ▼                   ▼
        Results DataFrame (merged)
              │
              ▼
     Visualizations + CSV Export
```

## Author

**Johnathan Duque**
