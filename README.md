# CyberSentinel

**A sentiment and bias analysis toolkit for cyber threat intelligence reports.**

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![NLP](https://img.shields.io/badge/NLP-VADER%20%2B%20Transformers-blueviolet)
![LDA](https://img.shields.io/badge/Topic%20Modeling-Gensim%20LDA-orange)

CyberSentinel ingests cybersecurity reports from multiple source types — government advisories, vendor whitepapers, news articles, academic papers, and international publications, and measures **how** they frame threats, not just **what** they report. It detects emotional tone via dual sentiment engines (VADER + DistilBERT), scans for four categories of editorial bias using custom lexicons, discovers latent topics with LDA, and produces a rich set of visualizations and a structured CSV export.

> **Academic Context:** Originally developed as a research project exploring how different stakeholders in the cybersecurity ecosystem frame the same threats with different language, tone, and intent.

---

## Table of Contents

- [Features](#features)
- [Bias Detection Categories](#bias-detection-categories)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Pipeline Overview](#pipeline-overview)
- [Project Structure](#project-structure)
- [Visualizations](#visualizations)
- [How It Works](#how-it-works)
- [Extending the Toolkit](#extending-the-toolkit)
- [Learning Outcomes](#learning-outcomes)

---

## Features

| Module | Description |
|--------|-------------|
| **Multi-Format Ingestion** | Reads `.txt`, `.pdf`, and `.docx` files from directories or `.zip` archives with recursive traversal and automatic format detection |
| **Language Detection & Translation** | Detects non-English documents via `langdetect` and translates them to English using Google Translate with chunking and rate-limit handling |
| **Text Preprocessing** | Cleans raw text (URL/email removal, regex normalization), tokenizes, removes stopwords + domain-specific noise words, and lemmatizes via NLTK |
| **Source Classification** | Heuristic keyword classifier that labels documents as government, vendor, media, research, international, or unknown based on filename and path |
| **VADER Sentiment** | Rule-based sentiment scoring optimized for news and social media content. Returns compound, positive, neutral, and negative scores per document |
| **Transformer Sentiment** | Deep learning sentiment via DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) with intelligent sentence chunking and voting-based aggregation |
| **Sentence-Level Sentiment** | Breaks documents into individual sentences and scores each one to surface localized emotional spikes hidden in otherwise neutral text |
| **Bias Detection Engine** | Scans for four bias categories (commercial, political, geopolitical, sensationalism) using curated lexicons with density scoring and context extraction |
| **Context-Aware Bias** | Combines keyword detection with sentence-level sentiment to distinguish between neutral mentions and emotionally charged usage of bias indicators |
| **LDA Topic Modeling** | Gensim-based Latent Dirichlet Allocation with automatic dictionary filtering, coherence scoring (`c_v`), and optional grid search for optimal topic count |
| **Visualization Suite** | Four publication-ready matplotlib/seaborn plots: sentiment distribution, bias-sentiment correlation, bias pattern heatmap dashboard, and topic modeling results |
| **CSV Export** | Flattened DataFrame export with all scores, labels, and metadata ready for further analysis in Excel, R, or Jupyter |

---

## Bias Detection Categories

| Category | What's Detected | Example Indicators |
|----------|-----------------|--------------------|
| **Commercial** | Vendor marketing disguised as threat intelligence, FUD tactics, product self-promotion | "industry-leading", "free trial", "our platform", "devastating", "unprecedented" |
| **Political** | State-attribution framing, nationalistic language, blame-oriented narratives, policy advocacy | "state-sponsored", "national security", "behind the attack", "sanctions" |
| **Geopolitical** | Cyber-conflict framing, alliance signaling, named threat actor emphasis, escalation language | "APT28", "cyber warfare", "Five Eyes", "escalation", "retaliation" |
| **Sensationalism** | Clickbait hyperbole, fear-mongering, artificial urgency, speculative hedging | "shocking", "nightmare", "breaking", "allegedly", "could", "reportedly" |

Each category contains subcategories (e.g., Commercial → promotional, sales, fear_appeal, product_mention) with curated keyword lists. Bias scores are calculated as keyword density normalized against document length, capped at 1.0.

---

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/yourusername/cybersentinel.git
cd cybersentinel
pip install -r requirements.txt
```

### Dependencies

```
pandas
numpy
nltk
vaderSentiment
gensim
matplotlib
seaborn
langdetect
deep-translator
pdfplumber
python-docx
transformers
torch
```

### Run the Analysis

```bash
# Option A — Place a zip archive of reports next to the script
# The script looks for 'Project 1 Files.zip' by default
python Sentiment_and_Bias_Analysis_of_Cyber_Threat_Media.py

# Option B — Place documents in the data directory
# Edit the 'data_dir' variable in Section 2 to point to your folder
python Sentiment_and_Bias_Analysis_of_Cyber_Threat_Media.py
```

Results are saved to the `./output/` directory, including four PNG visualizations and a CSV file with all computed scores.

---

## Usage

### Input Formats

| Format | Support |
|--------|---------|
| `.txt` | Full support (UTF-8 with fallback) |
| `.pdf` | Full support via pdfplumber (text-based PDFs) |
| `.docx` | Full support via python-docx |
| `.doc` | Not supported (legacy binary format) |
| `.zip` | Automatic extraction and recursive processing |

### Configuration

Key settings are in **Section 2** of the script:

```python
# Point to your data folder (leave empty string to use script directory)
data_dir = os.path.join(main_dir, '') + os.sep

# Output directory for visualizations and CSV
output_dir = os.path.join(main_dir, 'output') + os.sep
```

### Enabling Transformer Sentiment

By default, only VADER runs (fast, no GPU required). To enable the DistilBERT transformer model:

```python
# In main(), change use_transformer to True
df = run_analysis_pipeline(documents, use_transformer=True)
```

> **Note:** First run will download the `distilbert-base-uncased-finetuned-sst-2-english` model (~260 MB). Runs on CPU by default.

---

## Pipeline Overview

```
1. INGEST        → Load files from zip archive or directory tree
2. DETECT LANG   → Identify non-English documents (langdetect)
3. TRANSLATE     → Auto-translate to English via Google Translate API
4. CLEAN         → Remove URLs, emails, normalize whitespace, lowercase
5. TOKENIZE      → Word tokenization, stopword removal, lemmatization
6. CLASSIFY      → Label source type from filename/path keywords
7. SENTIMENT     → VADER (fast) + optional DistilBERT (deep) scoring
8. BIAS SCAN     → Keyword density analysis across 4 bias categories
9. TOPIC MODEL   → LDA topic discovery with coherence evaluation
10. VISUALIZE    → Generate 4 publication-ready plots
11. EXPORT       → Save all results to CSV
```

---

## Project Structure

```
cybersentinel/
├── Sentiment_and_Bias_Analysis_of_Cyber_Threat_Media.py   # Main script (all-in-one)
├── requirements.txt                                        # Python dependencies
├── Project 1 Files.zip                                     # Input data (your reports)
├── output/
│   ├── viz1_sentiment_distribution.png                     # Box plot + stacked bar
│   ├── viz2_bias_sentiment.png                             # Scatter + grouped bar
│   ├── viz3_bias_patterns.png                              # 4-panel bias dashboard
│   ├── viz4_topics.png                                     # Topic word weights + distribution
│   └── analysis_results.csv                                # Full results table
└── README.md
```

---

## Visualizations

The toolkit generates four plots automatically:

| Plot | Contents |
|------|----------|
| **Sentiment Distribution** | Box plot of VADER compound scores by source type + stacked bar chart of positive/neutral/negative document counts |
| **Bias vs Sentiment** | Scatter plot correlating commercial bias intensity with sentiment polarity + grouped bar chart of averages by source |
| **Bias Pattern Dashboard** | 4-panel view heatmap of bias scores across sources, grouped bar comparison, dominant bias frequency, and overall score histogram |
| **Topic Modeling Results** | Horizontal bar chart of top words per LDA topic + grouped bar chart showing topic distribution across source types |

All plots are saved at 300 DPI to `./output/` and displayed interactively via `plt.show()`.

---

## How It Works

### Sentiment Analysis

The toolkit uses a **dual-engine approach** to sentiment:

**VADER** is a rule-based model that uses a curated lexicon of words with pre-assigned sentiment scores, plus grammar rules for handling negation, capitalization, and punctuation emphasis. It returns a compound score from -1 (most negative) to +1 (most positive), with standard thresholds at ±0.05 for classification.

**DistilBERT** (optional) is a transformer model fine-tuned on the Stanford Sentiment Treebank. For long documents, the text is split into sentence-based chunks of ~500 characters, each chunk is classified independently, and results are aggregated via a voting mechanism that averages positive and negative confidence scores separately.

### Bias Detection

Bias detection works through **keyword density analysis** rather than machine learning. Four curated lexicons define indicator words for commercial, political, geopolitical, and sensationalist bias. For each document, the scanner counts keyword occurrences, extracts surrounding context snippets as evidence, and calculates a density score: `(total hits / total words) × 100`, capped at 1.0.

The **context-aware mode** goes further by isolating sentences that contain bias keywords and running VADER sentiment on each one — distinguishing between neutral technical mentions (e.g., "the solution mitigated the threat") and emotionally loaded usage (e.g., "only our industry-leading solution can stop this devastating attack").

### Topic Modeling

LDA (Latent Dirichlet Allocation) treats each document as a mixture of topics and each topic as a mixture of words. The model is trained on the preprocessed token lists with automatic dictionary filtering (removing words appearing in fewer than 2 documents or more than 90% of documents). Topic quality is measured via the `c_v` coherence metric, and an optional grid search tests topic counts from 3 to 10 to find the optimal number.

---

## Extending the Toolkit

### Add Custom Bias Lexicons

```python
# Add a new category to the BIAS_LEXICONS dictionary
BIAS_LEXICONS['technical_hype'] = {
    'buzzwords': ['zero-trust', 'blockchain', 'quantum-resistant', 'AI-powered',
                  'next-gen', 'revolutionary', 'paradigm shift'],
    'vaporware': ['coming soon', 'roadmap', 'planned', 'future release',
                  'beta', 'early access']
}
# comprehensive_bias_analysis() will automatically pick it up
```

### Add New File Formats

```python
# In extract_text_from_file(), add a new elif block
elif file_ext == '.html':
    from bs4 import BeautifulSoup
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        text = soup.get_text(separator='\n')
```

### Add New Source Types

```python
# In classify_source_type(), add a new keyword list and elif block
threat_intel_keywords = ['threat', 'intelligence', 'ioc', 'indicator',
                         'malware', 'campaign', 'apt']

# Add before the 'unknown' fallback
elif any(kw in combined for kw in threat_intel_keywords):
    return 'threat_intelligence'
```

### Use the Functions Independently

Each function is designed to work standalone:

```python
# Quick sentiment check on any text
result = detect_sentiment_vader("This ransomware attack was devastating for the company.")
print(result)  # {'compound': -0.5859, 'sentiment': 'negative', ...}

# Bias scan on a single string
bias = detect_bias_indicators("Our industry-leading solution provides comprehensive protection.", 
                               bias_type='commercial')
print(bias['score'])  # 0.2857

# Context-aware bias with sentiment overlay
contexts = analyze_bias_context("The unprecedented attack was catastrophic.", 
                                 bias_type='sensationalism')
```

---

## Sample Output

```
══════════════════════════════════════════════════════════════════════
🕵️  CYBERSECURITY SENTIMENT & BIAS ANALYSIS
══════════════════════════════════════════════════════════════════════

📦 Found zip file: ./Project 1 Files.zip
📄 Processing: cisa_advisory_2024.pdf
  ✅ Loaded (3,421 words, government)
📄 Processing: crowdstrike_threat_report.pdf
  ✅ Loaded (5,102 words, vendor)
📄 Processing: reuters_cyberattack_coverage.txt
  ✅ Loaded (1,845 words, media)

📊 Loaded 3 documents

🔍 STARTING ANALYSIS PIPELINE
══════════════════════════════════════════════════════════════════════

📄 Analyzing [1/3]: cisa_advisory_2024.pdf
  📊 Running VADER sentiment analysis...
  🔎 Detecting bias indicators...
  ✅ Complete - Sentiment: negative, Dominant Bias: political

📄 Analyzing [2/3]: crowdstrike_threat_report.pdf
  📊 Running VADER sentiment analysis...
  🔎 Detecting bias indicators...
  ✅ Complete - Sentiment: positive, Dominant Bias: commercial

📄 Analyzing [3/3]: reuters_cyberattack_coverage.txt
  📊 Running VADER sentiment analysis...
  🔎 Detecting bias indicators...
  ✅ Complete - Sentiment: negative, Dominant Bias: sensationalism

✅ ANALYSIS PIPELINE COMPLETE
══════════════════════════════════════════════════════════════════════

📊 CREATING VISUALIZATIONS
💾 Results saved to: ./output/analysis_results.csv

🎉 ANALYSIS COMPLETE!
```
Graph's Created:
<img width="1400" height="600" alt="Figure_4" src="https://github.com/user-attachments/assets/9e72f246-f89b-4a8f-887a-081c290bd2e3" />
<img width="1706" height="958" alt="Figure_3" src="https://github.com/user-attachments/assets/c74bdb5b-22ee-4cfb-9f5d-964aba4135cd" />
<img width="1400" height="600" alt="Figure_2" src="https://github.com/user-attachments/assets/5cfe7717-847b-4539-9f83-1cd71ff92d36" />
<img width="1400" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/2adc5a34-2d3c-40dd-908f-112de67d2410" />

---

## Learning Outcomes

| Topic | What You'll Learn |
|-------|-------------------|
| **NLP Pipelines** | Building an end-to-end text processing pipeline: extraction, cleaning, tokenization, lemmatization |
| **Sentiment Analysis** | Comparing rule-based (VADER) vs deep learning (transformer) approaches and when each excels |
| **Bias Detection** | Designing domain-specific lexicons and understanding the limits of keyword-based analysis |
| **Topic Modeling** | How LDA discovers latent themes, tuning hyperparameters, and interpreting coherence scores |
| **Data Visualization** | Creating multi-panel matplotlib/seaborn dashboards for research communication |
| **Media Literacy** | Recognizing how identical cyber events are framed differently by governments, vendors, and journalists |
| **Multilingual NLP** | Handling language detection, translation APIs, chunking for API limits, and rate limiting |
| **Research Methods** | Structuring reproducible analysis pipelines with clear separation of concerns |

---

<p align="center">
  <b>Analyze the narrative, not just the threat.</b><br>
  <sub>Created by <b>Johnathan Duque</b></sub>
</p>
