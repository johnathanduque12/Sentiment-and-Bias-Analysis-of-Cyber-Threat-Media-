# =============================================================================
# SENTIMENT AND BIAS ANALYSIS OF CYBER THREAT INTELLIGENCE REPORTS

# Name: Johnathan Duque
# Student Number: 501251971
# =============================================================================

# ======================= SECTION 1: IMPORTS =======================

# --- STANDARD DATA SCIENCE & UTILITIES ---
import pandas as pd # The industry standard for handling tabular data (rows/columns).
# USED IN: run_analysis_pipeline (to store results), create_bias_pattern_visualization (to aggregate stats).

import numpy as np #Performs heavy mathematical operations (averages, matrices).
# USED IN: comprehensive_bias_analysis (calculating mean scores), create_sentiment_distribution_plot (generating colors).

import os #Allows Python to talk to the Operating System (finding files/folders).
# USED IN: load_documents_from_directory (walking folder trees), extract_text_from_file (checking extensions).

import string #Contains constants like list of punctuation characters.
# USED IN: preprocess_for_nlp (to remove punctuation from tokens).

import re  # "Regular Expressions" - finds complex text patterns.
# USED IN: clean_text (removing emails/URLs), perform_topic_modeling (parsing the output string).

import zipfile # Tools to open and read .zip archives.
# USED IN: load_documents_from_zip.

import time # Handles time delays and measurements.
# USED IN: translate_text (to pause execution so we don't get blocked by the API).

from datetime import datetime # Handles timestamps.
# USED IN: (Optional) Useful if you want to timestamp your output files in 'main'.

from collections import Counter # A specialized tool for counting items in a list quickly.
# USED IN: (Implicitly) Useful for counting word frequencies in 'preprocess_for_nlp'.

import warnings
import logging
# WHAT IT DOES: Control system messages. 'warnings' can suppress "FutureWarning" messages 
# from libraries like Pandas/Gensim to keep your console output clean.


# --- NATURAL LANGUAGE PROCESSING (NLP) ---
import nltk # helps with text processing.
from nltk.corpus import stopwords
# USED IN: preprocess_for_nlp (to remove common words like "the", "is", "and").

from nltk.tokenize import word_tokenize, sent_tokenize
# USED IN: preprocess_for_nlp (splitting text into words), analyze_bias_context (splitting text into sentences).

from nltk.stem import WordNetLemmatizer
# USED IN: preprocess_for_nlp (converting "running" -> "run").


# --- SENTIMENT ANALYSIS ---
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # A rule based sentiment scanner optimized for social media and short text.
# USED IN: detect_sentiment_vader, analyze_bias_context (to detect emotion in biased sentences).


# --- TOPIC MODELING (MACHINE LEARNING) ---
import gensim # A specialized library for unsupervised semantic analysis.

from gensim import corpora
# USED IN: perform_topic_modeling (to build the Dictionary mapping words to IDs).

from gensim.models import LdaModel, CoherenceModel
# USED IN: perform_topic_modeling (LdaModel does the math, CoherenceModel checks the quality).

from gensim.utils import simple_preprocess
# USED IN: (Alternative) Can be used instead of 'preprocess_for_nlp' for simpler cleaning.


# --- VISUALIZATION ---
import matplotlib.pyplot as plt # The foundation for all Python plotting.
# USED IN: create_sentiment_distribution_plot, create_topic_visualization.

import seaborn as sns # A wrapper around Matplotlib that makes charts look prettier/more statistical.
# USED IN: create_bias_pattern_visualization (specifically for the Heatmap).

# Set up your workspace
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.WARNING)

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ======================= SECTION 2: CONFIGURATION =======================

# Set up your investigation workspace
main_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep
data_dir = os.path.join(main_dir, '') + os.sep #add your data folder name here if you have one (e.g., 'data')
output_dir = os.path.join(main_dir, 'output') + os.sep

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Configure pandas for better data display
pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

# ======================= SECTION 3: CORE FUNCTIONS =======================
# Data preprocessing and cleaning
# - Handle multiple file formats (PDF, DOCX, text)
# - Language detection and translation
# - Text cleaning and tokenization
# - Metadata extraction (source type, publication date, etc.)

#function to extract text from various file formats
def extract_text_from_file(file_path):
    """
    Extract text from various file formats (PDF, DOCX, TXT).
    """
    
    text = ""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # Handle Text Files 
        if file_ext == '.txt':
            # 'utf-8' is the standard web encoding.
            # errors='ignore' if the file has a weird binary character 
            # or a different encoding, it will skip that character rather than crashing the script.
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        
        # Handle PDF Files 
        elif file_ext == '.pdf':
            try:
                # Lazy Loading: import pdfplumber ONLY if we actually have a PDF.
                # This makes the script start faster if you are just processing .txt files,
                # as pdfplumber is a heavy library to load.
                import pdfplumber
                
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        # extract_text() is a method specific to pdfplumber that preserves layout well.
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                            
            except Exception as e:
                # Specific error catching for PDF issues (e.g., encrypted or corrupted PDFs)
                print(f"  ⚠️ PDF extraction error for {file_path}: {e}")
        
        # Handle Word Files (.docx) 
        elif file_ext == '.docx':
            try:
                # Lazy Loading: Import python-docx only when needed.
                import docx 
                
                doc = docx.Document(file_path)
                # Word docs are made of paragraphs. used a list comprehension to 
                # grab text from every paragraph and join them with newlines.
                text = "\n".join([para.text for para in doc.paragraphs])
                
            except Exception as e:
                print(f"  ⚠️ DOCX extraction error for {file_path}: {e}")
        
        # Handle Legacy Word Files (.doc) 
        elif file_ext in ['.doc']:
            # .doc is a binary format distinct from the XML based .docx.
            # The 'docx' library cannot read it, so we explicitly warn the user.
            print(f"  ⚠️ .doc format not directly supported: {file_path}")
            
    except Exception as e:
        # A broad catch all to ensure the program doesn't crash if the file path 
        # is invalid, the file is locked by another user, or permissions are denied.
        print(f"  ❌ Error reading {file_path}: {e}")
    
    # .strip() removes any leading/trailing whitespace (like extra newlines at the end of a file),
    # returning a clean string.
    return text.strip()


# Language Detection Function

def detect_language(text):
    """
    Detect the language of text using the 'langdetect' library.
    Defaults to English ('en') if detection fails or text is too short.
    """
    try:
        # 1. Lazy Import
        # import inside the function so the script doesn't crash at startup 
        from langdetect import detect
        
        # 2. Minimum Length Threshold
        # Language detection is statistical. If text is too short (e.g., "Hi" or "123"), 
        # the result is unreliable. Require at least 50 characters to attempt detection.
        if len(text) > 50:

            # 3. Performance Optimization
            # only analyze the first 5000 characters. 
            # Language profiles are repetitive; if the first 5000 chars are English, 
            # the rest of the document is almost certainly English too. 
            return detect(text[:5000])
            
        # If text is < 50 chars, we skip detection to avoid bad guesses.
        return 'en'
        
    except Exception:
        # 4. The "Safe Fallback"
        # If langdetect is missing, or if it crashes (which it can do on messy text),
        # we default to 'en' (English) rather than crashing the whole program.
        return 'en'
    
# Translation Function
def translate_text(text, source_lang='auto', target_lang='en'):
    """
    Translate text to English using Google Translate via deep_translator.
    Includes logic to handle API limits and language code mismatches.
    """
    
    # 1. Efficiency Check (The "Early Exit")
    # If the text is already in the target language (English), 
    # Return immediately to avoid unnecessary API calls and latency.
    if source_lang == 'en' or source_lang == target_lang:
        return text
    
    try:
        # Lazy Import: Only load the heavy translation library if we actually need to translate.
        from deep_translator import GoogleTranslator
        
        # 2. Standardization (Language Code Mapping)
        lang_code_map = {
            'zh-cn': 'zh-CN',   # Simplified Chinese
            'zh-tw': 'zh-TW',   # Traditional Chinese
            'zh': 'zh-CN',      # Default generic Chinese to Simplified
            'iw': 'he',         # Hebrew (Legacy vs Modern)
            'jw': 'jv',         # Javanese
            'fil': 'tl',        # Filipino vs Tagalog
        }
        
        # .get() looks up the code. If it's not in the map, it defaults to the original source_lang.
        source_lang_mapped = lang_code_map.get(source_lang.lower(), source_lang)
        
        # 3. Handling API Constraints (Chunking)
        # Google Translate has a hard limit (usually 5000 chars) per request.
        # set the limit to 4500 to provide a "safety buffer" for URL encoding overhead.
        max_chars = 4500
        
        # Scenario A: Short Text
        if len(text) <= max_chars:
            return GoogleTranslator(source=source_lang_mapped, target=target_lang).translate(text)
        
        # Scenario B: Long Text (The "Chunking" Strategy)
        # If the text is long, slice it into pieces of 4500 characters.
        chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
        
        translated_chunks = []
        for chunk in chunks:
            translated = GoogleTranslator(source=source_lang_mapped, target=target_lang).translate(chunk)
            translated_chunks.append(translated)
            
            # 4. Rate Limiting
            # pause for 0.5 seconds between requests. 
            # If you send requests too fast, Google's server will block your IP address (HTTP 429).
            time.sleep(0.5) 
            
        # Reassemble the pieces into one string.
        return " ".join(translated_chunks)

    except Exception as e:
        # 5. Fail Safe
        # If the internet is down or the API changes, print the error but return the original text so the user doesn't lose data.
        print(f"  ⚠️ Translation error: {e}")
        return text

# Text Cleaning Function
def clean_text(text):
    """
    Clean and preprocess text for analysis using Regular Expressions (regex).
    """
    # 1. Safety Check
    # If the text is None or empty, return an empty string immediately.
    # This prevents the code from crashing if it receives a null value.
    if not text:
        return ""
    
    # 2. Normalization (Lowercasing)
    # converting to lowercase ensures that the same word during analysis.
    text = text.lower()
    
    # 3. Noise Reduction (URLs)
    # Regex: Match http/https/www followed by non whitespace characters (\S+).
    # Replaced these with an empty string ('') to remove them entirely.
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # 4. Privacy & Noise Reduction (Emails)
    # Regex: Match non whitespace (\S+), then @, then non whitespace (\S+).
    # This is a simple pattern to catch email-like strings.
    text = re.sub(r'\S+@\S+', '', text)
    
    # 5. Selective Character Removal
    # Regex breakdown: [^ ... ] means "Replace everything that is NOT..."
    # \w = letters/numbers, \s = spaces, and .,!?- are the punctuation marksare kept.
    # Replace removed chars with a SPACE ' ', not an empty string.
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    
    # 6. Whitespace Cleanup
    # The previous steps likely created double spaces
    # Regex: \s+ matches "one or more spaces" and replaces them with a single space.
    # .strip() removes any remaining spaces at the very start or end of the string.
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# NLP Preprocessing Function
def preprocess_for_nlp(text):
    """
    Preprocess text for NLP analysis: Tokenizes, removes noise, and lemmatizes.
    """
    # 1. Safety Check
    if not text:
        return []
    
    # 2. Tokenization
    # word_tokenize() splits text into a list of words.
    tokens = word_tokenize(text.lower())
    
    # 3. Stopword Setup
    # Convert the list to a 'set' for O(1) lookup speed (much faster than lists).
    stop_words = set(stopwords.words('english'))
    
    # 4. Domain Adaptation
    # Add custom stopwords that are common but add little analytical value.
    custom_stops = {'said', 'also', 'would', 'could', 'may', 'might', 'one', 'two', 'first', 'new'}
    stop_words.update(custom_stops)
    
    # 5. Initialization
    # The Lemmatizer requires a dictionary lookup to find the "root" of a word.
    lemmatizer = WordNetLemmatizer()
    
    processed_tokens = []
    
    # 6. The Filtering Loop
    for token in tokens:
        # Apply four strict filters:
        if (token not in stop_words and        # Is it a meaningful word?
            token not in string.punctuation and # Is it not just a comma/period?
            len(token) > 2 and                  # Is it long enough to matter?
            token.isalpha()):                   # Is it actual letters (no numbers)?
            
            # 7. Lemmatization
            # Convert word to base form 
            processed_tokens.append(lemmatizer.lemmatize(token))
    
    return processed_tokens


# Orchestrator Function
def load_and_process_document(file_path):
    """
    Your document processing detective work.
    Handles PDF, DOCX, and text files with language detection.

    Loads a file and runs it through the full NLP pipeline.
    Returns a comprehensive dictionary containing raw data, processed data, and metadata.
    """
    # 1. Metadata Extraction
    # strip the folder path to get just the filename (e.g., "report.pdf").
    # This is useful for labeling the data later.
    filename = os.path.basename(file_path)
    
    # 2. Ingestion (The Extract Step)
    raw_text = extract_text_from_file(file_path)
    
    # 3. The "Fail Fast" Check
    # If the file was empty or unreadable, stop immediately to save resources.
    if not raw_text:
        return None
    
    # 4. Language Normalization
    # detect the language first.
    language = detect_language(raw_text)
    
    # ONLY translate if the text is not already English.
    if language != 'en':
        print(f"  🌐 Translating from {language} to English...")
        translated_text = translate_text(raw_text, source_lang=language)
    else:
        # If it's already English, skip the API call 
        translated_text = raw_text
    
    # 5. Cleaning
    cleaned_text = clean_text(translated_text)
    
    # 6. NLP Processing
    # Now that we have clean English text, we convert it into tokens/lemmas.
    tokens = preprocess_for_nlp(cleaned_text)
    
    # 7. Metadata Enrichment
    # (Assuming classify_source_type is a helper function that checks if it's a 'Report', 'Memo', etc.)
    source_type = classify_source_type(filename, file_path)
    
    # 8. The Data Object Return
    # Instead of just returning the tokens, we return a "Rich Object" (dictionary).
    # This keeps the original data alongside the processed data, which is vital for
    # debugging (e.g., checking why a specific token was generated).
    return {
        'filename': filename,
        'filepath': file_path,
        'raw_text': raw_text,           # The original truth
        'translated_text': translated_text, # The English version
        'cleaned_text': cleaned_text,   # The humanreadable normalized version
        'tokens': tokens,               # The machine readable version
        'language': language,
        'source_type': source_type,
        'word_count': len(cleaned_text.split()), # Calculated once here to avoid recalculating later
        'char_count': len(cleaned_text)
    }

# Source Type Classification Function
def classify_source_type(filename, filepath):
    """
    Classify the source type based on filename or path keywords.
    Uses a heuristic (rule based) priority system.
    """
    
    # 1. Normalization
    # convert inputs to lowercase immediately. This ensures that documents are treated exactly the same.
    filename_lower = filename.lower()
    filepath_lower = filepath.lower()
    
    # 2. Taxonomy Definitions (The Knowledge Base)
    # These lists act as our "rules." 
    # Design Note: Specific entities (NSA, FBI) are mixed with generic terms (federal, official).
    
    gov_keywords = ['government', 'gov', 'cisa', 'nsa', 'fbi', 'dhs', 'nist', 'agency', 
                    'federal', 'national', 'ministry', 'official']
    
    vendor_keywords = ['vendor', 'microsoft', 'crowdstrike', 'mandiant', 'paloalto',
                       'fireeye', 'kaspersky', 'symantec', 'mcafee', 'sophos', 
                       'fortinet', 'checkpoint', 'cisco', 'product', 'solution']
    
    media_keywords = ['media', 'news', 'bbc', 'cnn', 'reuters', 'times', 'post',
                      'journal', 'article', 'report', 'wired', 'techcrunch', 'zdnet',
                      'arstechnica', 'theregister', 'bleepingcomputer']
    
    research_keywords = ['research', 'academic', 'university', 'study', 'paper',
                         'analysis', 'independent', 'lab', 'institute']
    
    intl_keywords = ['international', 'foreign', 'china', 'russia', 'europe', 
                     'asia', 'german', 'french', 'spanish', 'chinese', 'russian']
    
    # 3. Data Aggregation
    # We combine the filename and the full path into one string for keyword searching.
    # Example: "C:/Documents/Vendors/Microsoft/update_log.txt" 
    # (The file is just 'update_log.txt', but the path confirms it's a Vendor source).
    combined = filename_lower + " " + filepath_lower
    
    # 4. The Priority Waterfall
    # check categories in a specific order using 'if/elif'.
    # Once a match is found, the function returns immediately and stops checking.
    
    # Priority 1: Government (Highest confidence/importance)
    # The 'any()' function checks the list efficiently.
    if any(kw in combined for kw in gov_keywords):
        return 'government'
        
    # Priority 2: Vendor
    elif any(kw in combined for kw in vendor_keywords):
        return 'vendor'
        
    # Priority 3: Media
    elif any(kw in combined for kw in media_keywords):
        return 'media'
        
    # Priority 4: Research
    elif any(kw in combined for kw in research_keywords):
        return 'research'
        
    # Priority 5: International (Lowest priority overlap)
    elif any(kw in combined for kw in intl_keywords):
        return 'international'
        
    # 5. Fallback
    # If no keywords match, explicitly return 'unknown' rather than None,
    # ensuring consistent data types for later analysis.
    else:
        return 'unknown'


# ======================= SECTION 4: SENTIMENT ANALYSIS =======================
# Sentiment analysis using:
# 1. VADER for quick analysis
# 2. Transformer models for deep analysis
# Compare results across source types


# VADER Sentiment Analysis Function
def detect_sentiment_vader(text):
    """
    Quick sentiment analysis using VADER.
    Perfect for news articles and social media content.

    Perform sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    Returns a normalized compound score and a categorical label.
    """
    
    # 1. Safety Check (Edge Case Handling)
    # If the text is empty, return a "neutral" zero score object immediately.
    # This prevents the analyzer from throwing errors on null inputs.
    if not text:
        return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0, 'sentiment': 'neutral'}
    
    # 2. Initialization
    # create an instance of the VADER analyzer.
    analyzer = SentimentIntensityAnalyzer()
    
    # 3. Scoring
    # polarity_scores() analyzes the text and returns a dictionary with four scores:
    # 'neg' (Negative), 'neu' (Neutral), 'pos' (Positive), and 'compound' (Combined).
    scores = analyzer.polarity_scores(text)
    
    # 4. Thresholding (The Decision Logic)
    # Extract the 'compound' score, which is a normalized metric between -1 (Most Negative)and +1 (Most Positive).
    compound = scores['compound']
    
    # Standard VADER thresholds:
    if compound >= 0.05:
        sentiment = 'positive'
    elif compound <= -0.05:
        sentiment = 'negative'
    else:
        # If the score is between -0.05 and 0.05, it is considered statistically neutral.
        sentiment = 'neutral'
    
    # 5. The Return Object
    # return the detailed breakdown plus our calculated label.
    return {
        'compound': compound,
        'pos': scores['pos'],
        'neu': scores['neu'],
        'neg': scores['neg'],
        'sentiment': sentiment
    }

# Module-level cache so the model is loaded only once across all documents.
_sentiment_pipeline = None

def _get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
    return _sentiment_pipeline

# Transformer Based Sentiment Analysis Function
def detect_sentiment_transformer(text, max_length=512):
    """
    Perform sentiment analysis using transformer model.
    """
    try:
        # 1. Model Initialization (loaded once and reused across all calls)
        sentiment_pipeline = _get_sentiment_pipeline()
        
        # 2. Hard Truncation
        # If text is massive, cut it off at 5000 chars to prevent memory overflows
        # and infinite processing times.
        if len(text) > 5000:
            text = text[:5000]
        
        # 3. Intelligent Chunking (The Core Complexity)
        # Transformer models have a hard limit (usually 512 tokens). 
        # Cannot feed a whole document at once. So, we must break it down.
        sentences = sent_tokenize(text)
        chunk_size = 500 # Estimate characters (roughly 200-300 tokens)
        chunks = []
        current_chunk = ""
        
        for sent in sentences:
            # If adding the next sentence keeps us under the limit, add it.
            if len(current_chunk) + len(sent) < chunk_size:
                current_chunk += " " + sent
            else:
                # Otherwise, seal the current chunk and start a new one.
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent

        # Append the final leftover chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 4. Analysis Loop
        results = []
        # Limit to the first 10 chunks to ensure the script finishes quickly.
        for chunk in chunks[:10]:
            if chunk:
                # The pipeline returns a list like [{'label': 'POSITIVE', 'score': 0.98}]
                result = sentiment_pipeline(chunk)[0]
                results.append(result)
        
        # Edge case: No results found
        if not results:
            return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 0}
        
        # 5. Aggregation (Voting Mechanism)
        # The model gives a score for every chunk. We need one score for the file.
        # Separated the scores into Positive and Negative buckets.
        pos_scores = [r['score'] for r in results if r['label'] == 'POSITIVE']
        neg_scores = [r['score'] for r in results if r['label'] == 'NEGATIVE']
        
        #Calculate averages for each sentiment
        avg_pos = np.mean(pos_scores) if pos_scores else 0
        avg_neg = np.mean(neg_scores) if neg_scores else 0
        
        # 6. Final Decision
        # Compare the average strength of positive vs negative chunks.
        if avg_pos > avg_neg:
            # Confidence is the gap between the two averages.
            return {'label': 'POSITIVE', 'score': avg_pos, 'confidence': avg_pos - avg_neg}
        else:
            return {'label': 'NEGATIVE', 'score': avg_neg, 'confidence': avg_neg - avg_pos}
            
    except Exception as e:
        print(f"  ⚠️ Transformer sentiment error: {e}")
        # Fallback to neutral on crash
        return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 0}
    

# Sentence Level Sentiment Analysis Function
def analyze_sentiment_by_sentence(text):
    """
    Break text into sentences and score each one individually.
    Useful for finding 'needles in the haystack' (e.g., one bad finding in a good report).
    """
    # 1. Segmentation
    # USed NLTK's sentence tokenizer to split text into sentences.
    sentences = sent_tokenize(text)
    
    analyzer = SentimentIntensityAnalyzer()
    
    sentence_sentiments = []
    
    # 2. Iterative Analysis
    for sent in sentences:
        # Score the individual sentence
        scores = analyzer.polarity_scores(sent)
        
        # 3. Data Formatting
        # Append a dictionary for every single sentence.
        sentence_sentiments.append({
            # TRUNCATION:
            # If a sentence is > 100 chars, we cut it off and add "..."
            # This keeps the output readable when you print it or save to CSV.
            'sentence': sent[:100] + '...' if len(sent) > 100 else sent,
            
            'compound': scores['compound'],
            
            # INLINE LOGIC:
            # This is a condensed 'if/else' statement.
            # >= 0.05 is Positive, <= -0.05 is Negative, everything else is Neutral.
            'sentiment': 'positive' if scores['compound'] >= 0.05 else 
                         ('negative' if scores['compound'] <= -0.05 else 'neutral')
        })
    
    # Returns a LIST of dictionaries, not just one dictionary.
    return sentence_sentiments


# ======================= SECTION 5: BIAS DETECTION =======================
# Bias detection pipeline
# Remember: keyword frequency alone doesn't prove bias!
# Combine quantitative analysis with qualitative interpretation


# Define bias indicator lexicons
# This dictionary acts as a "Knowledge Base" for the bias detection algorithm.
# It categorizes words that signal subjective intent rather than objective technical reporting.
BIAS_LEXICONS = {
    # CATEGORY 1: COMMERCIAL BIAS
    # Goal: Detect "Vendor Reports" that are actually disguised marketing.
    # Why: Vendors often exaggerate threats ("FUD" - Fear, Uncertainty, Doubt) to sell products.
    'commercial': {
        # Subjective adjectives used to inflate product value
        'promotional': ['solution', 'protect', 'secure', 'leading', 'best-in-class', 
                        'industry-leading', 'comprehensive', 'advanced', 'innovative',
                        'cutting-edge', 'state-of-the-art', 'next-generation', 'proven',
                        'trusted', 'reliable', 'enterprise-grade', 'world-class'],
        
        # explicit Call-To-Action (CTA) verbs usually found in brochures, not research
        'sales': ['contact us', 'free trial', 'demo', 'pricing', 'subscribe',
                  'buy now', 'purchase', 'license', 'upgrade', 'premium'],
        
        # "FUD" terms designed to induce anxiety in executives so they approve budgets
        'fear_appeal': ['critical', 'urgent', 'immediate action', 'devastating',
                        'catastrophic', 'unprecedented', 'massive', 'severe',
                        'dangerous', 'alarming', 'crisis', 'emergency'],
        
        # Self-attribution to establish authority ("WE found this", "OUR tool blocked this")
        'product_mention': ['our product', 'our solution', 'our platform', 
                            'our technology', 'our team', 'we detected', 'we discovered']
    },

    # CATEGORY 2: POLITICAL BIAS
    # Goal: Detect narratives aligned with government agendas or policy pushing.
    'political': {
        # Terms used to assign blame to foreign states (often before hard proof is available)
        'attribution': ['state-sponsored', 'nation-state', 'foreign adversary',
                        'hostile nation', 'enemy state', 'government-backed',
                        'military intelligence', 'foreign government'],
        
        # "In-group" language framing the event as an attack on "us" (the nation)
        'nationalistic': ['national security', 'homeland', 'domestic', 'patriotic',
                          'our nation', 'our country', 'american', 'western'],
        
        # Language focused on finding a culprit rather than fixing the technical issue
        'blame': ['responsible', 'culprit', 'perpetrator', 'behind the attack',
                  'attributed to', 'linked to', 'traced to', 'originated from'],
        
        # Language advocating for legal/regulatory responses
        'policy': ['sanctions', 'regulations', 'legislation', 'policy', 'mandate',
                   'compliance', 'requirement', 'law enforcement']
    },

    # CATEGORY 3: GEOPOLITICAL BIAS
    # Goal: Identify if the report is focused on international conflict/relations.
    'geopolitical': {
        # Specific named entities typically involved in cyber-espionage
        'country_actors': ['china', 'russia', 'iran', 'north korea', 'apt28', 'apt29',
                           'apt41', 'lazarus', 'fancy bear', 'cozy bear', 'equation group'],
        
        # Framing cyber-incidents as acts of war
        'conflict': ['cyber warfare', 'cyber attack', 'cyber espionage', 'cyber weapon',
                     'information warfare', 'hybrid warfare', 'digital battlefield'],
        
        # Grouping nations into blocs (Us vs. Them)
        'alliances': ['nato', 'five eyes', 'european union', 'allies', 'partners',
                      'coalition', 'international cooperation'],
        
        # Diplomatic consequences language
        'tensions': ['tensions', 'escalation', 'retaliation', 'response', 'counter',
                     'defensive', 'offensive', 'deterrence']
    },

    # CATEGORY 4: SENSATIONALISM
    # Goal: Detect "Clickbait" or emotional manipulation.
    'sensationalism': {
        # Exaggerated adjectives that lack scientific precision
        'hyperbole': ['massive', 'huge', 'enormous', 'unprecedented', 'shocking',
                      'stunning', 'explosive', 'bombshell', 'game-changing'],
        
        # Emotional language aiming to trigger a visceral reaction
        'fear': ['terrifying', 'frightening', 'scary', 'nightmare', 'horror',
                 'apocalyptic', 'doomsday', 'devastating'],
        
        # Artificial time pressure
        'urgency': ['breaking', 'alert', 'warning', 'urgent', 'immediate',
                    'critical', 'emergency', 'now'],
        
        # "Hedging" words that allow authors to make claims without evidence
        'speculation': ['could', 'might', 'may', 'possibly', 'potentially',
                        'allegedly', 'reportedly', 'rumored', 'suspected']
    }
}

# Bias Detection Function
def detect_bias_indicators(text, bias_type='commercial'):
    """
    Spot hidden agendas in cybersecurity reporting.
    Remember: context matters, not just keyword frequency!

    Scans text for specific bias keywords and calculates a density score.
    Returns the score, specific hits, and context snippets.
    """
    # 1. Validation & Safety
    # If text is missing, return a "zero" object immediately to prevent errors.
    if not text:
        return {'score': 0, 'indicators': [], 'evidence': []}
    
    # 2. Preprocessing
    # Convert text to lowercase for case insensitive matching.
    text_lower = text.lower()
    
    # Ensure the requested bias type (e.g., 'political') actually exists in our definitions.
    if bias_type not in BIAS_LEXICONS:
        return {'score': 0, 'indicators': [], 'evidence': []}
    
    # Load the specific dictionary for this bias type
    lexicon = BIAS_LEXICONS[bias_type]
    
    found_indicators = []
    evidence = []
    
    # 3. The Scanning Loop
    # Iterate through the categories and their keywords.
    for category, keywords in lexicon.items():
        for keyword in keywords:
            
            # Simple substring counting. 
            count = text_lower.count(keyword.lower())
            
            if count > 0:
                found_indicators.append({
                    'category': category,
                    'keyword': keyword,
                    'count': count
                })
                
                # 4. Context Extraction (Evidence)
                # We want to show the user *where* the bias was found.
                idx = text_lower.find(keyword.lower())
                if idx != -1:
                    # grab 50 characters before and after the keyword.
                    # max(0, ...) ensures we don't go to a negative index.
                    start = max(0, idx - 50)
                    # min(...) ensures we don't go past the end of the text.
                    end = min(len(text), idx + len(keyword) + 50)
                    
                    # We slice the original text (not lower) to preserve capitalization for readability.
                    context = text[start:end]
                    evidence.append(f"...{context}...")
    
    # 5. Scoring Logic (Density Calculation)
    total_words = len(text.split())
    indicator_count = sum(ind['count'] for ind in found_indicators)
    
    # The Formula: (Hits / Total Words) * Multiplier
    # multiply by 100 to make the number more sensitive (since bias words are rare).
    # min(1.0, ...) caps the score at 1.0 (100%) so we don't return a score of 1.5.
    bias_score = min(1.0, (indicator_count / max(total_words, 1)) * 100)
    
    return {
        'score': round(bias_score, 4), # Round to 4 decimals for clean data
        'indicators': found_indicators, # Detailed breakdown for debugging
        'evidence': evidence[:5],       # Limit to 5 examples to keep the UI clean
        'indicator_count': indicator_count,
        'total_words': total_words
    }

# Comprehensive Bias Analysis Function
def comprehensive_bias_analysis(text):
    """
    Wrapper function that runs the bias detector for ALL defined categories 
    and calculates summary statistics (Overall Score, Dominant Type).
    """
    results = {}
    
    # 1. The Dynamic Loop
    # Iterated through keys in the global BIAS_LEXICONS dictionary 
    # (Commercial, Political, Geopolitical, Sensationalism).
    # This design means if you add a new category to the lexicon later, 
    # this function automatically supports it without needing code changes.
    for bias_type in BIAS_LEXICONS.keys():
        results[bias_type] = detect_bias_indicators(text, bias_type)
    
    # 2. Statistical Aggregation
    # I extracted the 'score' from each result object to calculate the mean.
    # This gives us a single "Total Bias" metric for the document.
    overall_score = np.mean([r['score'] for r in results.values()])
    
    # 3. Determining the "Winner" (Dominant Bias)
    # Used the max() function with a lambda (anonymous) function.
    dominant_bias = max(results.keys(), key=lambda k: results[k]['score'])
    
    # 4. The Executive Summary Return
    return {
        'bias_scores': results,                 # The detailed breakdown (evidence, counts)
        'overall_score': round(overall_score, 4), # The average intensity
        'dominant_bias': dominant_bias,         # The primary flavor (e.g., "Commercial")
        'dominant_score': results[dominant_bias]['score'] # How strong that primary flavor is
    }

# Context Aware Bias Analysis Function
def analyze_bias_context(text, bias_type='commercial'):
    """
    Analyze bias with context awareness by combining keyword matching 
    with sentiment analysis at the sentence level.
    """
    # 1. Granularity (Sentence Segmentation)
    # Break the text into sentences. Bias is rarely distributed 5% evenly across a document; it usually spikes in specific sentences.
    sentences = sent_tokenize(text)
    bias_sentences = []
    
    # 2. Preparation (Flattening the Lexicon)
    # Flatten this into one big list of keywords for faster checking, because in this specific function, we only care IF a keyword exists, 
    # not which specific sub category it belongs to.
    lexicon = BIAS_LEXICONS.get(bias_type, {})
    all_keywords = [kw for keywords in lexicon.values() for kw in keywords]
    
    # 3. The Context Loop
    for sent in sentences:
        sent_lower = sent.lower()
        
        # Check if the sentence contains ANY bias keywords from the chosen type
        found_keywords = [kw for kw in all_keywords if kw.lower() in sent_lower]
        
        if found_keywords:
            # 4. Context Enrichment (Sentiment Overlay)
            # We found a bias word in this sentence.
            # Now we ask: Is this sentence Positive or Negative?
            # "Critical success" (Positive) vs. "Critical failure" (Negative).
            vader_scores = detect_sentiment_vader(sent)
            
            # 5. Data Structuring
            # We store the snippet, the specific trigger words, and the emotional score.
            bias_sentences.append({
                'sentence': sent,
                'keywords': found_keywords,
                'sentiment': vader_scores['sentiment'],
                'compound': vader_scores['compound']
            })
    
    return bias_sentences


# ======================= SECTION 6: TOPIC MODELING =======================
# Topic modeling using Gensim LDA
# Identify recurring themes and how they're framed differently


# Topic Modeling Function
def perform_topic_modeling(documents, num_topics=5, passes=15):
    """
    Discover hidden patterns across different sources.
    What themes keep appearing?

    Perform Latent Dirichlet Allocation (LDA) to discover hidden topics in documents.
    """
    # 1. Safety Check (Data Sufficiency)
    # Topic modeling looks for patterns *across* documents. 
    # If you only have 1 document, there are no patterns to find.
    if not documents or len(documents) < 2:
        return None
    
    # 2. Dictionary Creation (The Vocabulary)
    # This assigns a unique integer ID to every unique word in the entire dataset.
    dictionary = corpora.Dictionary(documents)
    
    # 3. Filtering Extremes (Noise Reduction)
    # no_below=2: Ignore words that appear in less than 2 docs (Typo or too rare).
    # no_above=0.9: Ignore words that appear in > 90% of docs (Stopwords like "the", "cyber").
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    
    # 4. Bag of Words (BoW) Conversion
    # We convert the text lists into frequency vectors using the dictionary IDs.
    # ["malware", "malware", "attack"] -> [(101, 2), (102, 1)]
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    
    # Safety Check: Did we filter everything out?
    if len(dictionary) < 10:
        print("  ⚠️ Not enough unique terms for topic modeling")
        return None
    
    # 5. The LDA Model (The Brain)
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,# How many piles to sort words into
        random_state=42, # Seed for reproducibility (so results don't change every run)
        passes=passes,  # How many times to read the documents to learn
        alpha='auto', # 'auto' learns the best document topic density
        eta='auto' # ''auto' learns the best word topic density
    )
    
    # 6. Quality Control (Coherence Score)
    # This calculates how "human readable" the topics are. 
    # High score (0.5+) = Good topics. Low score (<0.3) = Random words mixed together.
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=documents,
        dictionary=dictionary,
        coherence='c_v' # 'c_v' is the standard sliding window metric
    )
    coherence_score = coherence_model.get_coherence()
    
    # 7. Topic Extraction & Formatting
    # Gensim returns raw strings. We parse this into a clean dictionary for easier use later.
    topics = []
    for idx, topic in lda_model.print_topics(-1):
        # Regex to pull out the words inside quotes ""
        words = re.findall(r'"([^"]*)"', topic)
        # Regex to pull out the numeric weights
        weights = re.findall(r'([0-9.]+)\*', topic)
        
        topics.append({
            'topic_id': idx,
            'words': words,
            'weights': [float(w) for w in weights],
            'raw': topic
        })
    
    # 8. Document Classification
    # Get the main topic for every document in the corpus.
    doc_topics = []
    for doc_bow in corpus:
        topic_dist = lda_model.get_document_topics(doc_bow)
        doc_topics.append(topic_dist)
    
    return {
        'model': lda_model,
        'dictionary': dictionary,
        'corpus': corpus,
        'topics': topics,
        'coherence_score': coherence_score,
        'doc_topics': doc_topics
    }


# Dominant Topic Extraction Function
def get_dominant_topic(lda_model, corpus, documents):
    """
    Classify each document by its most prominent topic.
    Converts fuzzy probabilities into a single 'Winning Label'.
    """
    dominant_topics = []
    
    # 1. Iteration with Indexing
    # Used enumerate(corpus) because the corpus is just a list of vectors.
    # Need 'i' to link the result back to the original filename or text later.
    for i, doc_bow in enumerate(corpus):
        
        # 2. Model Inference
        # Get the topic distribution for this specific document.
        topic_dist = lda_model.get_document_topics(doc_bow)
        
        if topic_dist:
            # 3. The "Argmax" Logic (Finding the Winner)
            # We use the max() function to find the highest probability.
            # key=lambda x: x[1] tells Python: Sort based on the percentage (index 1), not the Topic ID.
            dominant = max(topic_dist, key=lambda x: x[1])
            
            dominant_topics.append({
                'doc_index': i,
                'dominant_topic': dominant[0],   # The ID 
                'topic_probability': dominant[1] # The Confidence 
            })
        else:
            # 4. Edge Case Handling
            # If a document was empty or filtered out entirely (0 words), 
            # the model returns an empty list. We flag this as Topic -1.
            dominant_topics.append({
                'doc_index': i,
                'dominant_topic': -1,
                'topic_probability': 0
            })
    
    return dominant_topics

# Optimal Topic Number Finder Function
def find_optimal_topics(documents, min_topics=3, max_topics=10):
    """
    Determine the best number of topics (k) by calculating Coherence Scores.
    It runs the model multiple times with different k values and picks the winner.
    """
    # 1. Edge Case Handling
    # LDA requires a decent amount of data to find patterns. 
    # If there are  fewer than 3 documents, we default to 5 topics and skip the math.
    if not documents or len(documents) < 3:
        return 5, [] 
    
    # 2. Data Preparation (Standard Gensim Setup)
    # Recreate the dictionary/corpus here to ensure the model runs on the exact data provided.
    dictionary = corpora.Dictionary(documents)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    
    coherence_scores = []
    
    # 3. The "Grid Search" Loop
    # Iterate through every possible number of topics from min to max.
    for num_topics in range(min_topics, max_topics + 1):
        
        # Build a temporary model for this specific number of topics
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42, # Keep seed fixed so comparisons are fair
            passes=10        # Enough passes to get a decent result, but limited for speed
        )
        
        # Calculate the Coherence Score (The "Quality" Metric)
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=documents,
            dictionary=dictionary,
            coherence='c_v' # The standard metric for human readability
        )
        
        score = coherence_model.get_coherence()
        
        # Store the result
        coherence_scores.append({
            'num_topics': num_topics,
            'coherence': score
        })
        
        print(f"  ... Tested {num_topics} topics. Coherence: {score:.4f}")
    
    # 4. Selection Logic
    # look at the list of results and find the one with the highest score.
    optimal = max(coherence_scores, key=lambda x: x['coherence'])
    
    # Return the winning number and the full history 
    return optimal['num_topics'], coherence_scores


# ======================= SECTION 7: DATA LOADING =======================
# Load your provided dataset and additional articles
# Process documents from different sources:
# - Government agencies
# - Security vendors  
# - Mainstream media
# - Independent researchers
# - International sources (including foreign language content)

# Directory Based Document Loader Function
def load_documents_from_directory(directory_path):
    """
    Recursively load and process all supported documents from a directory.
    Aggregates the results into a single list for analysis.
    """
    documents = []
    # 1. The Allow list
    # Explicitly define what we can read. 
    supported_extensions = ['.txt', '.pdf', '.docx']
    
    # 2. Path Validation
    # If the user provides a typo in the path, we catch it here to avoid a crash.
    if not os.path.exists(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        return documents
    
    # 3. Recursive Traversal (The Crawler)
    # os.walk() generates the file names in a directory tree by walking the tree 
    # either top down or bottom up. It automatically goes into sub folders.
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            
            # 4. Extension Filtering
            # Get the extension and lowercase it immediately so '.PDF' matches '.pdf'.
            ext = os.path.splitext(filename)[1].lower()
            
            if ext in supported_extensions:
                # Construct the absolute path so the file opener can find it
                file_path = os.path.join(root, filename)
                print(f"📄 Processing: {filename}")
                
                # 5. The Delegation Step
                # Call the 'Orchestrator' function we wrote earlier. 
                # This handles extraction, translation, cleaning, and detection.
                doc_data = load_and_process_document(file_path)
                
                # 6. Quality Gate
                # Only add the document if:
                # The processor didn't crash (doc_data is not None) OR The processor actually found text (cleaned_text is not empty)
                if doc_data and doc_data['cleaned_text']:
                    documents.append(doc_data)
                    print(f"  ✅ Loaded ({doc_data['word_count']} words, {doc_data['source_type']})")
                else:
                    # If it was a scanned PDF with no OCR text, we warn the user.
                    print(f"  ⚠️ No text extracted")
    
    return documents

# Zip Archive Document Loader Function
def load_documents_from_zip(zip_path, extract_to='./extracted_files/'):
    """
    Unzips an archive to a temporary folder, then loads documents using existing logic.
    """
    documents = []
    
    # 1. Validation
    # Ensure the zip file actually exists before trying to open it.
    if not os.path.exists(zip_path):
        print(f"❌ Zip file not found: {zip_path}")
        return documents
    
    # 2. Extraction (The "Unpack" Phase)
    print(f"📦 Extracting {zip_path}...")
    
    # The 'with' statement is a Context Manager. 
    # It ensures the zip file is properly closed after we are done, even if an error occurs.
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # extractall() dumps every file inside the zip to the folder specified by 'extract_to'.
        zip_ref.extractall(extract_to)
    
    # 3. Delegation (The "DRY" Principle)
    # Instead of rewriting the logic to walk through folders and check extensions,
    # Simply call the function we already wrote: load_documents_from_directory. We point it at the folder where we just dumped the files.
    documents = load_documents_from_directory(extract_to)
    
    return documents


# ======================= SECTION 8: VISUALIZATION =======================

# Visualization 1: Overall sentiment distribution across source types
# Show how government, vendor, media, and research sources differ
def create_sentiment_distribution_plot(df, output_path=None):
    """
    Generates a dual plot visualization:
    1. Box Plot: Shows the range and spread of sentiment scores.
    2. Stacked Bar Chart: Shows the volume and proportion of sentiment categories.
    """
    # 1. Canvas Setup
    # Create a figure with 1 row and 2 columns. 
    # figsize=(14, 6) ensures it is wide enough to see both plots clearly.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PLOT 1: The Box Plot (Left Side) 
    ax1 = axes[0]
    
    # Get unique source types (e.g., 'Media', 'Government', 'Vendor')
    source_types = df['source_type'].unique()
    
    # Generate distinct colors for each source type using the 'Set2' colormap
    colors = plt.cm.Set2(np.linspace(0, 1, len(source_types)))
    
    # Prepare data: Create a list of arrays, one for each source type
    data_by_source = [df[df['source_type'] == st]['vader_compound'].values for st in source_types]
    
    # patch_artist=True allows us to fill the boxes with color (otherwise they are just outlines)
    bp = ax1.boxplot(data_by_source, labels=source_types, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7) # Transparency so grid lines (if any) show through
    
    # Add a reference line at 0 (Neutral)
    # This makes it easy to see if a source leans positive (above line) or negative (below).
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Labels and Titles
    ax1.set_xlabel('Source Type', fontsize=12)
    ax1.set_ylabel('Sentiment Score (VADER Compound)', fontsize=12)
    ax1.set_title('Sentiment Distribution by Source Type', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45) # Rotate labels to prevent overlapping
    
    # PLOT 2: The Stacked Bar Chart (Right Side) 
    ax2 = axes[1]
    
    # Data Transformation:
    # 1. Group by Source and Sentiment
    # 2. .size() counts the documents
    # 3. .unstack() pivots the table so Sentiments become columns
    sentiment_counts = df.groupby(['source_type', 'vader_sentiment']).size().unstack(fill_value=0)
    
    # Safety Step: Ensure all 3 columns exist even if the data is missing one
    # (e.g., if there are no negative documents, the 'negative' column won't be created automatically)
    for cat in ['positive', 'neutral', 'negative']:
        if cat not in sentiment_counts.columns:
            sentiment_counts[cat] = 0
    
    # Reorder columns to ensure consistent coloring (Green -> Gray -> Red)
    sentiment_counts = sentiment_counts[['positive', 'neutral', 'negative']]
    
    # Plotting
    # stacked=True allows us to see the Total Volume + The Breakdown in one view.
    sentiment_counts.plot(kind='bar', stacked=True, ax=ax2, 
                          color=['#2ecc71', '#95a5a6', '#e74c3c'], alpha=0.8)
    
    ax2.set_xlabel('Source Type', fontsize=12)
    ax2.set_ylabel('Number of Documents', fontsize=12)
    ax2.set_title('Sentiment Categories by Source Type', fontsize=14, fontweight='bold')
    ax2.legend(title='Sentiment', loc='upper right')
    ax2.tick_params(axis='x', rotation=45)
    
    # Adjust layout to prevent labels from getting cut off
    plt.tight_layout()
    
    # Save to disk if requested
    if output_path:
        # dpi=300 ensures high resolution for reports/printing
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {output_path}")
    
    plt.show()
    return fig

# Visualization 2: Bias vs sentiment comparison
# Focus on one specific bias type and its relationship to sentiment
def create_bias_sentiment_comparison(df, bias_type='commercial', output_path=None):
    """
    Generates a dual plot visualization to correlate Bias intensity with Sentiment polarity.
    1. Scatter Plot: Individual document distribution.
    2. Grouped Bar Chart: Averages by source type.
    """
    # 1. Setup
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Construct the column name based on user input (e.g., 'commercial_bias_score')
    bias_col = f'{bias_type}_bias_score'
    
    # Safety Check: Ensure the analysis was actually run for this bias type
    if bias_col not in df.columns:
        print(f"⚠️ Bias column {bias_col} not found")
        return None
    
    # PLOT 1: Scatter Plot (Correlation) 
    ax1 = axes[0]
    
    source_types = df['source_type'].unique()
    
    # Defined Color Map: Manually assign colors to ensure consistency across all reports.
    # 'Vendor' is Red (often high bias), 'Government' is Blue, etc.
    colors = {'government': '#3498db', 'vendor': '#e74c3c', 'media': '#f39c12', 
              'research': '#2ecc71', 'international': '#9b59b6', 'unknown': '#95a5a6'}
    
    # Loop through sources to plot them layer by layer. 
    # This allows us to assign the specific colors and create a clear legend.
    for st in source_types:
        subset = df[df['source_type'] == st]
        ax1.scatter(subset[bias_col], subset['vader_compound'], 
                   label=st, color=colors.get(st, '#95a5a6'), 
                   alpha=0.7, s=100, edgecolors='white', linewidth=1)
    
    # Reference Line: Distinguishes Positive vs Negative sentiment
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral sentiment')
    
    ax1.set_xlabel(f'{bias_type.capitalize()} Bias Score', fontsize=12)
    ax1.set_ylabel('Sentiment Score (VADER)', fontsize=12)
    ax1.set_title(f'{bias_type.capitalize()} Bias vs Sentiment by Source', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # PLOT 2: Grouped Bar Chart (Averages)
    ax2 = axes[1]
    
    # Aggregation: Calculate the mean Bias and mean Sentiment for each source category
    avg_data = df.groupby('source_type').agg({
        bias_col: 'mean',
        'vader_compound': 'mean'
    }).reset_index()
    
    # Geometry setup for side by side bars
    x = np.arange(len(avg_data))
    width = 0.35 # Width of the bars
    
    # Draw two sets of bars: Red for Bias, Blue for Sentiment
    # Shifted them by +/- width/2 so they sit next to each other, not on top of each other.
    bars1 = ax2.bar(x - width/2, avg_data[bias_col], width, label=f'{bias_type.capitalize()} Bias',
                   color='#e74c3c', alpha=0.8)
    bars2 = ax2.bar(x + width/2, avg_data['vader_compound'], width, label='Sentiment',
                   color='#3498db', alpha=0.8)
    
    ax2.set_xlabel('Source Type', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title(f'Average {bias_type.capitalize()} Bias vs Sentiment', 
                 fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    # Rotate labels so they don't overlap
    ax2.set_xticklabels(avg_data['source_type'], rotation=45, ha='right')
    ax2.legend(loc='best')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {output_path}")
    
    plt.show()
    return fig

# Visualization 3: Bias pattern visualization
# Creative representation of your bias detection findings
def create_bias_pattern_visualization(df, output_path=None):
    """
    Generates a 4-panel dashboard analyzing bias patterns:
    1. Heatmap (Intensity overview)
    2. Grouped Bar (Category comparison)
    3. Dominant Bias Counts (Frequency)
    4. Score Distribution (Histogram)
    """
    # 1. Canvas Setup
    # Create a 2x2 grid. We make the figure tall (height=12) to fit everything.
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    bias_types = ['commercial', 'political', 'geopolitical', 'sensationalism']
    
    # 2. Data Preparation for Heatmap & Bar Chart
    # Aggregate the raw data (one row per doc) into averages (one row per source).
    heatmap_data = []
    for st in df['source_type'].unique():
        row = {'source_type': st}
        for bt in bias_types:
            col = f'{bt}_bias_score'
            # Safety check: ensure column exists
            if col in df.columns:
                # Calculate mean score for this specific source/bias combo
                row[bt] = df[df['source_type'] == st][col].mean()
            else:
                row[bt] = 0
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df = heatmap_df.set_index('source_type')
    
    # PLOT 1: The Heatmap (Top Left)
    ax1 = axes[0, 0]
    # sns.heatmap creates the heatmap visualization.
    # annot=True puts the actual numbers in the boxes.
    # cmap='YlOrRd' (Yellow-Orange-Red) intuitively shows intensity (Red = High Bias).
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                ax=ax1, cbar_kws={'label': 'Bias Score'})
    ax1.set_title('Bias Patterns Across Source Types', fontsize=14, fontweight='bold')
    
    # PLOT 2: The Grouped Bar Chart (Top Right) 
    ax2 = axes[0, 1]
    
    # Manually handle bar positioning to group them by source.
    for i, st in enumerate(heatmap_df.index):
        values = heatmap_df.loc[st].values
        x = np.arange(len(bias_types))
        # Shift the bar position by i*0.15 so they stand next to each other
        ax2.bar(x + i*0.15, values, width=0.15, label=st, alpha=0.8)
    
    ax2.set_xticks(np.arange(len(bias_types)) + 0.3)
    ax2.set_xticklabels([bt.capitalize() for bt in bias_types], rotation=45, ha='right')
    ax2.set_title('Bias Type Comparison by Source', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # PLOT 3: Dominant Bias Counts (Bottom Left) 
    ax3 = axes[1, 0]
    
    if 'dominant_bias' in df.columns:
        # Pivot data to count how many docs of each source have which dominant bias
        dominant_counts = df.groupby(['source_type', 'dominant_bias']).size().unstack(fill_value=0)
        dominant_counts.plot(kind='bar', ax=ax3, colormap='Set2', alpha=0.8)
        ax3.set_title('Dominant Bias by Source Type', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'Data Unavailable', ha='center')
    
    # PLOT 4: Overall Score Histogram (Bottom Right)
    ax4 = axes[1, 1]
    
    if 'overall_bias_score' in df.columns:
        # Loop through sources to overlay their histograms
        for st in df['source_type'].unique():
            subset = df[df['source_type'] == st]['overall_bias_score']
            # alpha=0.5 makes bars semi transparent so overlaps are visible
            ax4.hist(subset, bins=15, alpha=0.5, label=st, edgecolor='white')
        
        ax4.set_title('Distribution of Overall Bias Scores', fontsize=14, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'Data Unavailable', ha='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {output_path}")
    
    plt.show()
    return fig

# visualization 4: Topic Modeling Results
# Visual representation of discovered topics and their distribution
def create_topic_visualization(topic_results, df, output_path=None):
    """
    Visualizes Topic Modeling results:
    1. Horizontal Bar Chart: Key words defining each topic.
    2. Grouped Bar Chart: Which sources talk about which topics.
    """
    # 1. Safety Check
    # If topic modeling failed (e.g., dataset too small), skip this step.
    if not topic_results:
        print("⚠️ No topic modeling results to visualize")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PLOT 1: Top Words per Topic (Horizontal Bars) 
    ax1 = axes[0]
    
    topics = topic_results['topics']
    num_topics = len(topics)
    
    # Create base Y positions for the topics (0, 1, 2, 3...)
    y_positions = np.arange(num_topics)
    
    # 2. The Word Loop
    # List the top 5 words for each topic next to each other.
    for i, topic in enumerate(topics):
        words = topic['words'][:5]      # Grab top 5 words
        weights = topic['weights'][:5]  # Grab their statistical importance
        
        # 3. Manual Bar Positioning
        # Offset each bar slightly (j * 0.15) so they don't overlap.
        # This creates a cluster of 5 little bars for every 1 Topic.
        ax1.barh([i + j*0.15 for j in range(len(words))], weights, height=0.12,
                 color=plt.cm.tab10(i % 10)) # Cycle colors per topic
        
        # 4. Direct Annotation
        # Instead of a legend, we write the specific word right next to its bar.
        for j, (word, weight) in enumerate(zip(words, weights)):
            ax1.text(weight + 0.005, i + j*0.15, word, va='center', fontsize=8)
    
    # Set the main Y-axis labels to just be "Topic 1", "Topic 2", etc.
    ax1.set_yticks(y_positions + 0.3)
    ax1.set_yticklabels([f'Topic {i}' for i in range(num_topics)])
    ax1.set_xlabel('Weight (Importance)', fontsize=12)
    ax1.set_title('Top Words per Topic', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # PLOT 2: Topic Distribution (Who talks about what?) 
    ax2 = axes[1]
    
    if 'dominant_topic' in df.columns:
        # 5. Data Pivoting
        # Group by Source Type (Rows) and Topic (Cols).
        # .unstack() turns the long list into a wide matrix suitable for plotting.
        topic_source = df.groupby(['source_type', 'dominant_topic']).size().unstack(fill_value=0)
        
        # 'Set3' is a high contrast palette good for distinguishing categorical topics.
        topic_source.plot(kind='bar', ax=ax2, colormap='Set3', alpha=0.8)
        
        ax2.set_xlabel('Source Type', fontsize=12)
        ax2.set_ylabel('Document Count', fontsize=12)
        ax2.set_title('Topic Distribution by Source Type', fontsize=14, fontweight='bold')
        ax2.legend(title='Topic', loc='best', fontsize=9)
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {output_path}")
    
    plt.show()
    return fig


# ======================= SECTION 9: MAIN ANALYSIS PIPELINE =======================

# Master Analysis Pipeline Function
def run_analysis_pipeline(documents, use_transformer=False):
    """
    The Master Loop: Iterates through loaded documents and applies
    Sentiment, Bias, and Data Aggregation logic.
    Returns a Pandas DataFrame suitable for plotting and export.
    """
    print("\n" + "="*60)
    print("🔍 STARTING ANALYSIS PIPELINE")
    print("="*60)
    
    results = []
    
    # 1. The Main Processing Loop
    # Iterate through every document loaded from the folder/zip.
    for i, doc in enumerate(documents):
        print(f"\n📄 Analyzing [{i+1}/{len(documents)}]: {doc['filename']}")
        
        # 2. Base Metadata Population
        # Start the result dictionary with facts we already know (Filename, Type, etc.)
        result = {
            'filename': doc['filename'],
            'source_type': doc['source_type'],
            'language': doc['language'],
            'word_count': doc['word_count']
        }
        
        # 3. Sentiment Analysis (VADER)
        # VADER because it is fast and reliable for general tone.
        print("  📊 Running VADER sentiment analysis...")
        vader_result = detect_sentiment_vader(doc['cleaned_text'])
        
        # Flatten the VADER dictionary into columns for the DataFrame
        result['vader_compound'] = vader_result['compound']
        result['vader_pos'] = vader_result['pos']
        result['vader_neu'] = vader_result['neu']
        result['vader_neg'] = vader_result['neg']
        result['vader_sentiment'] = vader_result['sentiment']
        
        # 4. Sentiment Analysis (Transformer - Optional)
        if use_transformer:
            print("  🤖 Running transformer sentiment analysis...")
            transformer_result = detect_sentiment_transformer(doc['cleaned_text'])
            result['transformer_label'] = transformer_result['label']
            result['transformer_score'] = transformer_result['score']
        
        # 5. Bias Analysis
        # Runs the comprehensive scanner across all 4 bias types (Commercial, Political, etc.)
        print("  🔎 Detecting bias indicators...")
        bias_result = comprehensive_bias_analysis(doc['cleaned_text'])
        
        # Store high level summary stats
        result['overall_bias_score'] = bias_result['overall_score']
        result['dominant_bias'] = bias_result['dominant_bias']
        
        # Flatten specific bias scores (e.g., 'political_bias_score') so they can be plotted later
        for bias_type, scores in bias_result['bias_scores'].items():
            result[f'{bias_type}_bias_score'] = scores['score']
            result[f'{bias_type}_indicator_count'] = scores['indicator_count']
        
        # Append this document's finished row to the master list
        results.append(result)
        print(f"  ✅ Complete - Sentiment: {result['vader_sentiment']}, "
              f"Dominant Bias: {result['dominant_bias']}")
    
    # 6. Data Structure Conversion
    # Convert list of dicts -> Pandas DataFrame. 
    df = pd.DataFrame(results)
    
    # 7. Token Management
    # Separate the tokens (lists of strings) from the main DataFrame creation logic.
    tokens_list = [doc['tokens'] for doc in documents]
    df['tokens'] = tokens_list
    
    print("\n" + "="*60)
    print("✅ ANALYSIS PIPELINE COMPLETE")

    print("="*60)
    
    return df

# Topic Modeling Pipeline Function
def run_topic_modeling_analysis(df, num_topics=5):
    """
    Orchestrates the topic modeling process:
    1. Prepares data from the DataFrame.
    2. Runs the LDA algorithm.
    3. Prints a summary to the console.
    4. Enriches the DataFrame with the results.
    """
    print("\n" + "="*60)
    print("📚 RUNNING TOPIC MODELING")
    print("="*60)
    
    # 1. Data Conversion
    # Gensim requires a list of lists, Pandas stores this as a Series. .tolist() converts it to the format Gensim needs.
    documents = df['tokens'].tolist()
    
    # 2. The Core Calculation
    # Call the function we defined earlier to do the heavy mathematical lifting.
    topic_results = perform_topic_modeling(documents, num_topics=num_topics)
    
    if topic_results:
        # 3. Immediate Feedback (Console Output)
        # Print the Coherence Score immediately so the user knows if the model is "good".
        print(f"\n📊 Coherence Score: {topic_results['coherence_score']:.4f}")
        print("\n📝 Discovered Topics:")
        
        # Print the definitions of the topics found.
        for topic in topic_results['topics']:
            print(f"\n  Topic {topic['topic_id']}:")
            # show the top 7 words so the user can mentally label the topic
            top_words = ", ".join(topic['words'][:7])
            print(f"    Keywords: {top_words}")
        
        # 4. Document Classification
        # The model knows what the topics are. Now we ask: 
        # "Which topic does each specific document belong to?"
        dominant_topics = get_dominant_topic(
            topic_results['model'], 
            topic_results['corpus'], 
            documents
        )
        
        # 5. DataFrame Enrichment
        # Take the list of results and plug them back into the main DataFrame as new columns.
        # This aligns the abstract math (Topic 0) with the concrete file (Filename).
        df['dominant_topic'] = [dt['dominant_topic'] for dt in dominant_topics]
        df['topic_probability'] = [dt['topic_probability'] for dt in dominant_topics]
    
    # Return both the complex model object (for plotting) and the simple table (for Excel).
    return topic_results, df


# ======================= SECTION 10: MAIN EXECUTION =======================

# main() Function
def main():
    """
    The Orchestrator Function.
    1. Loads Data -> 2. Runs Analysis -> 3. Prints Stats -> 4. Generates Viz -> 5. Saves CSV
    """
    print("\n" + "="*70)
    print("🕵️  CYBERSECURITY SENTIMENT & BIAS ANALYSIS")
    print("="*70)
    
    # PHASE 1: INGESTION 
    documents = []
    
    # Priority 1: Check for a Zip File 
    zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Project 1 Files.zip')
    if os.path.exists(zip_path):
        print(f"\n📦 Found zip file: {zip_path}")
        documents = load_documents_from_zip(zip_path)
    
    # Priority 2: If no Zip, check the folder
    # We only run this if 'documents' is still empty.
    if not documents and os.path.exists(data_dir):
        print(f"\n📁 Loading from directory: {data_dir}")
        documents = load_documents_from_directory(data_dir)
    
    # If both fail, stop the program gracefully.
    if not documents:
        print("\n❌ No data files found. Please ensure your zip file or data directory exists.")
        print(f"   Looking for zip file: {zip_path}")
        print(f"   Looking for directory: {data_dir}")
        return None, None
    
    print(f"\n📊 Loaded {len(documents)} documents")
    
    # Print a receipt of what was loaded so the user can verify.
    print("\n" + "-"*50)
    print("DOCUMENT SUMMARY")
    print("-"*50)
    for doc in documents:
        print(f"  • {doc['filename']}: {doc['source_type']} ({doc['language']}, {doc['word_count']} words)")
    
    # PHASE 2: PROCESSING
    # Call the Master Analysis Pipeline (Sentiment + Bias)
    df = run_analysis_pipeline(documents, use_transformer=False)
    
    # Call the Topic Modeling Pipeline (LDA)
    topic_results, df = run_topic_modeling_analysis(df, num_topics=5)
    
    # PHASE 3: REPORTING (Console)
    print("\n" + "="*70)
    print("📈 ANALYSIS RESULTS SUMMARY")
    print("="*70)
    
    # Print quick stats so the user doesn't HAVE to open Excel to get answers.
    print("\n📊 Sentiment by Source Type:")
    # Group by Source, calculate mean Compound score, and find the most common Sentiment label.
    print(df.groupby('source_type')[['vader_compound', 'vader_sentiment']].agg({
        'vader_compound': 'mean',
        'vader_sentiment': lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A'
    }).round(4))
    
    print("\n🔎 Bias Scores by Source Type:")
    # Dynamically find all columns ending in '_bias_score' so we don't have to hard-code them.
    bias_cols = [col for col in df.columns if col.endswith('_bias_score')]
    if bias_cols:
        print(df.groupby('source_type')[bias_cols].mean().round(4))
    
    # PHASE 4: VISUALIZATION
    print("\n" + "="*70)
    print("📊 CREATING VISUALIZATIONS")
    print("="*70)
    
    # We pass the 'output_dir' so images are saved automatically.
    print("\n📊 Creating Visualization 1: Sentiment Distribution...")
    create_sentiment_distribution_plot(df, output_dir + 'viz1_sentiment_distribution.png')
    
    print("\n📊 Creating Visualization 2: Bias vs Sentiment...")
    create_bias_sentiment_comparison(df, 'commercial', output_dir + 'viz2_bias_sentiment.png')
    
    print("\n📊 Creating Visualization 3: Bias Patterns...")
    create_bias_pattern_visualization(df, output_dir + 'viz3_bias_patterns.png')
    
    if topic_results:
        print("\n📊 Creating Visualization 4: Topic Distribution...")
        create_topic_visualization(topic_results, df, output_dir + 'viz4_topics.png')
    
    # PHASE 5: EXPORT 
    output_csv = output_dir + 'analysis_results.csv'
    
    # Drop the 'tokens' column. 
    # Tokens are lists ['word', 'word'] which look messy in CSV/Excel and bloat file size.
    df_export = df.drop(columns=['tokens'], errors='ignore')
    
    df_export.to_csv(output_csv, index=False)
    print(f"\n💾 Results saved to: {output_csv}")
    
    print("\n" + "="*70)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*70)
    
    return df, topic_results

# Standard Python Entry Point
if __name__ == "__main__":
    df, topic_results = main()
