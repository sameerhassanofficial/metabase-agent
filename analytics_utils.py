import os
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from typing import Dict, Any, List, Optional
import re
from difflib import SequenceMatcher
from collections import defaultdict

# Domain-specific synonyms for better matching
SYNONYMS = {
    'male': ['men', 'boys', 'males', 'gentlemen'],
    'female': ['women', 'girls', 'females', 'ladies'],
    'child': ['children', 'kids', 'minors', 'young'],
    'patient': ['patients', 'people', 'individuals', 'persons'],
    'disease': ['diseases', 'illness', 'condition', 'symptoms'],
    'lab': ['laboratory', 'test', 'testing', 'diagnostic'],
    'prescription': ['prescriptions', 'medication', 'medicine', 'drugs'],
    'dispense': ['dispensing', 'distribution', 'handout'],
    'total': ['sum', 'count', 'number', 'amount'],
    'location': ['place', 'area', 'site', 'venue'],
    'mobile': ['movable', 'portable', 'traveling'],
    'health': ['medical', 'clinical', 'healthcare'],
    'unit': ['units', 'team', 'group', 'center']
}

def preprocess_text(text: str) -> str:
    """Preprocess text for better matching."""
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def expand_synonyms(text: str) -> str:
    """Expand text with synonyms for better matching."""
    expanded = text
    for word, synonyms in SYNONYMS.items():
        if word in text.lower():
            expanded += " " + " ".join(synonyms)
    return expanded

def fuzzy_match(query: str, target: str, threshold: float = 0.6) -> float:
    """Calculate fuzzy string similarity."""
    return SequenceMatcher(None, query.lower(), target.lower()).ratio()

def build_dynamic_domain_rules(all_cards: List[Dict]) -> Dict[str, List[str]]:
    """
    Dynamically build domain rules by analyzing actual card names and content.
    This makes the system work with any card names, not just hardcoded ones.
    """
    logger = logging.getLogger(__name__)
    
    # Define domain keywords and their related terms
    domain_keywords = {
        'patient': ['patient', 'patients', 'person', 'people', 'individual'],
        'male': ['male', 'men', 'boy', 'gentleman', 'males'],
        'female': ['female', 'women', 'girl', 'lady', 'females'],
        'child': ['child', 'children', 'kid', 'kids', 'minor', 'young'],
        'disease': ['disease', 'diseases', 'illness', 'condition', 'symptom', 'health_condition'],
        'lab': ['lab', 'laboratory', 'test', 'testing', 'diagnostic', 'report'],
        'prescription': ['prescription', 'prescriptions', 'medication', 'medicine', 'drug'],
        'dispense': ['dispense', 'dispensing', 'distribution', 'handout'],
        'location': ['location', 'place', 'area', 'site', 'venue', 'mobile', 'unit'],
        'campus': ['campus', 'conducted', 'session'],
        'consumption': ['consumption', 'medicine', 'drug'],
        'shipment': ['shipment', 'shipments', 'delivery'],
        'move': ['move', 'moves', 'people', 'personnel']
    }
    
    # Build dynamic rules
    dynamic_rules = {}
    
    for domain, keywords in domain_keywords.items():
        matching_cards = []
        
        for card in all_cards:
            card_data = card.get("card", {})
            if not card_data.get("card_id"):
                continue
                
            card_name = card_data.get("name", "").lower()
            card_desc = card_data.get("description", "").lower()
            
            # Check if any domain keyword matches the card name or description
            for keyword in keywords:
                if keyword in card_name or keyword in card_desc:
                    matching_cards.append(card_data.get("name"))
                    break  # Only add once per card
        
        if matching_cards:
            dynamic_rules[domain] = matching_cards
            logger.debug(f"Domain '{domain}' matched cards: {matching_cards}")
    
    logger.info(f"Built dynamic domain rules with {len(dynamic_rules)} domains")
    return dynamic_rules

def find_relevant_cards_enhanced(question: str, all_cards: list, top_n: int = 5) -> list:
    """
    Enhanced card selection using multiple strategies:
    1. Dynamic domain-specific rules
    2. Enhanced TF-IDF
    3. Fuzzy string matching
    4. Synonym expansion
    5. Weighted scoring
    """
    logger = logging.getLogger(__name__)
    
    if not all_cards:
        logger.warning("No cards provided for selection")
        return []
    
    # Preprocess question
    question_lower = preprocess_text(question)
    question_expanded = expand_synonyms(question_lower)
    
    logger.info(f"Original question: {question}")
    logger.info(f"Preprocessed question: {question_lower}")
    logger.info(f"Expanded question: {question_expanded}")
    
    # Strategy 1: Dynamic domain-specific rules
    dynamic_rules = build_dynamic_domain_rules(all_cards)
    rule_matches = []
    
    for keyword, card_list in dynamic_rules.items():
        if keyword in question_lower:
            rule_matches.extend(card_list)
    
    if rule_matches:
        logger.info(f"Dynamic domain rule matches: {rule_matches}")
        # Return up to top_n cards from domain rules
        return rule_matches[:top_n]
    
    # Strategy 2: Enhanced TF-IDF with preprocessing
    card_scores = defaultdict(float)
    card_texts = []
    card_names = []
    
    for c in all_cards:
        card_data = c.get("card", {})
        if c.get("card_id"):
            name = card_data.get("name", "")
            desc = card_data.get("description", "")
            schema = " ".join(col["name"] for col in card_data.get("schema", []))
            
            # Preprocess all text
            name_processed = preprocess_text(name)
            desc_processed = preprocess_text(desc)
            schema_processed = preprocess_text(schema)
            
            # Combine all text
            combined_text = f"{name_processed} {desc_processed} {schema_processed}"
            combined_text_expanded = expand_synonyms(combined_text)
            
            card_texts.append(combined_text_expanded)
            card_names.append(name)
    
    if card_texts:
        try:
            # TF-IDF with better parameters
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                stop_words='english',
                min_df=1,
                max_df=0.95
            )
            
            # Fit on all texts including the question
            all_texts = card_texts + [question_expanded]
            vectorizer.fit(all_texts)
            
            # Transform
            card_vecs = vectorizer.transform(card_texts)
            question_vec = vectorizer.transform([question_expanded])
            
            # Calculate similarities
            sims = cosine_similarity(question_vec, card_vecs)[0]
            
            # Add TF-IDF scores
            for i, score in enumerate(sims):
                card_scores[card_names[i]] += score * 0.4  # 40% weight
            
            logger.info(f"TF-IDF similarities: {list(zip(card_names, sims))}")
            
        except Exception as e:
            logger.error(f"TF-IDF calculation failed: {e}")
    
    # Strategy 3: Fuzzy string matching
    for i, card_name in enumerate(card_names):
        # Fuzzy match against card name
        name_similarity = fuzzy_match(question_lower, card_name)
        card_scores[card_name] += name_similarity * 0.3  # 30% weight
        
        # Fuzzy match against description
        if card_texts[i]:
            desc_similarity = fuzzy_match(question_lower, card_texts[i])
            card_scores[card_name] += desc_similarity * 0.2  # 20% weight
    
    # Strategy 4: Direct keyword matching
    question_words = set(question_lower.split())
    for i, card_name in enumerate(card_names):
        card_words = set(card_texts[i].split())
        keyword_overlap = len(question_words & card_words) / max(len(question_words), 1)
        card_scores[card_name] += keyword_overlap * 0.1  # 10% weight
    
    # Sort by score and return top results
    sorted_cards = sorted(card_scores.items(), key=lambda x: x[1], reverse=True)
    selected_cards = [name for name, score in sorted_cards if score > 0.1][:top_n]
    
    logger.info(f"Final card selection scores: {sorted_cards[:top_n]}")
    logger.info(f"Selected cards: {selected_cards}")
    
    return selected_cards

def get_openai_embedding(text: str, api_key: str) -> List[float]:
    """Get OpenAI embedding for a given text."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": "text-embedding-ada-002"
    }
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers=headers,
        json=data
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def find_relevant_cards_openai_embed(question: str, all_cards: list, top_n: int = 5) -> list:
    """
    Use OpenAI Embeddings to find the most semantically relevant cards for a question.
    Returns a list of card names.
    """
    logger = logging.getLogger(__name__)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set. Skipping OpenAI Embeddings card selection.")
        return []
    card_texts = []
    card_names = []
    for c in all_cards:
        card_data = c.get("card", {})
        if c.get("card_id"):
            text = (card_data.get("name") or "") + " " + (card_data.get("description") or "")
            text += " " + " ".join(col["name"] for col in card_data.get("schema", []))
            card_texts.append(text)
            card_names.append(card_data.get("name"))
    if not card_texts:
        logger.info("No card texts found for OpenAI Embeddings matching.")
        return []
    try:
        question_emb = get_openai_embedding(question, api_key)
        card_embs = [get_openai_embedding(text, api_key) for text in card_texts]
        sims = cosine_similarity([question_emb], card_embs)[0]
        top_idx = np.argsort(sims)[::-1][:top_n]
        selected = [(card_names[i], float(sims[i])) for i in top_idx if sims[i] > 0]
        logger.info(f"OpenAI Embeddings card selection: {selected}")
        return [name for name, score in selected]
    except Exception as e:
        logger.error(f"OpenAI Embeddings card selection failed: {e}")
        return []

def find_relevant_cards_sklearn(question: str, all_cards: list, top_n: int = 8) -> list:
    """
    Use TF-IDF vectorization (with n-grams, stopwords, and lowercasing) and cosine similarity to find the most relevant cards for a question.
    Returns a list of card names.
    """
    logger = logging.getLogger(__name__)
    card_texts = []
    card_names = []
    for c in all_cards:
        card_data = c.get("card", {})
        if c.get("card_id"):
            text = (card_data.get("name") or "") + " " + (card_data.get("description") or "")
            text += " " + " ".join(col["name"] for col in card_data.get("schema", []))
            text = text.lower()  # Lowercase for consistency
            card_texts.append(text)
            card_names.append(card_data.get("name"))
    if not card_texts:
        logger.info("No card texts found for TF-IDF matching.")
        return []
    # Lowercase the question as well
    question = question.lower()
    # Combine custom stopwords with English stopwords
    vectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words=['english'])
    vectorizer.fit(card_texts + [question])
    card_vecs = vectorizer.transform(card_texts)
    question_vec = vectorizer.transform([question])
    sims = cosine_similarity(question_vec, card_vecs)[0]
    top_idx = np.argsort(sims)[::-1][:top_n]
    selected = [(card_names[i], float(sims[i])) for i in top_idx if sims[i] > 0]
    logger.info(f"TF-IDF card selection: {selected}")
    logger.debug(f"Prompt: {question}")
    logger.debug(f"Card texts: {card_texts}")
    logger.debug(f"Similarity scores: {sims}")
    return [name for name, score in selected]

def aggregate_data_by_column(data: List[Dict], group_column: str, value_columns: List[str] = None) -> Dict:
    """
    Aggregate data by a specific column and calculate totals for numeric columns.
    Returns aggregated data suitable for charts and tables.
    """
    if not data:
        return {"data": [], "summary": "No data available"}
    
    # If no value columns specified, use all numeric columns
    if not value_columns:
        sample_row = data[0] if data else {}
        value_columns = [col for col, val in sample_row.items() 
                        if isinstance(val, (int, float)) and col != group_column]
    
    # Group data by the specified column
    grouped_data = {}
    for row in data:
        group_value = row.get(group_column, "Unknown")
        if group_value not in grouped_data:
            grouped_data[group_value] = []
        grouped_data[group_value].append(row)
    
    # Calculate totals for each group
    aggregated_data = []
    for group_value, group_rows in grouped_data.items():
        group_summary = {group_column: group_value}
        
        # Calculate totals for each value column
        for col in value_columns:
            total = sum(row.get(col, 0) for row in group_rows 
                       if isinstance(row.get(col), (int, float)))
            group_summary[f"Total_{col}"] = total
            group_summary[f"Count_{col}"] = len([row for row in group_rows 
                                               if row.get(col) is not None])
        
        aggregated_data.append(group_summary)
    
    # Sort by the group column for better presentation
    aggregated_data.sort(key=lambda x: str(x.get(group_column, "")))
    
    return {
        "data": aggregated_data,
        "summary": f"Aggregated by {group_column}",
        "total_groups": len(aggregated_data),
        "value_columns": value_columns
    }

def clean_and_validate_data(data: List[Dict]) -> List[Dict]:
    """
    Clean and validate data before aggregation.
    Removes None values, converts types, and handles edge cases.
    """
    if not data:
        return []
    
    cleaned_data = []
    for row in data:
        if not isinstance(row, dict):
            continue
        
        cleaned_row = {}
        for key, value in row.items():
            # Handle None, empty strings, and whitespace
            if value is None or (isinstance(value, str) and value.strip() == ""):
                cleaned_row[key] = "Unknown"
            elif isinstance(value, str):
                cleaned_row[key] = value.strip()
            else:
                cleaned_row[key] = value
        
        cleaned_data.append(cleaned_row)
    
    return cleaned_data

def count_by_column(data: List[Dict], count_column: str) -> Dict:
    """
    Count occurrences of each unique value in a specific column.
    Useful for requests like "count by MHU" or "patients by location".
    """
    if not data:
        return {"data": [], "summary": "No data available"}
    
    # Clean the data first
    cleaned_data = clean_and_validate_data(data)
    
    if not cleaned_data:
        return {"data": [], "summary": "No valid data after cleaning"}
    
    # Log the first few rows for debugging
    logger = logging.getLogger(__name__)
    logger.info(f"Counting by column: {count_column}")
    logger.info(f"Sample data (first 3 rows): {cleaned_data[:3]}")
    logger.info(f"Available columns: {list(cleaned_data[0].keys()) if cleaned_data else []}")
    
    # Check if the column exists
    if count_column not in cleaned_data[0]:
        logger.warning(f"Column '{count_column}' not found in data. Available columns: {list(cleaned_data[0].keys())}")
        return {"data": [], "summary": f"Column '{count_column}' not found in data"}
    
    # Count occurrences
    counts = {}
    total_rows = 0
    
    for row in cleaned_data:
        total_rows += 1
        value = row.get(count_column, "Unknown")
        
        # Handle different data types
        if value is None:
            value = "Unknown"
        elif isinstance(value, (int, float)):
            value = str(value)  # Convert numbers to strings for consistent counting
        elif isinstance(value, str):
            value = value.strip()
            if not value:
                value = "Unknown"
        
        counts[value] = counts.get(value, 0) + 1
    
    logger.info(f"Total rows processed: {total_rows}")
    logger.info(f"Unique values found: {len(counts)}")
    logger.info(f"Count breakdown: {counts}")
    
    # Convert to list format for charts/tables
    count_data = [{"Category": key, "Count": value} for key, value in counts.items()]
    
    # Sort by count (descending) for better presentation
    count_data.sort(key=lambda x: x["Count"], reverse=True)
    
    return {
        "data": count_data,
        "summary": f"Count by {count_column}",
        "total_categories": len(count_data),
        "total_count": sum(counts.values()),
        "debug_info": {
            "total_rows_processed": total_rows,
            "column_found": count_column in cleaned_data[0] if cleaned_data else False,
            "available_columns": list(cleaned_data[0].keys()) if cleaned_data else []
        }
    }

def get_numeric_summaries(df: pd.DataFrame) -> Dict[str, Any]:
    """Return mean, median, std, min, max for all numeric columns."""
    summaries = {}
    for col in df.select_dtypes(include='number').columns:
        arr = df[col].dropna().values
        if arr.size > 0:
            summaries[col] = {
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'sum': float(np.sum(arr)),
            }
    return summaries

def predict_trend(df: pd.DataFrame, x_col: str, y_col: str, periods_ahead: int = 1) -> Optional[float]:
    """
    Fit a linear regression to predict y_col from x_col and forecast periods_ahead into the future.
    x_col should be numeric or datetime (will be converted to ordinal).
    Returns the predicted y value for the next period.
    """
    if x_col not in df.columns or y_col not in df.columns:
        return None
    x = df[x_col]
    y = df[y_col]
    if np.issubdtype(x.dtype, np.datetime64):
        x = x.map(pd.Timestamp.toordinal)
    X = np.array(x).reshape(-1, 1)
    y = np.array(y)
    if len(X) < 2:
        return None
    model = LinearRegression()
    model.fit(X, y)
    next_x = X[-1][0] + periods_ahead
    y_pred = model.predict(np.array([[next_x]]))
    return float(y_pred[0]) 