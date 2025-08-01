import os
import json
import logging
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import openai
import openai
from metabase_client import MetabaseConfig, MetabaseClient
from typing import Tuple
from typing import Tuple
# Removed concurrent.futures import - no longer using threading

# Configure logging to suppress OpenAI debug messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
logging.getLogger('openai._base_client').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class MetabaseLLMAgent:
    """AI agent that uses Metabase data to answer questions conversationally."""
    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

        if not self.gemini_api_key and not self.openai_api_key:
            st.error("Neither GEMINI_API_KEY nor OPENAI_API_KEY environment variables found!")
            logger.error("No LLM API key set.")
            st.stop()
        
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.openai_client = None
        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)

        # Initialize Metabase client (will be overridden if shared from app.py)
        metabase_url = os.getenv("METABASE_URL", "").rstrip("/")
        metabase_username = os.getenv("METABASE_USERNAME", "")
        metabase_password = os.getenv("METABASE_PASSWORD", "")

        if not all([metabase_url, metabase_username, metabase_password]):
            st.error("Missing Metabase environment variables: METABASE_URL, METABASE_USERNAME, METABASE_PASSWORD")
            logger.error("Missing Metabase environment variables.")
            st.stop()

        self.metabase_config = MetabaseConfig(
            base_url=metabase_url,
            username=metabase_username,
            password=metabase_password
        )
        # Create default client (will be overridden if shared from app.py)
        self.metabase_client = MetabaseClient(self.metabase_config)
        # Note: Authentication will be handled by the shared client from app.py
        
        # Performance optimization: In-memory cache
        self._card_definition_cache = {}
        self._sample_data_cache = {}
        self._cache_timeout = 300  # 5 minutes
        
        # Request deduplication for concurrent requests
        self._pending_requests = {}
        self._request_locks = {}

    def _call_gemini_llm(self, prompt: str) -> str:
        headers = {'Content-Type': 'application/json'}
        params = {'key': self.gemini_api_key}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        max_retries = 1  # Changed to 1 attempt
        backoff_factor = 0.0 # No backoff needed for single attempt
        initial_delay = 0 # No initial delay for single attempt

        for attempt in range(max_retries):
            try:
                response = requests.post(self.gemini_url, headers=headers, params=params, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                if 'candidates' in result and result['candidates'] and 'content' in result['candidates'][0] and 'parts' in result['candidates'][0]['content']:
                    return result['candidates'][0]['content']['parts'][0]['text']
                logger.error(f"❌ Unexpected Gemini AI response format")
                return "Error: The Gemini AI returned a response in an unexpected format."

            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️  Gemini AI call failed (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    delay = initial_delay * (backoff_factor ** attempt)
                    logger.info(f"🔄 Retrying...")
                    time.sleep(delay)
                else:
                    logger.error("❌ Gemini AI service unavailable after multiple attempts")
                    return f"Error: Could not connect to the Gemini AI service after multiple attempts. Details: {e}"
            
            except (KeyError, IndexError) as e:
                logger.error(f"❌ Error parsing Gemini AI response")
                return "Error: The Gemini AI returned an invalid response."
        
        return "Error: The Gemini AI service is currently unavailable."

    def _call_openai_llm(self, prompt: str, max_retries: int = 1, purpose: str = "General") -> str:
        if not self.openai_client:
            return "Error: OpenAI API key not configured."
        
        for attempt in range(max_retries):
            try:
                chat_completion = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    timeout=60
                )
                return chat_completion.choices[0].message.content
            except openai.APIError as e:
                logger.warning(f"⚠️  AI call failed (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    logger.info(f"🔄 Retrying...")
                    time.sleep(5) # Increased backoff
                    continue
                else:
                    logger.error("❌ AI service unavailable after multiple attempts")
                    return f"Error: Could not connect to the AI service after multiple attempts. Details: {e}"
            except Exception as e:
                logger.error(f"❌ Unexpected error with AI service: {e}")
                return f"Error: An unexpected error occurred with AI service. Details: {e}"
        
        return "Error: The OpenAI AI service is currently unavailable."

    def _call_llm(self, prompt: str, max_retries: int = 1, purpose: str = "General") -> str:
        # Try OpenAI first
        if self.openai_api_key:
            openai_response = self._call_openai_llm(prompt, max_retries, purpose)
            if not openai_response.startswith("Error:"):
                return openai_response
            logger.warning("OpenAI failed, attempting Gemini fallback.")
        
        # Fallback to Gemini
        if self.gemini_api_key:
            return self._call_gemini_llm(prompt)
        
        return "Error: No active LLM service available (OpenAI failed and Gemini not configured or failed)."

    def _call_llm_with_cache(self, prompt: str, max_retries: int = 1, purpose: str = "General") -> str:
        """
        LLM call with simple caching for repeated prompts.
        """
        import hashlib
        
        # Create cache key from prompt hash
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        cache_key = f"{purpose}_{prompt_hash}"
        
        # Check cache
        if cache_key in self._sample_data_cache and \
           datetime.now() - self._sample_data_cache[cache_key]["timestamp"] < timedelta(seconds=60):  # 1 minute cache for LLM responses
            logger.info(f"Using cached LLM response for {purpose}")
            return self._sample_data_cache[cache_key]["data"]
        
        # Call LLM
        response = self._call_llm(prompt, max_retries, purpose)
        
        # Cache response
        self._sample_data_cache[cache_key] = {
            "data": response,
            "timestamp": datetime.now()
        }
        
        return response

    def _deduplicated_api_call(self, call_type: str, call_id: str, api_func, *args, **kwargs):
        """
        Prevents duplicate API calls for the same data during concurrent operations.
        """
        import threading
        import time
        
        cache_key = f"{call_type}_{call_id}"
        
        # Check if we already have the data
        if cache_key in self._sample_data_cache and \
           datetime.now() - self._sample_data_cache[cache_key]["timestamp"] < timedelta(seconds=self._cache_timeout):
            logger.info(f"Using cached result for {call_type}: {call_id}")
            return self._sample_data_cache[cache_key]["data"]
        
        # Check if request is already in progress
        if cache_key in self._pending_requests:
            logger.info(f"Request already in progress for {call_type}: {call_id}, waiting...")
            # Wait for the pending request to complete
            while cache_key in self._pending_requests:
                time.sleep(0.1)
            # Return the result from the completed request
            if cache_key in self._sample_data_cache:
                return self._sample_data_cache[cache_key]["data"]
        
        # Mark request as in progress
        self._pending_requests[cache_key] = True
        
        try:
            # Make the actual API call
            result = api_func(*args, **kwargs)
            
            # Cache the result
            self._sample_data_cache[cache_key] = {
                "data": result,
                "timestamp": datetime.now()
            }
            
            return result
            
        finally:
            # Remove from pending requests
            if cache_key in self._pending_requests:
                del self._pending_requests[cache_key]

    def _pre_filter_cards(self, cards: List[Dict], prompt: str) -> List[Dict]:
        """
        Fast pre-filtering to reduce cards before sending to LLM.
        Returns top 8-10 most relevant cards based on keyword matching.
        """
        prompt_lower = prompt.lower()
        scored_cards = []
        
        # Quick keyword scoring
        for card in cards:
            score = 0
            card_name = self._safe_get_card_field(card, 'card_name').lower()
            card_desc = self._safe_get_card_field(card, 'card_description').lower()
            
            # High-priority keywords
            high_priority_words = ['disease', 'patient', 'mhu', 'health', 'prescription', 'lab', 'pharmacy']
            for word in high_priority_words:
                if word in prompt_lower and word in card_name:
                    score += 10
                elif word in prompt_lower and word in card_desc:
                    score += 5
            
            # Exact phrase matches
            prompt_words = prompt_lower.split()
            for word in prompt_words:
                if len(word) > 3:
                    if word in card_name:
                        score += 3
                    elif word in card_desc:
                        score += 1
            
            scored_cards.append((card, score))
        
        # Sort and return top cards
        scored_cards.sort(key=lambda x: x[1], reverse=True)
        top_cards = [card for card, score in scored_cards[:10]]  # Top 10 cards
        
        logger.info(f"🔍 Pre-filtered {len(cards)} cards down to {len(top_cards)} candidates")
        return top_cards

    def find_relevant_cards(self, cards: List[Dict], prompt: str, dashboard_id: int = None, dashboard_details: Dict = None) -> Tuple[List[Dict], str]:
        """
        Find relevant cards using a two-stage approach for better performance.
        Stage 1: Select 3 most relevant cards using metadata only
        Stage 2: Get sample data for those 3 cards and make final selection
        Returns tuple of (relevant_cards, explanation)
        """
        import time
        start_time = time.time()
        
        if not cards:
            return [], "No cards available in the dashboard."

        # ========================================
        # STAGE 1: INITIAL CARD SELECTION (Metadata Only)
        # ========================================
        logger.info("🔍 STAGE 1: Initial card selection using metadata...")
        
        initial_prompt = f'''
You are an intelligent healthcare data analyst assistant. Your task is to identify the 3 most relevant Metabase dashboard cards to answer a user's question.

IMPORTANT INSTRUCTIONS:
1. Analyze each card's name, description, and column metadata carefully
2. Look for semantic matches, not just exact keyword matches
3. Select exactly 3 cards that are most relevant to the question
4. For healthcare questions, prioritize cards with patient data, MHU information, and demographic breakdowns
5. For disease-related questions, prioritize cards with disease information
6. For count/summary questions, prefer aggregated data over individual records

Available Cards:
```json
{json.dumps(cards, indent=2)}
```

User's Question: "{prompt}"

Respond with a JSON object in this format:
{{
    "selected_cards": [
        // Array of exactly 3 card objects that are most relevant
    ],
    "explanation": "Brief explanation of why these 3 cards were selected"
}}

IMPORTANT: Return exactly 3 cards, no more, no less.
'''

        initial_response = self._call_llm_with_cache(initial_prompt, purpose="Initial Card Selection")
        
        try:
            # Clean and parse the initial response
            clean_str = initial_response.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            
            initial_result = json.loads(clean_str.strip())
            selected_cards = initial_result.get("selected_cards", [])
            
            if len(selected_cards) != 3:
                logger.warning(f"⚠️  AI didn't return exactly 3 cards ({len(selected_cards)}), using fallback...")
                selected_cards, _ = self._fallback_keyword_matching(cards, prompt)
                selected_cards = selected_cards[:3]  # Ensure exactly 3 cards
            
            logger.info(f"✅ Stage 1 complete: Selected {len(selected_cards)} cards")
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"❌ Initial AI response parsing failed, using fallback...")
            selected_cards, _ = self._fallback_keyword_matching(cards, prompt)
            selected_cards = selected_cards[:3]  # Ensure exactly 3 cards

        # ========================================
        # STAGE 2: SAMPLE DATA FOR SELECTED CARDS (PARALLEL)
        # ========================================
        logger.info("🔍 STAGE 2: Fetching sample data for selected cards...")
        
        enhanced_cards = []
        if dashboard_id and dashboard_details:
            # Use parallel fetching for better performance
            enhanced_cards = self._get_card_sample_data_parallel(selected_cards, dashboard_id, dashboard_details)
            logger.info(f"✅ Enhanced {len(enhanced_cards)} cards with sample data")
        else:
            enhanced_cards = selected_cards
            logger.warning("⚠️  No dashboard context available, using cards without sample data")

        # ========================================
        # STAGE 3: FINAL CARD SELECTION WITH SAMPLE DATA
        # ========================================
        logger.info("🔍 STAGE 3: Final card selection with sample data...")
        
        final_prompt = f'''
You are an intelligent healthcare data analyst assistant. Your task is to select the SINGLE most relevant card from the 3 pre-selected cards to answer the user's question.

IMPORTANT INSTRUCTIONS:
1. Analyze each card's name, description, column metadata, AND sample data carefully
2. Use the sample data to understand what kind of information each card contains
3. Select the ONE card that will provide the best answer to the user's question
4. Consider the actual data content, not just the card name
5. For disease-related questions, prefer cards with actual disease data
6. For count/summary questions, prefer aggregated data

Available Cards (with sample data):
```json
{json.dumps(enhanced_cards, indent=2)}
```

User's Question: "{prompt}"

Respond with a JSON object in this format:
{{
    "best_card": // The single most relevant card object
    "explanation": "Brief explanation of why this card is the best choice",
    "confidence": "high|medium|low"  // Your confidence in the selection
}}

IMPORTANT: Return exactly ONE card as the best choice.
'''

        final_response = self._call_llm_with_cache(final_prompt, purpose="Final Card Selection")
        
        try:
            # Clean and parse the final response
            clean_str = final_response.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            
            final_result = json.loads(clean_str.strip())
            best_card = final_result.get("best_card")
            explanation = final_result.get("explanation", "Card selected based on sample data analysis.")
            
            if best_card:
                logger.info(f"✅ Final selection: {self._safe_get_card_field(best_card, 'card_name', 'Unknown')}")
                return [best_card], explanation
            else:
                logger.warning("⚠️  No best card returned, using first selected card")
                return [selected_cards[0]], "Using first selected card as fallback"
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"❌ Final AI response parsing failed, using first selected card...")
            return [selected_cards[0]], "Using first selected card due to parsing error"
        
        # Log performance metrics
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"⚡ Card selection completed in {total_time:.2f} seconds")

    def _fallback_keyword_matching(self, cards: List[Dict], prompt: str) -> Tuple[List[Dict], str]:
        """Fallback strategy using simple keyword matching."""
        prompt_lower = prompt.lower()
        scored_cards = []
        
        for card in cards:
            score = 0
            card_text = f"{self._safe_get_card_field(card, 'card_name')} {self._safe_get_card_field(card, 'card_description')}"
            
            # Add column names to searchable text
            for col in card.get('result_metadata', []):
                card_text += f" {col.get('name') or ''} {col.get('display_name') or ''}"
            
            card_text_lower = card_text.lower()
            
            # Simple keyword scoring
            prompt_words = set(prompt_lower.split())
            card_words = set(card_text_lower.split())
            common_words = prompt_words.intersection(card_words)
            score = len(common_words)
            
            # Boost score for exact phrase matches
            if any(word in card_text_lower for word in prompt_words if len(word) > 3):
                score += 2
            
            scored_cards.append((card, score))
        
        # Sort by score and return top cards
        scored_cards.sort(key=lambda x: x[1], reverse=True)
        
        if scored_cards and scored_cards[0][1] > 0:
            # Return cards with positive scores
            relevant_cards = [card for card, score in scored_cards if score > 0][:3]
            explanation = f"Selected cards based on keyword matching with your query. Found {len(relevant_cards)} potentially relevant cards."
            logger.info(f"🔍 Keyword matching found {len(relevant_cards)} relevant cards")
        else:
            # If no good matches, return top 2 cards anyway
            relevant_cards = [card for card, _ in scored_cards[:2]]
            explanation = "No direct matches found, but selected the most comprehensive cards that might contain relevant information."
            logger.info(f"🔍 No keyword matches, using top 2 cards as fallback")
        
        return relevant_cards, explanation

    def _intelligent_fallback_matching(self, cards: List[Dict], prompt: str, is_count_request: bool, is_summary_request: bool, primary_intent: str = "general") -> Tuple[List[Dict], str]:
        """Fast intelligent matching using simplified keyword scoring."""
        prompt_lower = prompt.lower()
        scored_cards = []
        
        for card in cards:
            score = 0
            card_name = self._safe_get_card_field(card, 'card_name').lower()
            
            # Enhanced scoring for general patient count questions
            if "total patients count" in prompt_lower or "total patients" in prompt_lower:
                # For general patient count, prefer cards with "total" and "patients" but NOT gender-specific
                if "total" in card_name and "patient" in card_name and not any(gender in card_name for gender in ['male', 'female', 'child']):
                    score += 50  # Very high priority for general patient totals
                elif "total" in card_name and "patient" in card_name:
                    score += 30  # High priority for any patient totals
                elif "patient" in card_name and not any(gender in card_name for gender in ['male', 'female', 'child']):
                    score += 25  # Good priority for general patient cards
            elif is_count_request or is_summary_request:
                # Prefer aggregated cards for count/summary requests
                if any(word in card_name for word in ['total', 'count', 'summary', 'breakdown']):
                    score += 20
                elif any(word in card_name for word in ['list', 'details', 'patients']):
                    score += 5
            else:
                # For other requests, prefer detailed cards
                if any(word in card_name for word in ['list', 'details', 'patients', 'records']):
                    score += 15
                elif any(word in card_name for word in ['total', 'count', 'summary']):
                    score += 8
            
            # Simple keyword matching
            prompt_words = prompt_lower.split()
            for word in prompt_words:
                if len(word) > 3 and word in card_name:
                    score += 5
            
            # Healthcare term boost
            if any(term in card_name for term in ['patient', 'mhu', 'health']) and any(term in prompt_lower for term in ['patient', 'mhu', 'health']):
                score += 10
            
            scored_cards.append((card, score))
        
        # Sort and return top card
        scored_cards.sort(key=lambda x: x[1], reverse=True)
        
        if scored_cards:
            top_card = scored_cards[0][0]
            explanation = f"Selected '{self._safe_get_card_field(top_card, 'card_name', 'Unknown')}' using enhanced matching"
            return [top_card], explanation
        else:
            # Fallback to first card
            return [cards[0]], "No matches found, using first available card"

    def _process_individual_patient_data(self, card_data: Dict) -> Dict:
        """
        Fast processing of individual patient data with minimal aggregations.
        """
        if not card_data.get("data", {}).get("rows"):
            return card_data
        
        rows = card_data["data"]["rows"]
        
        # Quick aggregations without DataFrame overhead
        total_patients = len(rows)
        
        # Simple aggregations using dictionaries
        mhu_breakdown = {}
        gender_breakdown = {}
        
        for row in rows:
            # MHU breakdown (column 9 based on mapping)
            if len(row) > 9 and row[9]:
                mhu = str(row[9])
                mhu_breakdown[mhu] = mhu_breakdown.get(mhu, 0) + 1
            
            # Gender breakdown (column 2 based on mapping)
            if len(row) > 2 and row[2]:
                gender = str(row[2])
                gender_breakdown[gender] = gender_breakdown.get(gender, 0) + 1
        
        # Add minimal aggregations
        card_data["aggregations"] = {
            "total_patients": total_patients,
            "mhu_breakdown": mhu_breakdown,
            "gender_breakdown": gender_breakdown
        }
        card_data["data_type"] = "individual_patients"
        
        return card_data

    def _intelligent_data_filtering(self, card_data: Dict, question: str, date_value: Optional[str] = None) -> Tuple[Dict, str]:
        """
        Intelligently filter data based on size and apply date filters when needed.
        Returns (filtered_data, filter_explanation)
        """
        if not card_data.get("data", {}).get("rows"):
            return card_data, "No data available"
        
        rows = card_data["data"]["rows"]
        original_count = len(rows)
        
        # Check if question requires complete data (no filtering)
        question_lower = question.lower()
        requires_complete_data = any(phrase in question_lower for phrase in [
            "which has the most", "which has the highest", "which has the lowest",
            "highest number", "lowest number", "most patients", "least patients",
            "top", "bottom", "rank", "ranking", "compare", "comparison",
            "mobile health unit has the most", "mhu has the most"
        ])
        
        # For comparison questions, ALWAYS use complete data regardless of size
        if requires_complete_data:
            logger.info(f"📊 COMPARISON QUESTION DETECTED: Using complete dataset ({original_count} records) for accurate analysis")
            return card_data, f"Using complete dataset for accurate comparison ({original_count} records)"
        
        # If data is under 1000 rows, return as is
        if original_count <= 1000:
            return card_data, f"Using complete dataset ({original_count} records)"
        
        # Data is large, need to apply intelligent filtering
        logger.info(f"📊 Large dataset detected: {original_count} rows, applying intelligent filtering")
        
        # Check if we already have a date filter from user input
        if date_value:
            # User already specified a date range, use that
            return card_data, f"Using user-specified date filter: {date_value} ({original_count} records)"
        
        # No user date filter, apply intelligent date filtering
        # Look for date columns in the data
        date_columns = []
        if card_data.get("data", {}).get("cols"):
            for i, col in enumerate(card_data["data"]["cols"]):
                col_type = col.get("base_type", "")
                if col_type in ["type/DateTime", "type/Date"]:
                    date_columns.append((i, col.get("name", "")))
        
        if not date_columns:
            # No date columns found, use simple truncation with warning
            logger.warning("⚠️  No date columns found, using simple truncation")
            card_data["data"]["rows"] = rows[:1000]
            card_data["data"]["filtered"] = True
            card_data["data"]["filter_info"] = f"Truncated to first 1000 records (original: {original_count})"
            return card_data, f"Data truncated to first 1000 records (no date columns available)"
        
        # Apply intelligent date filtering based on question context
        date_filter_applied = False
        
        # Determine appropriate time period based on question
        if any(word in question_lower for word in ["recent", "latest", "current", "now", "today"]):
            # Recent data - last 30 days
            time_period = "last 30 days"
            days_back = 30
        elif any(word in question_lower for word in ["trend", "pattern", "over time", "history"]):
            # Historical analysis - last 90 days
            time_period = "last 90 days"
            days_back = 90
        elif any(word in question_lower for word in ["week", "weekly"]):
            # Weekly analysis - last 7 days
            time_period = "last 7 days"
            days_back = 7
        elif any(word in question_lower for word in ["month", "monthly"]):
            # Monthly analysis - last 30 days
            time_period = "last 30 days"
            days_back = 30
        else:
            # Default to last 30 days for general questions
            time_period = "last 30 days"
            days_back = 30
        
        # Try to filter by date
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        filtered_rows = []
        date_col_index = date_columns[0][0]  # Use first date column
        
        for row in rows:
            if len(row) > date_col_index and row[date_col_index]:
                try:
                    # Try to parse the date
                    if isinstance(row[date_col_index], str):
                        row_date = datetime.fromisoformat(row[date_col_index].replace('Z', '+00:00'))
                    else:
                        row_date = row[date_col_index]
                    
                    if row_date >= cutoff_date:
                        filtered_rows.append(row)
                except (ValueError, TypeError):
                    # If date parsing fails, include the row
                    filtered_rows.append(row)
            else:
                # No date value, include the row
                filtered_rows.append(row)
        
        # Check if filtering was effective
        if len(filtered_rows) <= 1000:
            # Date filtering worked well
            card_data["data"]["rows"] = filtered_rows
            card_data["data"]["filtered"] = True
            card_data["data"]["filter_info"] = f"Filtered by {time_period} (original: {original_count}, filtered: {len(filtered_rows)})"
            date_filter_applied = True
            logger.info(f"✅ Date filtering applied: {len(filtered_rows)} records from {time_period}")
        else:
            # Date filtering didn't reduce enough, use truncation
            logger.warning("⚠️  Date filtering insufficient, using truncation")
            card_data["data"]["rows"] = filtered_rows[:1000]
            card_data["data"]["filtered"] = True
            card_data["data"]["filter_info"] = f"Filtered by {time_period} then truncated to 1000 (original: {original_count})"
            date_filter_applied = True
        
        if date_filter_applied:
            return card_data, f"Applied {time_period} filter: {len(card_data['data']['rows'])} records (original: {original_count})"
        else:
            # Fallback to simple truncation
            card_data["data"]["rows"] = rows[:1000]
            card_data["data"]["filtered"] = True
            card_data["data"]["filter_info"] = f"Truncated to first 1000 records (original: {original_count})"
            return card_data, f"Data truncated to first 1000 records (date filtering failed)"

    # Removed _preprocess_for_aggregation method to reduce processing overhead

    def _fetch_card_data_with_params(
        self,
        card_id: int,
        dashcard_id: Optional[int],
        date_column_name: Optional[str],
        date_value: Optional[str],
        card_definition: Dict,
        static_date_filter: Optional[Dict] = None,
        dashboard_id: Optional[int] = None
    ) -> Optional[Dict]:
        # Simplified parameter handling
        parameters = [static_date_filter] if static_date_filter else []

        # Fetch data using dashboard endpoint
        card_data = self.metabase_client.get_dashboard_card_data(dashboard_id, dashcard_id, card_id, parameters=parameters)
        if not card_data:
            logger.warning(f"Failed to get data for card ID {card_id}")
            return None

        # Simplified data extraction
        data_rows = card_data.get("data", {}).get("rows", [])
        column_names = []
        
        if card_data.get("data", {}).get("cols"):
            column_names = [col.get("display_name", col.get("name", "")) for col in card_data["data"]["cols"]]

        return {
            "card_id": card_id,
            "name": card_definition.get("name", ""),
            "data": {
                "rows": data_rows,
                "column_names": column_names
            }
        }

    def _optimize_card_data_fetch(self, card_id: int, dashcard_id: int, dashboard_id: int, limit: int = 5) -> Optional[Dict]:
        """
        Optimized card data fetching with minimal data transfer.
        Uses the most efficient Metabase API approach.
        """
        try:
            # Use dashboard card endpoint for better performance
            card_data = self.metabase_client.get_dashboard_card_data(
                dashboard_id, dashcard_id, card_id, parameters=[]
            )
            
            if not card_data or not card_data.get("data", {}).get("rows"):
                return None
            
            # Extract only necessary data
            rows = card_data["data"]["rows"]
            cols = card_data["data"].get("cols", [])
            
            # Limit rows for performance
            limited_rows = rows[:limit] if len(rows) > limit else rows
            
            return {
                "rows": limited_rows,
                "total_rows": len(rows),
                "columns": cols,
                "sample_size": len(limited_rows)
            }
            
        except Exception as e:
            logger.error(f"Error fetching optimized data for card {card_id}: {e}")
            return None

    def _get_card_sample_data(self, card: Dict, dashboard_id: int, dashboard_details: Dict) -> Dict:
        """
        Fetch sample data (10 rows) from a card to help with card selection.
        Returns card with sample data added.
        """
        try:
            card_id = card.get("card_id")
            if not card_id:
                return card
            
            # Check cache first
            if card_id in self._sample_data_cache and \
               datetime.now() - self._sample_data_cache[card_id]["timestamp"] < timedelta(seconds=self._cache_timeout):
                logger.info(f"Using cached sample data for card {card.get('card_name', card_id)}")
                return self._sample_data_cache[card_id]["data"]

            # Find dashcard_id for this card
            dashcard_id = None
            if dashboard_details and "dashcards" in dashboard_details:
                for dc in dashboard_details["dashcards"]:
                    if dc.get("card_id") == card_id:
                        dashcard_id = dc.get("id")
                        break
            
            # Get card definition
            card_definition = self.metabase_client.get_card_definition(card_id)
            if not card_definition:
                logger.warning(f"Could not get definition for card {card_id}")
                return card
            
            # Use optimized data fetching
            optimized_data = self._optimize_card_data_fetch(card_id, dashcard_id, dashboard_id, limit=5)
            
            if optimized_data:
                # Add sample data to card
                sample_data = {
                    "rows": optimized_data["rows"],
                    "total_rows": optimized_data["total_rows"],
                    "columns": optimized_data["columns"]
                }
                card["sample_data"] = sample_data
                
                # Cache the sample data
                self._sample_data_cache[card_id] = {
                    "data": card,
                    "timestamp": datetime.now()
                }
                
                logger.info(f"✅ Added optimized sample data for card {card.get('card_name', card_id)}: {len(optimized_data['rows'])} rows")
            else:
                logger.warning(f"No data available for card {card.get('card_name', card_id)}")
                
        except Exception as e:
            logger.error(f"Error fetching sample data for card {card.get('card_name', 'Unknown')}: {e}")
        
        return card

    def _get_card_sample_data_parallel(self, cards: List[Dict], dashboard_id: int, dashboard_details: Dict) -> List[Dict]:
        """
        Fetch sample data for multiple cards in parallel for better performance.
        Returns list of cards with sample data added.
        """
        import concurrent.futures
        import time
        
        def fetch_single_card_sample(card):
            return self._get_card_sample_data(card, dashboard_id, dashboard_details)
        
        start_time = time.time()
        logger.info(f"🔄 Fetching sample data for {len(cards)} cards in parallel...")
        
        # Use ThreadPoolExecutor for parallel API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            enhanced_cards = list(executor.map(fetch_single_card_sample, cards))
        
        end_time = time.time()
        logger.info(f"⚡ Parallel sample data fetch completed in {end_time - start_time:.2f} seconds")
        
        return enhanced_cards

    def _batch_fetch_card_definitions(self, card_ids: List[int]) -> Dict[int, Dict]:
        """
        Fetch multiple card definitions efficiently.
        Since Metabase doesn't support batch API, we use parallel requests.
        """
        import concurrent.futures
        
        def fetch_single_definition(card_id):
            try:
                return card_id, self.metabase_client.get_card_definition(card_id)
            except Exception as e:
                logger.warning(f"Failed to fetch definition for card {card_id}: {e}")
                return card_id, {}
        
        # Use parallel requests to fetch all definitions
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(fetch_single_definition, card_ids))
        
        # Convert to dictionary
        definitions = {card_id: definition for card_id, definition in results if definition}
        logger.info(f"✅ Batch fetched {len(definitions)} card definitions")
        return definitions

    def answer_question(self, question: str, dashboard_id: int, chat_history: List[Dict], date_value: Optional[str] = None, simplified_card_details: List[Dict] = None, dashboard_details: Dict = None) -> str:
        if not simplified_card_details:
            logger.warning(f"No simplified card details provided for dashboard ID: {dashboard_id}")
            return json.dumps({
                "response_parts": [{
                    "type": "text",
                    "content": f"I couldn't get card details for dashboard ID {dashboard_id}. Please ensure the dashboard exists and contains cards."
                }]
            })

        # ========================================
        # 🚀 STARTING ANALYSIS PROCESS
        # ========================================
        logger.info("=" * 50)
        logger.info("🚀 STARTING ANALYSIS")
        logger.info(f"📊 Dashboard ID: {dashboard_id}")
        logger.info(f"📝 Question: '{question}'")
        logger.info("=" * 50)

        # ========================================
        # STEP 1: FINDING RELEVANT CARDS
        # ========================================
        logger.info("")
        logger.info("🔍 STEP 1: FINDING RELEVANT CARDS")
        logger.info("🤖 Calling AI (For Card Selection)...")
        
        # Pre-filter cards to reduce the number of cards sent to the LLM
        pre_filtered_cards = self._pre_filter_cards(simplified_card_details, question)
        relevant_cards_data, explanation = self.find_relevant_cards(pre_filtered_cards, question, dashboard_id, dashboard_details)
        
        if not relevant_cards_data:
            logger.warning("❌ No relevant cards found!")
            return json.dumps({
                "response_parts": [{
                    "type": "text",
                    "content": f"I couldn't identify any relevant cards for your question in dashboard ID {dashboard_id}. Please try rephrasing your question or select a different dashboard."
                }]
            })

        # Select the most relevant card
        top_card = relevant_cards_data[0]
        card_names = [card.get('card_name', 'Unknown') for card in relevant_cards_data[:3]]
        logger.info(f"📋 Found {len(relevant_cards_data)} relevant cards - {', '.join(card_names)}")
        logger.info(f"💡 Selected - {top_card.get('card_name', 'Unknown')} as Top Match")
        logger.info("=" * 50)

        # ========================================
        # STEP 2: FETCHING DATA
        # ========================================
        logger.info("")
        logger.info(f"📊 STEP 2: FETCHING DATA FROM {top_card.get('card_name', 'Unknown')}")
        
        card_id = top_card["card_id"]
        
        # Build mapping from card_id to dashcard_id and find exact card match
        cardid_to_dashcardid = {}
        exact_card_match = None
        
        if dashboard_details and "dashcards" in dashboard_details:
            for dc in dashboard_details["dashcards"]:
                if dc.get("card_id") and dc.get("id"):
                    cardid_to_dashcardid[dc["card_id"]] = dc["id"]
                    # Find exact card name match
                    if dc.get("card", {}).get("name") == top_card.get('card_name'):
                        exact_card_match = dc

        # Use exact match if found, otherwise fall back to original card_id
        if exact_card_match:
            card_id = exact_card_match["card_id"]
            dashcard_id = exact_card_match["id"]
        else:
            dashcard_id = cardid_to_dashcardid.get(card_id)
        
        # Get card definition
        card_definition = self.metabase_client.get_card_definition(card_id)
        if not card_definition:
            logger.error("❌ Failed to get card definition!")
            return json.dumps({
                "response_parts": [{
                    "type": "text",
                    "content": f"I found a relevant card but couldn't retrieve its definition. Please try again."
                }]
            })

        # Check for date columns
        date_column_name = None
        if card_definition.get("result_metadata"):
            for col in card_definition["result_metadata"]:
                col_type = col.get("base_type", "")
                if col_type in ["type/DateTime", "type/Date"]:
                    date_column_name = col.get("name", "")
                    break

        # Apply date filter if needed
        static_date_filter = None
        if date_value:
            static_date_filter = {
                "type": "date/all-options",
                "value": date_value,
                "id": "7f30ced2",
                "target": ["dimension", ["field", "date", {"base-type": "type/DateTime"}]]
            }
            logger.info(f"📅 Applied date filter: {date_value}")

        # Fetch the actual data
        try:
            card_data = self._fetch_card_data_with_params(
                card_id, dashcard_id, date_column_name, date_value, 
                card_definition, static_date_filter, dashboard_id
            )
            
            if not card_data:
                logger.error("❌ No data returned from the card!")
                return json.dumps({
                    "response_parts": [{
                        "type": "text",
                        "content": "I found a relevant card but couldn't retrieve data from it. This might be due to permissions or data availability."
                    }]
                })
            
            data_rows = card_data.get("data", {}).get("rows", [])
            logger.info(f"✅ DATA RETRIEVED: All Rows Data Included")
            
        except Exception as exc:
            logger.error(f"❌ Data fetch failed: {exc}")
            return json.dumps({
                "response_parts": [{
                    "type": "text",
                    "content": "I encountered an error while fetching data. Please try again."
                }]
            })

        # ========================================
        # STEP 3: PROCESSING DATA
        # ========================================
        logger.info("")
        logger.info("⚙️  STEP 3: PROCESSING DATA")
        
        processed_card = self._process_individual_patient_data(card_data)
        
        # Apply intelligent data filtering
        filtered_card, filter_explanation = self._intelligent_data_filtering(processed_card, question, date_value)
        logger.info(f"📊 Data filtering: {filter_explanation}")
        
        # Debug: Log the actual data being sent to LLM
        data_rows = filtered_card.get("data", {}).get("rows", [])
        logger.info(f"🔍 DEBUG: Sending {len(data_rows)} rows to LLM for analysis")
        if data_rows and len(data_rows) > 0:
            logger.info(f"🔍 DEBUG: First row: {data_rows[0]}")
            logger.info(f"🔍 DEBUG: Last row: {data_rows[-1]}")
            
            # Log ALL rows to see the complete data structure
            logger.info(f"🔍 DEBUG: Complete data structure:")
            for i, row in enumerate(data_rows):
                logger.info(f"🔍 DEBUG: Row {i}: {row}")
            
            # Log the highest value in the data
            highest_value = 0
            highest_item = None
            for row in data_rows:
                if len(row) >= 2:
                    try:
                        value = float(row[-1]) if isinstance(row[-1], (int, float)) else 0
                        if value > highest_value:
                            highest_value = value
                            highest_item = row[0] if len(row) > 0 else "Unknown"
                    except (ValueError, TypeError):
                        continue
            logger.info(f"🔍 DEBUG: Actual highest value in data: {highest_item} = {highest_value}")
        
        # Enhanced data preprocessing for better AI analysis
        enhanced_card = self._enhance_data_for_analysis(filtered_card, question)
        
        dashboard_context = {
            "dashboard_name": dashboard_details.get("name", "Unknown Dashboard"),
            "cards": [enhanced_card]
        }
        
        # Debug: Log what we're sending to the AI
        logger.info(f"🔍 AI INPUT DEBUG: Sending enhanced card to AI:")
        logger.info(f"🔍 AI INPUT DEBUG: Card name: {enhanced_card.get('card_name', 'Unknown')}")
        logger.info(f"🔍 AI INPUT DEBUG: Data analysis: {enhanced_card.get('data_analysis', {})}")
        logger.info(f"🔍 AI INPUT DEBUG: Analysis hints: {enhanced_card.get('analysis_hints', {})}")
        
        # Manual calculation debug
        self._debug_ai_calculation(data_rows, question)
        
        # Prepare data for LLM processing
        llm_dashboard_context = json.loads(json.dumps(dashboard_context))
        
        # Add filtering information
        filter_info = ""
        if filtered_card.get("data", {}).get("filtered"):
            filter_info = filtered_card["data"].get("filter_info", "Data was filtered")
        
        # Stage 1: Data Analysis AI - Perform calculations
        analysis_results = self._stage1_data_analysis(llm_dashboard_context, question)
        
        # Stage 2: Response Generation AI - Create final response
        final_response = self._stage2_response_generation(analysis_results, question, filter_info)
        
        # Convert to JSON string for return
        if isinstance(final_response, dict):
            return json.dumps(final_response, indent=2)
        else:
            return final_response

    def _stage1_data_analysis(self, dashboard_context: Dict, question: str) -> Dict:
        """
        Stage 1: AI performs all calculations and data analysis.
        """
        prompt = f"""
You are an expert data analyst. Your task is to analyze the provided data and perform all necessary calculations to answer the user's question.

**Data:**
```json
{json.dumps(dashboard_context, indent=2)}
```

**Question:** {question}

**Instructions:**
1. **ANALYZE THE RAW DATA**: Examine the data structure and identify key patterns
2. **PERFORM ALL CALCULATIONS**: 
   - Sum values across categories
   - Calculate percentages and ratios
   - Find maximum, minimum, average values
   - Perform aggregations by groups
   - Calculate totals, differences, or other derived metrics
3. **IDENTIFY KEY INSIGHTS**: Find trends, outliers, and significant patterns
4. **VALIDATE YOUR CALCULATIONS**: Double-check all math
5. **ORGANIZE RESULTS**: Structure your findings clearly

**Output Format:**
Return ONLY valid JSON with your analysis results:

```json
{{
    "calculations": {{
        "totals": {{"location": value, ...}},
        "percentages": {{"location": percentage, ...}},
        "rankings": ["location1", "location2", ...],
        "key_metrics": {{"highest": "location", "lowest": "location", "average": value}}
    }},
    "insights": [
        "Key insight 1",
        "Key insight 2",
        "Key insight 3"
    ],
    "data_summary": {{
        "total_records": number,
        "locations_analyzed": number,
        "categories_processed": number
    }},
    "chart_data": {{
        "chart_type": "bar|pie|line",
        "data": [{{"category": "value", "count": number}}],
        "title": "Chart title"
    }}
}}
```

**Important:** Focus ONLY on calculations and analysis. Do not write explanations or narratives.
"""
        
        try:
            analysis_response = self._call_llm_with_cache(prompt, max_retries=1, purpose="DataAnalysis")
            logger.info(f"🔍 STAGE1 DEBUG: Analysis AI response received")
            
            # Clean the response - remove markdown code blocks
            clean_response = analysis_response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            logger.info(f"🔍 STAGE1 DEBUG: Cleaned response: {clean_response[:200]}...")
            
            # Parse the analysis results
            try:
                analysis_results = json.loads(clean_response)
                logger.info(f"🔍 STAGE1 DEBUG: Analysis results parsed successfully")
                return analysis_results
            except json.JSONDecodeError as e:
                logger.error(f"❌ STAGE1 ERROR: Failed to parse analysis JSON: {e}")
                logger.error(f"❌ STAGE1 ERROR: Raw response: {analysis_response}")
                logger.error(f"❌ STAGE1 ERROR: Cleaned response: {clean_response}")
                return {"error": "Failed to parse analysis results"}
                
        except Exception as e:
            logger.error(f"❌ STAGE1 ERROR: Analysis AI failed: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def _stage2_response_generation(self, analysis_results: Dict, question: str, filter_info: str) -> str:
        """
        Stage 2: AI generates final user-friendly response based on analysis results.
        """
        if "error" in analysis_results:
            return f"❌ **Analysis Error**: {analysis_results['error']}"
        
        prompt = f"""
You are an expert healthcare data analyst creating executive summaries. Use the provided analysis results to create a comprehensive, user-friendly response.

**Question:** {question}

**Analysis Results:**
```json
{json.dumps(analysis_results, indent=2)}
```

**Filter Information:** {filter_info}

**Instructions:**
1. **CREATE EXECUTIVE SUMMARY**: Write a clear, professional analysis
2. **INCLUDE KEY NUMBERS**: Use the calculated totals, percentages, and rankings
3. **PROVIDE INSIGHTS**: Explain what the data means and why it matters
4. **CREATE VISUALIZATION**: Use the chart data to create an appropriate chart
5. **GENERATE FOLLOW-UP QUESTIONS**: Suggest strategic next steps
6. **USE PROFESSIONAL LANGUAGE**: Suitable for executive presentation

**Output Format:**
Return ONLY valid JSON in this exact format:

```json
{{
    "response_parts": [
        {{
            "type": "text",
            "content": "Comprehensive executive summary with specific numbers, insights, and recommendations. Include any data filtering or time periods applied."
        }},
        {{
            "type": "chart",
            "spec": {{
                "chart_type": "bar|pie|line",
                "data": [{{"Category": "Example", "Patients": 150}}],
                "title": "Clear, Descriptive Chart Title",
                "labels_column": "Category",
                "values_column": "Patients"
            }}
        }}
    ],
    "suggested_questions": [
        "Strategic question about operational implications",
        "Analytical question exploring deeper trends",
        "Predictive question for future planning"
    ]
}}
```

**Important:** Focus on creating a professional, actionable response that executives can use for decision-making.
"""
        
        try:
            response = self._call_llm_with_cache(prompt, max_retries=1, purpose="ResponseGeneration")
            logger.info(f"🔍 STAGE2 DEBUG: Response AI completed")
            
            # Clean the response - remove markdown code blocks
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            logger.info(f"🔍 STAGE2 DEBUG: Cleaned response: {clean_response[:200]}...")
            
            # Parse the final response
            try:
                response_json = json.loads(clean_response)
                logger.info(f"🔍 STAGE2 DEBUG: Response parsed successfully")
                return response_json
            except json.JSONDecodeError as e:
                logger.error(f"❌ STAGE2 ERROR: Failed to parse response JSON: {e}")
                logger.error(f"❌ STAGE2 ERROR: Raw response: {response}")
                logger.error(f"❌ STAGE2 ERROR: Cleaned response: {clean_response}")
                return {"error": "Failed to parse final response"}
                
        except Exception as e:
            logger.error(f"❌ STAGE2 ERROR: Response generation failed: {e}")
            return {"error": f"Response generation failed: {str(e)}"}

        # Prepare data for LLM processing

    def parse_date_filters(self, question: str, selected_date: Optional[str] = None) -> Dict[str, Optional[str]]:
        date_value = None

        if selected_date: # Prioritize date from Streamlit date picker
            date_value = datetime.strptime(str(selected_date), "%Y-%m-%d").date().isoformat()
        else: # Try to parse relative dates from the prompt
            today = datetime.now().date()
            question_lower = question.lower()

            if "yesterday" in question_lower:
                date_value = "yesterday"
            elif "today" in question_lower:
                date_value = "today"
            elif "last week" in question_lower:
                start_of_current_week = today - timedelta(days=today.weekday())
                start_of_last_week = start_of_current_week - timedelta(days=7)
                end_of_last_week = start_of_last_week + timedelta(days=6)
                date_value = f"{start_of_last_week.isoformat()}~{end_of_last_week.isoformat()}"
            elif "last month" in question_lower:
                # Calculate first day of current month
                first_day_current_month = today.replace(day=1)
                # Calculate last day of previous month
                end_of_last_month = first_day_current_month - timedelta(days=1)
                # Calculate first day of previous month
                start_of_last_month = end_of_last_month.replace(day=1)
                date_value = f"{start_of_last_month.isoformat()}~{end_of_last_month.isoformat()}"
            elif "this week" in question_lower:
                date_value = "thisweek"
            elif "this month" in question_lower:
                date_value = "thismonth"
            elif "this year" in question_lower:
                date_value = "thisyear"
            # Add more relative date parsing here as needed (e.g., "last year", specific date ranges)
        
        return {"date_value": date_value}

    def _safe_get_card_field(self, card: Dict, field: str, default: str = '') -> str:
        """
        Safely get card field value, handling None values.
        """
        value = card.get(field)
        if value is None:
            return default
        return str(value)

    def _validate_comparison_response(self, response_json: Dict, filtered_card: Dict, question: str) -> None:
        """
        Validates if the AI's response for comparison questions (e.g., "highest", "most")
        aligns with the actual data in the filtered_card.
        """
        question_lower = question.lower()
        is_comparison_request = any(phrase in question_lower for phrase in [
            "which has the most", "which has the highest", "which has the lowest",
            "highest number", "lowest number", "most patients", "least patients",
            "top", "bottom", "rank", "ranking", "compare", "comparison"
        ])

        if not is_comparison_request:
            return

        # Check if data was filtered for a comparison question
        if filtered_card.get("data", {}).get("filtered"):
            filter_info = filtered_card["data"].get("filter_info", "")
            logger.warning(f"⚠️  COMPARISON WARNING: Data was filtered ({filter_info}) for a comparison question. This may affect accuracy.")
            
            # Add a warning to the response
            if "response_parts" in response_json and response_json["response_parts"]:
                for part in response_json["response_parts"]:
                    if part.get("type") == "text":
                        part["content"] += f"\n\n⚠️ **Note**: Data was filtered for this analysis. For the most accurate comparison, consider asking for complete data."
                        break

    def _validate_and_correct_comparison(self, response_json: Dict, filtered_card: Dict, question: str) -> Dict:
        """
        Validates and corrects AI responses for comparison questions by checking actual data.
        """
        question_lower = question.lower()
        is_comparison_request = any(phrase in question_lower for phrase in [
            "which has the most", "which has the highest", "which has the lowest",
            "highest number", "lowest number", "most patients", "least patients",
            "top", "bottom", "rank", "ranking", "compare", "comparison"
        ])

        if not is_comparison_request:
            return response_json

        # Get the actual data
        data_rows = filtered_card.get("data", {}).get("rows", [])
        if not data_rows:
            return response_json

        # Enhanced analysis: Look at all numeric columns and perform calculations
        analysis_results = self._perform_data_analysis(data_rows, question)
        
        # Debug logging
        logger.info(f"🔍 VALIDATION DEBUG: Analysis results: {analysis_results}")
        
        # Check if AI response matches the calculated results
        ai_content = ""
        for part in response_json.get("response_parts", []):
            if part.get("type") == "text":
                ai_content = part.get("content", "").lower()
                break

        # Validate against calculated results
        corrections_needed = []
        
        if analysis_results.get("highest_item") and analysis_results.get("highest_value"):
            highest_item = analysis_results["highest_item"]
            highest_value = analysis_results["highest_value"]
            
            logger.info(f"🔍 VALIDATION DEBUG: Expected highest: {highest_item} ({highest_value:,.0f})")
            logger.info(f"🔍 VALIDATION DEBUG: AI content contains '{highest_item.lower()}'? {highest_item.lower() in ai_content}")
            
            # Check if AI mentioned the wrong highest value
            # Look for various ways the AI might indicate "most" or "highest"
            ai_mentions_highest = any(phrase in ai_content for phrase in [
                "most patients", "highest", "leads", "top", "highest number", "most"
            ])
            
            if ai_mentions_highest and highest_item.lower() not in ai_content:
                corrections_needed.append(f"Highest value: {highest_item} ({highest_value:,.0f})")
                logger.warning(f"🔍 VALIDATION DEBUG: AI mentioned 'most' but didn't mention the correct highest value!")
                logger.warning(f"🔍 VALIDATION DEBUG: AI content: {ai_content[:200]}...")
        
        if analysis_results.get("lowest_item") and analysis_results.get("lowest_value"):
            lowest_item = analysis_results["lowest_item"]
            lowest_value = analysis_results["lowest_value"]
            
            # Check if AI mentioned the wrong lowest value
            ai_mentions_lowest = any(phrase in ai_content for phrase in [
                "least patients", "lowest", "bottom", "lowest number", "least"
            ])
            
            if ai_mentions_lowest and lowest_item.lower() not in ai_content:
                corrections_needed.append(f"Lowest value: {lowest_item} ({lowest_value:,.0f})")

        # Apply corrections if needed
        if corrections_needed:
            logger.warning(f"⚠️  AI INCORRECT: Found discrepancies, correcting response")
            
            # Create a more user-friendly correction message
            correction_summary = self._create_user_friendly_correction(corrections_needed, ai_content, analysis_results)
            
            # Update the response
            for part in response_json.get("response_parts", []):
                if part.get("type") == "text":
                    part["content"] = correction_summary
                    break
        else:
            logger.info(f"✅ VALIDATION DEBUG: AI response appears to be correct!")

        return response_json

    def _perform_data_analysis(self, data_rows: List, question: str) -> Dict:
        """
        Performs comprehensive data analysis to support validation.
        """
        results = {}
        
        if not data_rows:
            return results
        
        # Find numeric columns
        numeric_columns = []
        for i, value in enumerate(data_rows[0]):
            try:
                float(value)
                numeric_columns.append(i)
            except (ValueError, TypeError):
                continue
        
        # For patient-related questions, perform aggregation analysis
        question_lower = question.lower()
        if "patients" in question_lower and len(numeric_columns) > 1:
            # This is likely a breakdown by gender/age - aggregate the data
            logger.info(f"🔍 ANALYSIS DEBUG: Performing patient aggregation analysis")
            logger.info(f"🔍 ANALYSIS DEBUG: Found {len(numeric_columns)} numeric columns")
            aggregated_results = self._aggregate_patient_data(data_rows, numeric_columns)
            logger.info(f"🔍 ANALYSIS DEBUG: Aggregation results: {aggregated_results}")
            if aggregated_results:
                results["primary_analysis"] = aggregated_results
                results["highest_item"] = aggregated_results.get("highest_item")
                results["highest_value"] = aggregated_results.get("highest_value")
                results["lowest_item"] = aggregated_results.get("lowest_item")
                results["lowest_value"] = aggregated_results.get("lowest_value")
                logger.info(f"🔍 ANALYSIS DEBUG: Set primary analysis - highest: {results['highest_item']} ({results['highest_value']})")
                return results
        
        # Fallback to individual column analysis
        for col_idx in numeric_columns:
            values = []
            items = []
            
            for row in data_rows:
                if len(row) > col_idx:
                    try:
                        value = float(row[col_idx])
                        values.append(value)
                        items.append(row[0] if len(row) > 0 else f"Item_{len(items)}")
                    except (ValueError, TypeError):
                        continue
            
            if values:
                # Find highest and lowest
                max_idx = values.index(max(values))
                min_idx = values.index(min(values))
                
                results[f"column_{col_idx}"] = {
                    "highest_item": items[max_idx],
                    "highest_value": values[max_idx],
                    "lowest_item": items[min_idx],
                    "lowest_value": values[min_idx],
                    "total": sum(values),
                    "average": sum(values) / len(values),
                    "count": len(values)
                }
        
        # For comparison questions, focus on the most relevant column
        if "patients" in question_lower:
            # Look for patient-related columns
            for col_idx in numeric_columns:
                if col_idx < len(data_rows[0]) and "patient" in str(data_rows[0][col_idx]).lower():
                    results["primary_analysis"] = results.get(f"column_{col_idx}", {})
                    break
        
        # If no specific column found, use the last numeric column
        if "primary_analysis" not in results and numeric_columns:
            last_col = numeric_columns[-1]
            results["primary_analysis"] = results.get(f"column_{last_col}", {})
        
        # Set overall results
        if "primary_analysis" in results:
            results["highest_item"] = results["primary_analysis"].get("highest_item")
            results["highest_value"] = results["primary_analysis"].get("highest_value")
            results["lowest_item"] = results["primary_analysis"].get("lowest_item")
            results["lowest_value"] = results["primary_analysis"].get("lowest_value")
        
        return results

    def _create_user_friendly_correction(self, corrections_needed: List[str], ai_content: str, analysis_results: Dict) -> str:
        """
        Creates a user-friendly correction message that's professional and easy to understand.
        """
        # Extract the key information
        highest_item = analysis_results.get("highest_item", "")
        highest_value = analysis_results.get("highest_value", 0)
        lowest_item = analysis_results.get("lowest_item", "")
        lowest_value = analysis_results.get("lowest_value", 0)
        
        # Create a professional correction message
        correction_message = f"""
## 📊 **Data Analysis Results**

Based on comprehensive calculations of the patient data across all Mobile Health Units, here are the accurate findings:

### 🏆 **Highest Patient Count**
**{highest_item}** leads with **{highest_value:,.0f} total patients**

### 📈 **Complete Breakdown**
"""
        
        # Add detailed breakdown if available
        if "primary_analysis" in analysis_results and "breakdown" in analysis_results["primary_analysis"]:
            breakdown = analysis_results["primary_analysis"]["breakdown"]
            for location, total in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
                correction_message += f"- **{location}**: {total:,.0f} patients\n"
        
        correction_message += f"""
### 🔍 **Analysis Summary**
The data was analyzed by aggregating patient counts across all categories (male, female, children) for each Mobile Health Unit. This comprehensive approach ensures accurate total patient counts for comparison.

### 📋 **Key Insights**
- **Total Locations Analyzed**: {analysis_results.get('primary_analysis', {}).get('location_count', 'N/A')}
- **Data Processing**: Aggregated patient counts by location
- **Validation**: Cross-verified with raw data calculations

*This analysis has been automatically validated against the source data to ensure accuracy.*
"""
        
        return correction_message

    def _debug_ai_calculation(self, data_rows: List, question: str) -> None:
        """
        Debug function to manually calculate what the AI should be seeing.
        """
        if not data_rows:
            logger.warning("🔍 AI CALC DEBUG: No data rows to analyze")
            return
        
        logger.info(f"🔍 AI CALC DEBUG: Manual calculation for AI verification")
        logger.info(f"🔍 AI CALC DEBUG: Question: {question}")
        logger.info(f"🔍 AI CALC DEBUG: Data rows: {len(data_rows)}")
        
        # Find numeric columns
        numeric_columns = []
        for i, value in enumerate(data_rows[0]):
            try:
                float(value)
                numeric_columns.append(i)
            except (ValueError, TypeError):
                continue
        
        logger.info(f"🔍 AI CALC DEBUG: Numeric columns found: {numeric_columns}")
        
        # Manual aggregation calculation
        location_totals = {}
        for row in data_rows:
            if len(row) == 0:
                continue
                
            location = row[0]
            total = 0
            
            logger.info(f"🔍 AI CALC DEBUG: Processing {location}:")
            for col_idx in numeric_columns:
                if len(row) > col_idx:
                    try:
                        value = float(row[col_idx])
                        total += value
                        logger.info(f"🔍 AI CALC DEBUG:   Column {col_idx}: {value} -> Running total: {total}")
                    except (ValueError, TypeError):
                        logger.warning(f"🔍 AI CALC DEBUG:   Column {col_idx}: Invalid value {row[col_idx]}")
                        continue
            
            location_totals[location] = total
            logger.info(f"🔍 AI CALC DEBUG: {location} FINAL TOTAL: {total}")
        
        # Find the highest
        if location_totals:
            highest_location = max(location_totals, key=location_totals.get)
            highest_value = location_totals[highest_location]
            logger.info(f"🔍 AI CALC DEBUG: MANUAL CALCULATION RESULT:")
            logger.info(f"🔍 AI CALC DEBUG: Highest: {highest_location} with {highest_value:,.0f} patients")
            logger.info(f"🔍 AI CALC DEBUG: All totals: {location_totals}")
        else:
            logger.warning("🔍 AI CALC DEBUG: No valid totals calculated")

    def _aggregate_patient_data(self, data_rows: List, numeric_columns: List[int]) -> Dict:
        """
        Aggregates patient data by summing across gender/age categories for each location.
        """
        if not data_rows or not numeric_columns:
            return {}
        
        # Group by location (first column) and sum all numeric columns
        location_totals = {}
        
        logger.info(f"🔍 AGGREGATION DEBUG: Processing {len(data_rows)} rows with {len(numeric_columns)} numeric columns")
        
        for row in data_rows:
            if len(row) == 0:
                continue
                
            location = row[0]  # First column is location
            total_patients = 0
            
            # Sum all numeric columns for this location
            for col_idx in numeric_columns:
                if len(row) > col_idx:
                    try:
                        value = float(row[col_idx])
                        total_patients += value
                        logger.info(f"🔍 AGGREGATION DEBUG: {location} - column {col_idx}: {value} -> running total: {total_patients}")
                    except (ValueError, TypeError):
                        continue
            
            location_totals[location] = total_patients
            logger.info(f"🔍 AGGREGATION DEBUG: {location} final total: {total_patients}")
        
        if not location_totals:
            return {}
        
        # Find highest and lowest totals
        locations = list(location_totals.keys())
        totals = list(location_totals.values())
        
        max_idx = totals.index(max(totals))
        min_idx = totals.index(min(totals))
        
        return {
            "highest_item": locations[max_idx],
            "highest_value": totals[max_idx],
            "lowest_item": locations[min_idx],
            "lowest_value": totals[min_idx],
            "total_patients": sum(totals),
            "location_count": len(locations),
            "breakdown": location_totals
        }

    def _enhance_data_for_analysis(self, card_data: Dict, question: str) -> Dict:
        """
        Enhances data structure to help AI perform better analysis and calculations.
        """
        enhanced_data = card_data.copy()
        data_rows = card_data.get("data", {}).get("rows", [])
        
        if not data_rows:
            return enhanced_data
        
        # Add data structure analysis
        enhanced_data["data_analysis"] = {
            "total_rows": len(data_rows),
            "column_count": len(data_rows[0]) if data_rows else 0,
            "data_types": [],
            "numeric_columns": [],
            "categorical_columns": [],
            "summary_stats": {}
        }
        
        if data_rows:
            # Analyze first row to understand structure
            first_row = data_rows[0]
            enhanced_data["data_analysis"]["column_names"] = card_data.get("data", {}).get("column_names", [])
            
            # Identify numeric and categorical columns
            for i, value in enumerate(first_row):
                try:
                    float_val = float(value)
                    enhanced_data["data_analysis"]["numeric_columns"].append(i)
                    enhanced_data["data_analysis"]["data_types"].append("numeric")
                except (ValueError, TypeError):
                    enhanced_data["data_analysis"]["categorical_columns"].append(i)
                    enhanced_data["data_analysis"]["data_types"].append("categorical")
            
            # Calculate summary statistics for numeric columns
            for col_idx in enhanced_data["data_analysis"]["numeric_columns"]:
                values = []
                for row in data_rows:
                    if len(row) > col_idx:
                        try:
                            values.append(float(row[col_idx]))
                        except (ValueError, TypeError):
                            continue
                
                if values:
                    enhanced_data["data_analysis"]["summary_stats"][f"column_{col_idx}"] = {
                        "sum": sum(values),
                        "count": len(values),
                        "max": max(values),
                        "min": min(values),
                        "average": sum(values) / len(values)
                    }
        
        # Add question-specific analysis hints
        question_lower = question.lower()
        if any(phrase in question_lower for phrase in ["most", "highest", "lowest", "compare"]):
            enhanced_data["analysis_hints"] = {
                "requires_aggregation": True,
                "suggested_calculations": ["sum", "max", "min", "group_by"],
                "comparison_needed": True
            }
        elif any(phrase in question_lower for phrase in ["percentage", "ratio", "proportion"]):
            enhanced_data["analysis_hints"] = {
                "requires_aggregation": True,
                "suggested_calculations": ["percentage", "ratio", "division"],
                "comparison_needed": False
            }
        else:
            enhanced_data["analysis_hints"] = {
                "requires_aggregation": False,
                "suggested_calculations": ["basic_stats"],
                "comparison_needed": False
            }
        
        return enhanced_data

    

    