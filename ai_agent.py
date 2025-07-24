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

    def find_relevant_cards(self, cards: List[Dict], prompt: str) -> Tuple[List[Dict], str]:
        """
        Find relevant cards with fallback strategies to ensure non-empty responses.
        Returns tuple of (relevant_cards, explanation)
        """
        if not cards:
            return [], "No cards available in the dashboard."

        # Enhanced LLM prompt with fallback instructions
        llm_prompt = f'''
You are an intelligent healthcare data analyst assistant. Your task is to identify the most relevant Metabase dashboard cards to answer a user's question.

IMPORTANT INSTRUCTIONS:
1. Analyze each card's name, description, and column metadata carefully
2. Look for semantic matches, not just exact keyword matches
3. If no cards seem directly relevant, choose the cards that could provide related or contextual information
4. ALWAYS return at least 1-3 cards, never an empty response
5. Provide a brief explanation of why you selected these cards
6. For healthcare questions, prioritize cards with patient data, MHU information, and demographic breakdowns
7. For count/summary questions, prefer aggregated data over individual records
8. For general patient count questions, avoid gender-specific cards unless specifically requested

Available Cards:
```json
{json.dumps(cards, indent=2)}
```

User's Question: "{prompt}"

Respond with a JSON object in this format:
{{
    "relevant_cards": [
        // Array of full card objects that are most relevant (1-3 cards)
    ],
    "explanation": "Brief explanation of why these cards were selected and how they relate to the question",
    "confidence": "high|medium|low"  // Your confidence in the relevance
}}

Remember: ALWAYS include at least one card, even if the match isn't perfect.
'''

        response_str = self._call_llm(llm_prompt, purpose="Card Selection")
        
        try:
            # Clean and parse the response
            clean_str = response_str.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            
            result = json.loads(clean_str.strip())
            
            relevant_cards = result.get("relevant_cards", [])
            explanation = result.get("explanation", "Cards selected based on available data.")
            
            # Fallback: if no cards returned, use keyword-based matching
            if not relevant_cards:
                logger.warning("⚠️  AI didn't find exact matches, using keyword fallback...")
                relevant_cards, explanation = self._fallback_keyword_matching(cards, prompt)
            
            return relevant_cards, explanation
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"❌ AI response parsing failed, using fallback...")
            # Use fallback strategy
            return self._fallback_keyword_matching(cards, prompt)

    def _fallback_keyword_matching(self, cards: List[Dict], prompt: str) -> Tuple[List[Dict], str]:
        """Fallback strategy using simple keyword matching."""
        prompt_lower = prompt.lower()
        scored_cards = []
        
        for card in cards:
            score = 0
            card_text = f"{card.get('card_name', '')} {card.get('card_description', '')}"
            
            # Add column names to searchable text
            for col in card.get('result_metadata', []):
                card_text += f" {col.get('name', '')} {col.get('display_name', '')}"
            
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
            card_name = card.get('card_name', '').lower()
            
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
            explanation = f"Selected '{top_card.get('card_name', 'Unknown')}' using enhanced matching"
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
        
        relevant_cards_data, explanation = self.find_relevant_cards(simplified_card_details, question)
        
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
        
        dashboard_context = {
            "dashboard_name": dashboard_details.get("name", "Unknown Dashboard"),
            "cards": [processed_card]
        }

        # Limit data for AI processing
        llm_dashboard_context = json.loads(json.dumps(dashboard_context))
        for card in llm_dashboard_context.get("cards", []):
            if "data" in card and isinstance(card["data"], list):
                if len(card["data"]) > 20:
                    card["data"] = card["data"][:20]

        # ========================================
        # STEP 4: GENERATING RESPONSE
        # ========================================
        logger.info("")
        logger.info("🤖 STEP 4: GENERATING RESPONSE")
        
        prompt = """
You are a healthcare data analyst. Analyze the data and provide a clear, concise response.

**Data:**
```json
{}
```

**Question:** {}

**Requirements:**
1. Provide a brief executive summary
2. Include a simple chart (bar/pie/line as appropriate)
3. Add 2-3 relevant follow-up questions
4. Use meaningful column names
5. Focus on key insights

Return ONLY valid JSON in this format:
```json
{{
    "response_parts": [
        {{
            "type": "text",
            "content": "Brief executive summary with key numbers"
        }},
        {{
            "type": "chart",
            "spec": {{
                "chart_type": "bar|pie|line",
                "data": [{{"Category": "Value", "Count": 123}}],
                "title": "Chart Title",
                "labels_column": "Category",
                "values_column": "Count"
            }}
        }}
    ],
    "suggested_questions": [
        "Follow-up question 1",
        "Follow-up question 2"
    ]
}}
```
""".format(
            json.dumps(llm_dashboard_context, indent=2),
            question
        )
        
        raw_response = self._call_llm(prompt, purpose="Response Generation")
        
        # ========================================
        # STEP 5: FINALIZING RESPONSE
        # ========================================
        logger.info("")
        logger.info("✨ STEP 5: FINALIZING RESPONSE")
        logger.info("=" * 50)
        
        try:
            # Clean the response
            clean_response = raw_response.strip()
            
            # Remove markdown code blocks if present
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            
            clean_response = clean_response.strip()
            
            # Remove JSON comments
            lines = clean_response.split('\n')
            cleaned_lines = []
            for line in lines:
                if '//' in line:
                    line = line.split('//')[0].rstrip()
                if line.strip():
                    cleaned_lines.append(line)
            clean_response = '\n'.join(cleaned_lines)
            
            # Parse JSON
            response_json = json.loads(clean_response)
            
            # Validate response structure
            if not isinstance(response_json, dict) or "response_parts" not in response_json:
                raise ValueError("Invalid response structure")
            
            logger.info("🎉 ANALYSIS COMPLETE")
            
            return json.dumps(response_json, indent=2)
            
        except Exception as e:
            logger.error(f"❌ Response processing failed: {e}")
            
            # Simple fallback response
            fallback_response = {
                "response_parts": [
                    {
                        "type": "text",
                        "content": "I analyzed the data but encountered an issue with the response format. Please try rephrasing your question."
                    }
                ],
                "suggested_questions": [
                    "Show me the total patient count",
                    "What is the gender distribution?",
                    "Which MHU has the most patients?"
                ]
            }
            return json.dumps(fallback_response, indent=2)

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

    

    