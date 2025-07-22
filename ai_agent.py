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
import re
from collections import Counter
import logging
from nltk.stem import PorterStemmer

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

    def _call_gemini_llm(self, prompt: str) -> str:
        headers = {'Content-Type': 'application/json'}
        params = {'key': self.gemini_api_key}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        max_retries = 1  # Only one attempt for Gemini as requested
        backoff_factor = 0.0 # No backoff needed for single attempt
        initial_delay = 0 # No initial delay for single attempt

        for attempt in range(max_retries):
            try:
                logger.debug(f"Calling Gemini LLM (Attempt {attempt + 1}/{max_retries})...")
                response = requests.post(self.gemini_url, headers=headers, params=params, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                if 'candidates' in result and result['candidates'] and 'content' in result['candidates'][0] and 'parts' in result['candidates'][0]['content']:
                    return result['candidates'][0]['content']['parts'][0]['text']
                logger.error(f"Unexpected Gemini LLM response format: {result}")
                return "Error: The Gemini AI returned a response in an unexpected format."

            except requests.exceptions.RequestException as e:
                logger.warning(f"Gemini LLM API call failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = initial_delay * (backoff_factor ** attempt)
                    logger.info(f"Retrying Gemini in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error("Gemini LLM API call failed after multiple retries.")
                    return f"Error: Could not connect to the Gemini AI service after multiple attempts. Details: {e}"
            
            except (KeyError, IndexError) as e:
                logger.error(f"Error parsing Gemini LLM response: {e}")
                return "Error: The Gemini AI returned an invalid response."
        
        return "Error: The Gemini AI service is currently unavailable."

    def _call_openai_llm(self, prompt: str) -> str:
        if not self.openai_client:
            return "Error: OpenAI API key not configured."
        try:
            logger.debug("Calling OpenAI LLM...")
            chat_completion = self.openai_client.chat.completions.create(
                model="gpt-4o", # Changed to gpt-4o as requested
                messages=[{"role": "user", "content": prompt}],
                timeout=60 # OpenAI timeout
            )
            return chat_completion.choices[0].message.content
        except openai.APIError as e:
            logger.error(f"OpenAI LLM API call failed: {e}")
            return f"Error: Could not connect to the OpenAI AI service. Details: {e}"
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI LLM call: {e}")
            return f"Error: An unexpected error occurred with OpenAI. Details: {e}"

    def _call_llm(self, prompt: str) -> str:
        # Try Gemini first
        if self.gemini_api_key:
            gemini_response = self._call_gemini_llm(prompt)
            if not gemini_response.startswith("Error:"):
                return gemini_response
            logger.warning("Gemini failed, attempting OpenAI fallback.")
        
        # Fallback to OpenAI
        if self.openai_api_key:
            return self._call_openai_llm(prompt)
        
        return "Error: No active LLM service available (Gemini failed and OpenAI not configured or failed)."

    def _preprocess_for_aggregation(self, dashboard_context: Dict, question: str) -> Dict:
        """
        Pre-process dashboard context for aggregation requests.
        Detects aggregation keywords and prepares data accordingly.
        """
        from analytics_utils import count_by_column, aggregate_data_by_column
        
        question_lower = question.lower()
        aggregation_keywords = ['by mhu', 'by location', 'count by', 'group by', 'total by', 'breakdown by']
        
        # Check if this is an aggregation request
        is_aggregation = any(keyword in question_lower for keyword in aggregation_keywords)
        
        if not is_aggregation:
            return dashboard_context
        
        logger = logging.getLogger(__name__)
        logger.info(f"Processing aggregation request: {question}")
        
        # Process each card's data for aggregation
        processed_context = dashboard_context.copy()
        processed_cards = []
        
        for card in dashboard_context.get("cards", []):
            card_data = card.get("data", [])
            card_name = card.get("name", "Unknown Card")
            
            logger.info(f"Processing card: {card_name} with {len(card_data)} rows")
            
            if not card_data:
                processed_cards.append(card)
                continue
            
            # Log the structure of the first row for debugging
            if card_data:
                first_row = card_data[0]
                logger.info(f"Card '{card_name}' columns: {list(first_row.keys())}")
                logger.info(f"Sample row: {first_row}")
            
            # Look for MHU-related aggregation
            if any(keyword in question_lower for keyword in ['mhu', 'mobile health unit']):
                # Check for various MHU column names
                mhu_columns = ['mhu', 'MHU', 'mobile_health_unit', 'mobile health unit', 'unit', 'health_unit']
                mhu_column_found = None
                
                for col in mhu_columns:
                    if card_data and col in card_data[0]:
                        mhu_column_found = col
                        break
                
                if mhu_column_found:
                    logger.info(f"Found MHU column '{mhu_column_found}' in card '{card_name}'")
                    aggregated = count_by_column(card_data, mhu_column_found)
                    
                    if aggregated["data"]:
                        logger.info(f"Aggregation result: {aggregated['data'][:3]}...")  # Show first 3 results
                        
                        # Add aggregated data as a new card
                        aggregated_card = {
                            "name": f"{card_name} - Aggregated by MHU",
                            "description": f"Patient count breakdown by Mobile Health Unit (using column: {mhu_column_found})",
                            "data": aggregated["data"],
                            "aggregated": True,
                            "original_card": card_name,
                            "debug_info": aggregated.get("debug_info", {})
                        }
                        processed_cards.append(aggregated_card)
                        logger.info(f"Added aggregated card for MHU breakdown")
                    else:
                        logger.warning(f"No aggregation data generated for MHU in card '{card_name}'")
                else:
                    logger.warning(f"No MHU column found in card '{card_name}'. Available columns: {list(card_data[0].keys()) if card_data else []}")
            
            # Look for location-related aggregation
            elif any(keyword in question_lower for keyword in ['location', 'place', 'area']):
                # Check for location-related columns
                location_columns = ['location', 'place', 'area', 'site', 'venue', 'Location', 'Place', 'Area']
                location_column_found = None
                
                for col in location_columns:
                    if card_data and col in card_data[0]:
                        location_column_found = col
                        break
                
                if location_column_found:
                    logger.info(f"Found location column '{location_column_found}' in card '{card_name}'")
                    aggregated = count_by_column(card_data, location_column_found)
                    
                    if aggregated["data"]:
                        aggregated_card = {
                            "name": f"{card_name} - Aggregated by {location_column_found.title()}",
                            "description": f"Patient count breakdown by {location_column_found}",
                            "data": aggregated["data"],
                            "aggregated": True,
                            "original_card": card_name,
                            "debug_info": aggregated.get("debug_info", {})
                        }
                        processed_cards.append(aggregated_card)
            
            # Look for gender-related aggregation
            elif any(keyword in question_lower for keyword in ['gender', 'male', 'female', 'child']):
                gender_columns = ['gender', 'sex', 'patient_type', 'Gender', 'Sex', 'Patient_Type']
                gender_column_found = None
                
                for col in gender_columns:
                    if card_data and col in card_data[0]:
                        gender_column_found = col
                        break
                
                if gender_column_found:
                    logger.info(f"Found gender column '{gender_column_found}' in card '{card_name}'")
                    aggregated = count_by_column(card_data, gender_column_found)
                    
                    if aggregated["data"]:
                        aggregated_card = {
                            "name": f"{card_name} - Aggregated by {gender_column_found.title()}",
                            "description": f"Patient count breakdown by {gender_column_found}",
                            "data": aggregated["data"],
                            "aggregated": True,
                            "original_card": card_name,
                            "debug_info": aggregated.get("debug_info", {})
                        }
                        processed_cards.append(aggregated_card)
            
            # Keep original card as well
            processed_cards.append(card)
        
        logger.info(f"Preprocessing complete. Original cards: {len(dashboard_context.get('cards', []))}, Processed cards: {len(processed_cards)}")
        processed_context["cards"] = processed_cards
        return processed_context

    def answer_question(self, question: str, dashboard_context: Dict, chat_history: List[Dict], date_value: Optional[str] = None) -> str:
        logger.debug(f"Answering question: '{question}' for dashboard: '{dashboard_context.get('dashboard_name')}'")

        # Check if we have valid data
        if not dashboard_context or not dashboard_context.get("cards"):
            logger.warning("No dashboard context or cards available")
            return json.dumps({
                "response_parts": [{
                    "type": "text",
                    "content": "I couldn't find any data to analyze. Please check if the selected cards contain data."
                }]
            })

        # Log the data we're working with
        logger.info(f"Dashboard context has {len(dashboard_context.get('cards', []))} cards")
        for card in dashboard_context.get("cards", []):
            logger.debug(f"Card: {card.get('name')} - Data rows: {len(card.get('data', []))}")

        # Pre-process for aggregation requests
        processed_dashboard_context = self._preprocess_for_aggregation(dashboard_context, question)
        logger.info(f"After aggregation preprocessing: {len(processed_dashboard_context.get('cards', []))} cards")

        example_ai_response = {
            "response_parts": [
                {
                    "type": "text",
                    "content": "Here is a breakdown of the patient demographics based on the available data."
                },
                {
                    "type": "chart",
                    "spec": {
                        "chart_type": "pie",
                        "data": [
                            {"Gender": "Female", "Count": 188728},
                            {"Gender": "Child", "Count": 234253},
                            {"Gender": "Male", "Count": 153954}
                        ],
                        "title": "Patient Demographics",
                        "labels_column": "Gender",
                        "values_column": "Count"
                    }
                }
            ],
            "suggested_questions": [
                "What are the total patient counts by MHU?",
                "Show me the trend of male patients over time.",
                "Which MHU has the highest number of patients?",
                "Compare patient counts between different locations."
            ]
        }
        example_ai_response_json = json.dumps(example_ai_response, indent=2)

        # Create a copy of dashboard_context for logging and truncate data for brevity
        llm_dashboard_context = json.loads(json.dumps(processed_dashboard_context)) # Deep copy
        
        # Limit data rows to prevent token overflow but keep enough for analysis
        for card in llm_dashboard_context.get("cards", []):
            if "data" in card and isinstance(card["data"], list):
                if len(card["data"]) > 50:  # Keep more rows for better analysis
                    card["data"] = card["data"][:50]
                    card["data"].append({"note": f"... and {len(card['data'])} more rows"})

        logger.debug(f"Dashboard context prepared for LLM with {len(llm_dashboard_context.get('cards', []))} cards")

        # Limit chat history to the last few turns to save tokens
        limited_chat_history = chat_history[-4:] # Keep last 4 messages

        # --- Enhanced, Data-Driven Prompt ---
        prompt = """
System: You are a highly skilled data analyst AI for a healthcare organization, helping users understand and explore Metabase dashboards. Use all provided context and conversation history to answer the user's question as accurately and helpfully as possible.

IMPORTANT: You MUST respond with a COMPLETE JSON object. Do not truncate or leave incomplete responses.
CRITICAL: Only use the actual data provided in the dashboard context. Do NOT make up numbers, totals, or calculations that are not explicitly present in the data.

---

**Dashboard Context (filtered for the user's request):**
```json
{}
```

**Date Filter:**
{}

**Conversation History:**
```json
{}
```

**User's Latest Question:**
{}

---

**Instructions:**
- Carefully analyze the user's question and the conversation history to understand intent, context, and any follow-up references (e.g., "next", "continue").
- If a date filter is provided, only use data within that range. The data is already filtered for you.
- Extract the most relevant numbers, trends, and breakdowns from the dashboard context to answer the question.
- For aggregation requests (e.g., "by MHU", "count by", "group by"), analyze the data structure and provide aggregated results.
- Respond with a COMPLETE JSON object containing a `response_parts` array. Each part can be:
    - `{{"type": "text", "content": "..."}}` - for explanations, summaries, or single numbers
    - `{{"type": "table", "data": [ ... ]}}` - for tabular data
    - `{{"type": "chart", "spec": {{ ... }}}}` - for visualizations
- Prefer charts or tables for visualizable data. Use text for summaries, explanations, or when the answer is a single number.
- Always provide clear, concise chart/table titles and labels. Avoid cluttered visuals.
- If a card's data contains a `total_count` or similar pre-calculated value, use it directly.
- If the user asks for "next", "continue", or similar, use the conversation history to determine what to show next (e.g., next page, next metric, next chart).
- If appropriate, include a `suggested_questions` array with relevant follow-up questions.
- ALWAYS provide a complete response with at least one text part explaining the findings.

**Data Analysis Guidelines:**
- For "count by" or "group by" requests, analyze the data structure and provide aggregated counts
- For "total by" requests, sum numeric values grouped by the specified column
- For patient data, look for columns like 'mhu', 'location', 'gender', etc. for grouping
- Always include both the raw data and any calculated aggregations in your response
- ONLY use numbers that are actually present in the provided data
- Do NOT create totals by adding up numbers from different cards unless they are clearly related
- Be explicit about what data you are using and what calculations you are performing

**Suggested Questions Guidelines:**
- ALWAYS include a `suggested_questions` array with 3-4 relevant follow-up questions
- Questions should be related to the current data and analysis
- Include questions about different aspects: totals, trends, comparisons, specific groups
- Make questions specific and actionable based on the available data
- Examples: "Show me X by Y", "Compare A and B", "Which has the highest/lowest", "Trends over time"

**Example response structure:**
```json
{}
```

---

Return ONLY the COMPLETE JSON response. Do not include explanations or extra text outside the JSON. Ensure the JSON is properly formatted and complete. Use ONLY the actual data provided.
""".format(
            json.dumps(llm_dashboard_context, indent=2),
            date_value if date_value else "No specific date filter applied.",
            json.dumps(limited_chat_history, indent=2),
            question,
            example_ai_response_json.replace('{', '{{').replace('}', '}}')
        )
        
        # Get response from LLM
        raw_response = self._call_llm(prompt)
        logger.debug(f"Raw LLM response: {raw_response}")
        
        # Try to parse the response
        try:
            # Clean the response
            clean_response = raw_response.strip()
            
            # Remove markdown code blocks if present
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            
            clean_response = clean_response.strip()
            
            # Parse JSON
            response_json = json.loads(clean_response)
            
            # Validate response structure
            if not isinstance(response_json, dict):
                raise ValueError("Response is not a dictionary")
            
            if "response_parts" not in response_json:
                raise ValueError("Response missing 'response_parts'")
            
            if not response_json["response_parts"]:
                raise ValueError("Response has empty 'response_parts'")
            
            logger.info(f"Successfully parsed response with {len(response_json['response_parts'])} parts")
            return json.dumps(response_json, indent=2)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Failed to parse response: {raw_response}")
            
            # Create a fallback response
            fallback_response = {
                "response_parts": [
                    {
                        "type": "text",
                        "content": f"I analyzed the data but encountered an issue with the response format. Here's what I found: {raw_response[:200]}..."
                    }
                ]
            }
            return json.dumps(fallback_response, indent=2)
            
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            
            # Create a fallback response
            fallback_response = {
                "response_parts": [
                    {
                        "type": "text",
                        "content": "I encountered an error while processing the response. Please try rephrasing your question."
                    }
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

    def _is_total_count_query(self, question: str) -> bool:
        """Checks if the question is asking for a total count."""
        question_lower = question.lower()
        keywords = ["total count", "how many", "number of", "total number"]
        return any(keyword in question_lower for keyword in keywords)

    def identify_relevant_cards(self, question: str, all_cards: list) -> list:
        """
        Enhanced card selection using multiple strategies:
        1. Enhanced multi-strategy approach (domain rules, TF-IDF, fuzzy matching, synonyms)
        2. OpenAI Embeddings (if available)
        3. Fallback to hybrid TF-IDF/keyword/LLM logic
        Returns a list of card names.
        """
        from analytics_utils import find_relevant_cards_enhanced, find_relevant_cards_openai_embed, find_relevant_cards_sklearn
        import os
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Starting card selection for question: '{question}'")
        logger.info(f"Total cards available: {len(all_cards)}")
        
        # Strategy 1: Enhanced multi-strategy approach
        try:
            enhanced_cards = find_relevant_cards_enhanced(question, all_cards, top_n=5)
            if enhanced_cards:
                logger.info(f"Enhanced strategy found cards: {enhanced_cards}")
                return enhanced_cards
        except Exception as e:
            logger.error(f"Enhanced card selection failed: {e}")
        
        # Strategy 2: OpenAI Embeddings (if available)
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                openai_cards = find_relevant_cards_openai_embed(question, all_cards, top_n=5)
                if openai_cards:
                    logger.info(f"OpenAI Embeddings found cards: {openai_cards}")
                    return openai_cards
        except Exception as e:
            logger.error(f"OpenAI Embeddings failed: {e}")
        
        # Strategy 3: Fallback to hybrid TF-IDF/keyword/LLM logic
        logger.info("Falling back to hybrid TF-IDF/keyword/LLM logic")
        try:
            ps = PorterStemmer()
            sklearn_cards = find_relevant_cards_sklearn(question, all_cards, top_n=8)
            logger.info(f"TF-IDF relevant cards: {sklearn_cards}")
            
            # Keyword overlap with stemming and weighted scoring
            question_keywords = set(ps.stem(w) for w in re.findall(r'\w+', question.lower()))
            card_scores = Counter()
            card_names = {}
            
            for c in all_cards:
                card_data = c.get("card", {})
                if c.get("card_id"):
                    name = card_data.get("name") or ""
                    desc = card_data.get("description") or ""
                    schema = " ".join(col["name"] for col in card_data.get("schema", []))
                    
                    # Stem all words
                    name_stems = set(ps.stem(w) for w in re.findall(r'\w+', name.lower()))
                    desc_stems = set(ps.stem(w) for w in re.findall(r'\w+', desc.lower()))
                    schema_stems = set(ps.stem(w) for w in re.findall(r'\w+', schema.lower()))
                    
                    # Weighted overlap: name=3, desc=2, schema=1
                    overlap = (
                        3 * len(question_keywords & name_stems) +
                        2 * len(question_keywords & desc_stems) +
                        1 * len(question_keywords & schema_stems)
                    )
                    if overlap > 0:
                        card_scores[name] = overlap
                    card_names[name] = (name, desc, schema)
            
            top_keyword_cards = [name for name, _ in card_scores.most_common(8)]
            logger.info(f"Keyword overlap relevant cards: {top_keyword_cards}")
            
            # Combine: only keep cards that appear in both TF-IDF and keyword overlap
            combined_cards = [name for name in sklearn_cards if name in top_keyword_cards]
            logger.info(f"Combined TF-IDF & keyword relevant cards: {combined_cards}")
            if combined_cards:
                return combined_cards[:5]
            
            # If no intersection, fallback to union of both (up to 5)
            union_cards = list(dict.fromkeys(sklearn_cards + top_keyword_cards))[:5]
            if union_cards:
                logger.info(f"Union fallback relevant cards: {union_cards}")
                return union_cards
            
            # Final fallback: LLM selection from shortlist
            card_info = []
            shortlist_names = list(dict.fromkeys(sklearn_cards + top_keyword_cards))[:8]
            for c in all_cards:
                card_data = c.get("card", {})
                if c.get("card_id") and card_data.get("name") in shortlist_names:
                    simplified_schema = [col["name"] for col in card_data.get("schema", [])]
                    info = {
                        "name": card_data.get("name"),
                        "description": card_data.get("description"),
                        "columns": simplified_schema
                    }
                    card_info.append(info)
            
            logger.info(f"Using LLM for final card selection from shortlist: {shortlist_names}")
            prompt = f"""
System: You are an intelligent routing assistant for a Metabase dashboard. Your task is to identify the most relevant data cards to answer a user's question. Consider the card's name, description, and especially its schema (column names and types) for relevance. If a question involves dates (e.g., 'yesterday', 'last week', 'monthly trend'), and a card's description or schema mentions dates or timestamps, that card is likely relevant.

You are given a shortlist of candidate cards. Respond with ONLY a JSON array of the card names that are directly relevant. For example: ["Card Name 1", "Card Name 2"].
If multiple cards are relevant, list them all, but try to limit to the top 5 most relevant cards. If the user's question is general (e.g., 'summarize this dashboard', 'tell me about this dashboard', 'show me the key metrics', 'suggest me questions') or involves any form of date/time analysis (e.g., 'yesterday', 'today', 'last week', 'monthly trend', 'over time', 'by date'), return ALL available card names, but still try to limit to the top 5 most relevant cards.

Shortlisted Cards:
```json
{json.dumps(card_info, indent=2)}
```

User's Question: "{question}"

Your JSON Response:
"""
            response_str = self._call_llm(prompt)
            logger.info(f"LLM response for card identification: {response_str}")
            try:
                clean_str = response_str.strip().replace("```json", "").replace("```", "")
                card_names = json.loads(clean_str)
                if isinstance(card_names, list):
                    return card_names
                return []
            except json.JSONDecodeError:
                logger.error(f"Failed to decode card identification response: {response_str}")
                return []
                
        except Exception as e:
            logger.error(f"Hybrid fallback logic failed: {e}")
            return []
        
        # Ultimate fallback: return first 5 cards if everything else fails
        logger.warning("All card selection strategies failed, returning first 5 cards")
        fallback_cards = []
        for c in all_cards:
            if c.get("card_id") and len(fallback_cards) < 5:
                fallback_cards.append(c.get("card", {}).get("name"))
        return fallback_cards