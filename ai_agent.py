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

    def answer_question(self, question: str, dashboard_context: Dict, chat_history: List[Dict], date_value: Optional[str] = None) -> str:
        logger.debug(f"Answering question: '{question}' for dashboard: '{dashboard_context.get('dashboard_name')}'")

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
                "Show me the trend of male patients over time."
            ]
        }
        example_ai_response_json = json.dumps(example_ai_response, indent=2)

        # Create a copy of dashboard_context for logging and truncate data for brevity
        # For the LLM prompt, we'll truncate data even more aggressively to save tokens.
        llm_dashboard_context = json.loads(json.dumps(dashboard_context)) # Deep copy
        # Remove or relax truncation: send all data rows
        # for card in llm_dashboard_context.get("cards", []):
        #     if "data" in card and isinstance(card["data"], list):
        #         card["data"] = card["data"][:1] # Limit to 1 row for the LLM
        #         if len(card["data"]) > 1:
        #             card["data"].append("... (truncated)")
        logger.debug("Dashboard context sent to LLM (truncated for logs): %s" % json.dumps(llm_dashboard_context, indent=2))

        # Limit chat history to the last few turns to save tokens
        # Assuming each message in chat_history is a dict with 'role' and 'content'
        # You might need to adjust this number based on typical conversation length
        limited_chat_history = chat_history[-4:] # Keep last 4 messages

        # --- Advanced, Data-Driven Prompt ---
        prompt = """
System: You are a highly skilled data analyst AI for a healthcare organization, helping users understand and explore Metabase dashboards. Use all provided context and conversation history to answer the user's question as accurately and helpfully as possible.

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
- Respond with a JSON object containing a `response_parts` array. Each part can be:
    - `{{"type": "text", "content": "..."}}`
    - `{{"type": "table", "data": [ ... ]}}`
    - `{{"type": "chart", "spec": {{ ... }}}}`
- Prefer charts or tables for visualizable data. Use text for summaries, explanations, or when the answer is a single number.
- Always provide clear, concise chart/table titles and labels. Avoid cluttered visuals.
- If a card's data contains a `total_count` or similar pre-calculated value, use it directly.
- If the user asks for "next", "continue", or similar, use the conversation history to determine what to show next (e.g., next page, next metric, next chart).
- If appropriate, include a `suggested_questions` array with relevant follow-up questions.
- Example response:
```json
{}
```

---

Return ONLY the JSON response. Do not include explanations or extra text outside the JSON.
""".format(
            json.dumps(llm_dashboard_context, indent=2),
            date_value if date_value else "No specific date filter applied.",
            json.dumps(limited_chat_history, indent=2),
            question,
            example_ai_response_json.replace('{', '{{').replace('}', '}}')
        )
        return self._call_llm(prompt)

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

    def identify_relevant_cards(self, question: str, all_cards: List[Dict]) -> List[str]:
        """
        Use sklearn TF-IDF similarity, then keyword overlap, then LLM to find relevant cards. Returns a list of card names.
        """
        from analytics_utils import find_relevant_cards_sklearn
        import re
        from collections import Counter
        # 1. Try sklearn TF-IDF similarity
        sklearn_cards = find_relevant_cards_sklearn(question, all_cards, top_n=5)
        if sklearn_cards:
            logger.info(f"sklearn TF-IDF found relevant cards: {sklearn_cards}")
            return sklearn_cards
        # 2. Fallback: keyword overlap
        question_keywords = set(re.findall(r'\w+', question.lower()))
        card_scores = Counter()
        card_names = {}
        for c in all_cards:
            card_data = c.get("card", {})
            if c.get("card_id"):
                text = (card_data.get("name") or "") + " " + (card_data.get("description") or "")
                text += " " + " ".join(col["name"] for col in card_data.get("schema", []))
                text = text.lower()
                card_names[card_data.get("name")] = text
                overlap = sum(1 for word in question_keywords if word in text)
                if overlap > 0:
                    card_scores[card_data.get("name")] = overlap
        if card_scores:
            top_cards = [name for name, _ in card_scores.most_common(5)]
            logger.info(f"Keyword matching found relevant cards: {top_cards}")
            return top_cards
        # 3. Fallback: LLM
        card_info = []
        for c in all_cards:
            card_data = c.get("card", {})
            if c.get("card_id"):
                simplified_schema = [col["name"] for col in card_data.get("schema", [])]
                info = {
                    "name": card_data.get("name"),
                    "description": card_data.get("description"),
                    "columns": simplified_schema
                }
                card_info.append(info)
        logger.info(f"No sklearn or keyword matches, using LLM for card selection.")
        prompt = f"""
System: You are an intelligent routing assistant for a Metabase dashboard. Your task is to identify the most relevant data cards to answer a user's question. Consider the card's name, description, and especially its schema (column names and types) for relevance. If a question involves dates (e.g., \"yesterday\", \"last week\", \"monthly trend\"), and a card's description or schema mentions dates or timestamps, that card is likely relevant.
Respond with ONLY a JSON array of the card names that are directly relevant. For example: [\"Card Name 1\", \"Card Name 2\"].
If multiple cards are relevant, list them all, but try to limit to the top 5 most relevant cards. If the user's question is general (e.g., \"summarize this dashboard\", \"tell me about this dashboard\", \"show me the key metrics\", \"suggest me questions\") or involves any form of date/time analysis (e.g., \"yesterday\", \"today\", \"last week\", \"monthly trend\", \"over time\", \"by date\"), return ALL available card names, but still try to limit to the top 5 most relevant cards.

Available Cards:
```json
{json.dumps(card_info, indent=2)}
```

User's Question: \"{question}\"

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