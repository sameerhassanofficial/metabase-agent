import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
import streamlit as st
import re
import uuid

logger = logging.getLogger(__name__)

@dataclass
class MetabaseConfig:
    """Configuration for Metabase connection."""
    base_url: str
    username: str
    password: str
    session_token: Optional[str] = None

class MetabaseClient:
    """Enhanced Metabase API client with robust error handling and retries."""
    def __init__(self, config: MetabaseConfig):
        self.config = config
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.timeout = 60

    def authenticate(self) -> bool:
        """Authenticates with Metabase and stores the session token."""
        auth_url = f"{self.config.base_url}/api/session"
        auth_data = {"username": self.config.username, "password": self.config.password}
        try:
            logger.info(f"Authenticating with Metabase at {self.config.base_url}")
            response = self.session.post(auth_url, json=auth_data)
            response.raise_for_status()
            session_data = response.json()
            self.config.session_token = session_data.get('id')
            if self.config.session_token:
                self.session.headers.update({'X-Metabase-Session': self.config.session_token})
                logger.info("Authentication successful.")
                return True
            logger.error("Authentication failed: No session token in response.")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication error: {e}")
            return False

    def api_get_call(self, endpoint: str) -> Union[List[Dict], Dict, None]:
        url = f"{self.config.base_url}/api/{endpoint}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API GET call to {url} failed: {e}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {url}")
        return None



class MetabaseMetadataProvider:
    """Provides and caches Metabase metadata using Streamlit's caching."""
    def __init__(self, session: requests.Session, config: MetabaseConfig):
        self.session = session
        self.config = config

    @st.cache_data(ttl=600)
    def get_dashboards(_self) -> List[Dict]:
        logger.info("Fetching fresh data for 'dashboards'.")
        url = f"{_self.config.base_url}/api/dashboard"
        try:
            response = _self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API GET call to {url} failed: {e}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {url}")
        return []

    @st.cache_data(ttl=300)
    def get_dashboard_details(_self, dashboard_id: int) -> Dict:
        """Retrieves and caches details for a specific dashboard, including card schemas."""
        logger.info(f"Fetching details for dashboard ID: {dashboard_id}.")
        url = f"{_self.config.base_url}/api/dashboard/{dashboard_id}"
        try:
            response = _self.session.get(url)
            response.raise_for_status()
            dashboard_details = response.json()

            # Fetch schema for each card
            for dashcard in dashboard_details.get("dashcards", []):
                card_id = dashcard.get("card_id")
                if card_id:
                    card_definition = _self.get_card_definition(card_id)
                    if card_definition and "result_metadata" in card_definition:
                        dashcard["card"]["schema"] = [{"name": col["name"], "base_type": col["base_type"]} for col in card_definition["result_metadata"]]
                    else:
                        logger.warning(f"Could not retrieve schema for card {card_id}")
            return dashboard_details
        except requests.exceptions.RequestException as e:
            logger.error(f"API GET call to {url} failed: {e}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {url}")
        return {}

    def _execute_dashboard_card_query(_self, url: str, session_token: Optional[str], parameters: Optional[List[Dict]] = None) -> Optional[Dict]:
        headers = {'X-Metabase-Session': session_token} if session_token else {}
        payload = {"parameters": parameters} if parameters is not None else {}
        logger.log(0, f"Sending payload to Metabase API: {json.dumps(payload, indent=2)} to URL: {url}")
        logger.log(0, f"Headers: {headers}")
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            query_result = response.json()
            logger.debug(f"Metabase query result for {url}: {json.dumps(query_result, indent=2)}")
            return query_result
        except requests.exceptions.RequestException as e:
            logger.error(f"API POST call to {url} failed: {e}")
            if e.response is not None:
                logger.error(f"Response status code: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {url}")
        return None

    @st.cache_data(ttl=3600) # Cache card definitions for an hour
    def get_card_definition(_self, card_id: int) -> Dict:
        """Retrieves and caches the full definition of a specific card, including its result_metadata."""
        logger.info(f"Fetching definition for card ID: {card_id}.")
        url = f"{_self.config.base_url}/api/card/{card_id}"
        try:
            response = _self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API GET call to {url} failed: {e}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {url}")
        return {}

    def get_dashboard_context_generator(_self, dashboard_id: int, relevant_cards: List[str], date_value: Optional[str] = None, age_value: Optional[int] = None, mr_no_value: Optional[str] = None) -> Dict:
        """Generator that yields dashboard context with progress updates."""
        logger.debug(f"Fetching context for dashboard ID: {dashboard_id}.")
        details = _self.get_dashboard_details(dashboard_id)
        if not details:
            yield {"status": "error", "message": "Could not retrieve dashboard details."}
            return

        context = {
            "dashboard_name": details.get("name"),
            "dashboard_description": details.get("description"),
            "cards": []
        }
        
        dashcards = [dc for dc in details.get("dashcards", []) if dc.get("card", {}).get("name") in relevant_cards]

        total_cards = len(dashcards)

        # Use ThreadPoolExecutor for concurrent fetching
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_card = {}
            for dash_card in dashcards:
                card_name = "" # Defensive initialization
                card_id = dash_card.get("card_id")
                dashcard_id = dash_card.get("id")
                card_details = dash_card.get("card", {})
                display_card_id = card_id if card_id is not None else "Unknown"
                card_name = card_details.get("name", f"Card {display_card_id}")
                
                # Construct the dashboard-specific query URL
                url = f"{_self.config.base_url}/api/dashboard/{dashboard_id}/dashcard/{dashcard_id}/card/{card_id}/query"
                
                parameters = []
                if date_value:
                    parameters.append({
                        "type": "date/all-options",
                        "value": date_value,
                        "id": "7f30ced2", # Hardcoded ID from your working curl example
                        "target": ["dimension", ["field", "date", {"base-type": "type/DateTime"}]] # Hardcoded target from your working curl example
                    })

                # Execute the query directly here
                future_to_card[executor.submit(_self._execute_dashboard_card_query, url, _self.config.session_token, parameters)] = dash_card
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_card)):
                dash_card = future_to_card[future]
                card_id = dash_card.get("card_id")
                card_details = dash_card.get("card", {})
                card_name = card_details.get("name", f"Card {card_id}")

                yield {"status": "progress", "message": f"Fetching data for card: {card_name}", "progress": (i + 1) / total_cards}
                
                try:
                    query_result = future.result()
                    card_context = {
                        "name": card_name,
                        "description": card_details.get("description"),
                        "data": []
                    }
                    if query_result and query_result.get('status') == 'completed':
                        data = query_result.get('data', {})
                        if 'rows' in data and 'cols' in data:
                            col_names = [col.get('display_name', col.get('name')) for col in data['cols']]
                            card_context["data"] = [dict(zip(col_names, row)) for row in data['rows']]  # Remove [:30] limit
                            # Add _filtered_by to each row if date_value is set
                            if date_value:
                                for row in card_context["data"]:
                                    row["_filtered_by"] = date_value
                            # Add schema information
                            card_definition = _self.get_card_definition(card_id)
                            if card_definition and "result_metadata" in card_definition:
                                card_context["schema"] = [{"name": col["name"], "base_type": col["base_type"]} for col in card_definition["result_metadata"]]
                            else:
                                logger.warning(f"Could not retrieve schema for card {card_name} (ID: {card_id})")
                        else:
                            card_context["data"] = data.get('rows', [])  # Remove [:30] limit
                            # Add _filtered_by to each row if date_value is set
                            if date_value:
                                for row in card_context["data"]:
                                    if isinstance(row, dict):
                                        row["_filtered_by"] = date_value
                            # Attempt to infer schema if result_metadata is not available
                            if card_context["data"] and isinstance(card_context["data"], list) and len(card_context["data"]) > 0:
                                first_row = card_context["data"][0]
                                card_context["schema"] = [{"name": k, "base_type": "unknown"} for k in first_row.keys()]
                            else:
                                card_context["schema"] = []
                        logger.debug(f"Successfully processed card {card_name}. Data length: {len(card_context['data'])}")
                    else:
                        card_context["error"] = f"Could not retrieve data. Status: {query_result.get('status') if query_result else 'N/A'}"
                        logger.warning(f"Failed to retrieve data for card {card_name}. Status: {query_result.get('status') if query_result else 'N/A'}")
                    context["cards"].append(card_context)
                except Exception as exc:
                    logger.error(f'{card_name} generated an exception: {exc}')
                    context["cards"].append({
                        "name": card_name,
                        "description": card_details.get("description"),
                        "error": f"Failed to load data: {exc}"
                    })

        yield {"status": "complete", "context": context}
