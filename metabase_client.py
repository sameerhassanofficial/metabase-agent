import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
import streamlit as st



logger = logging.getLogger(__name__)

@dataclass
class MetabaseConfig:
    """Configuration for Metabase connection."""
    base_url: str
    username: str
    password: str
    session_token: Optional[str] = None
    cache_timeout: int = 300  # 5 minutes cache timeout



class MetabaseClient:
    """Enhanced Metabase API client with robust error handling and retries."""
    def __init__(self, config: MetabaseConfig):
        self.config = config
        self.session = requests.Session()
        retry_strategy = Retry(
            total=2,  # Reduced from 3 for faster failures
            backoff_factor=0.3,  # Reduced backoff factor
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS"]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy, 
            pool_connections=20,  # Increased for parallel requests
            pool_maxsize=30,      # Increased for parallel requests
            pool_block=False      # Don't block when pool is full
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.timeout = 15  # Reduced timeout for faster failures

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

    def get_card_definition(self, card_id: int) -> Dict:
        """Retrieves the full definition of a specific card, including its result_metadata."""
        url = f"{self.config.base_url}/api/card/{card_id}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            card_definition = response.json()
            return card_definition
        except requests.exceptions.RequestException as e:
            logger.error(f"API GET call to {url} failed: {e}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {url}")
        return {}

    def get_card_data(self, card_id: int, parameters: Optional[list] = None, dashboard_id: Optional[int] = None) -> Dict[str, Any]:
        """Retrieves the actual data for a specific card, optionally within a dashboard context."""
        url = f"{self.config.base_url}/api/card/{card_id}/query"
        payload = {}
        if parameters:
            payload["parameters"] = parameters
        if dashboard_id is not None:
            payload["dashboard_id"] = dashboard_id
        try:
            logger.info(f"Fetching card data for ID: {card_id} (dashboard_id={dashboard_id})")
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            card_data = response.json()
            return card_data
        except requests.exceptions.RequestException as e:
            logger.error(f"API POST call to {url} failed: {e}")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {url}")
            return {}

    def get_dashboard_card_data(self, dashboard_id: int, dashcard_id: int, card_id: int, parameters: Optional[list] = None) -> Dict[str, Any]:
        """Retrieves data for a card in the context of a dashboard/dashcard, supporting dashboard filters."""
        url = f"{self.config.base_url}/api/dashboard/{dashboard_id}/dashcard/{dashcard_id}/card/{card_id}/query"
        payload = {}
        if parameters:
            payload["parameters"] = parameters
        payload["dashboard_id"] = dashboard_id
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API POST call to {url} failed: {e}")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {url}")
            return {}

@dataclass
class CacheEntry:
    """Cache entry with timestamp for TTL."""
    data: Any
    timestamp: float = field(default_factory=time.time)
    
    def is_expired(self, timeout: int) -> bool:
        return time.time() - self.timestamp > timeout

class MetabaseMetadataProvider:
    """Provides Metabase metadata with intelligent caching."""
    def __init__(self, session: requests.Session, config: MetabaseConfig):
        self.session = session
        self.config = config
        self._cache: Dict[str, CacheEntry] = {}

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid and not expired."""
        if cache_key not in self._cache:
            return False
        return not self._cache[cache_key].is_expired(self.config.cache_timeout)

    def get_dashboards(self) -> List[Dict]:
        logger.info("Fetching fresh data for 'dashboards'.")
        url = f"{self.config.base_url}/api/dashboard"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API GET call to {url} failed: {e}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {url}")
        return []

    def get_dashboard_details(self, dashboard_id: int) -> Dict:
        """Retrieves details for a specific dashboard, including card schemas."""
        cache_key = f"dashboard_{dashboard_id}"
        if self._is_cache_valid(cache_key):
            logger.info(f"Using cached dashboard details for ID: {dashboard_id}")
            return self._cache[cache_key].data

        url = f"{self.config.base_url}/api/dashboard/{dashboard_id}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            dashboard_details = response.json()

            # Simplified card schema fetching - no concurrency to avoid duplicate calls
            dashcards_with_cards = [dc for dc in dashboard_details.get("dashcards", []) if dc.get("card_id")]
            
            if dashcards_with_cards:
                for dashcard in dashcards_with_cards:
                    try:
                        card_definition = self.get_card_definition(dashcard.get("card_id"))
                        if card_definition and "result_metadata" in card_definition:
                            schema = [
                                {
                                    "name": col["name"], 
                                    "display_name": col.get("display_name", col["name"]), 
                                    "base_type": col["base_type"],
                                    "description": col.get("description", "")
                                } 
                                for col in card_definition["result_metadata"]
                            ]
                            dashcard["card"]["schema"] = schema
                        else:
                            logger.warning(f"Could not retrieve schema for card {dashcard.get('card_id')}")
                            dashcard["card"]["schema"] = []
                    except Exception as exc:
                        logger.error(f"Error fetching card definition: {exc}")
                        dashcard["card"]["schema"] = []

            self._cache[cache_key] = CacheEntry(dashboard_details)
            return dashboard_details
        except requests.exceptions.RequestException as e:
            logger.error(f"API GET call to {url} failed: {e}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {url}")
        return {}

    def get_card_definition(self, card_id: int) -> Dict:
        """Retrieves the full definition of a specific card, including its result_metadata."""
        cache_key = f"card_{card_id}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key].data

        url = f"{self.config.base_url}/api/card/{card_id}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            card_definition = response.json()
            self._cache[cache_key] = CacheEntry(card_definition)
            return card_definition
        except requests.exceptions.RequestException as e:
            logger.error(f"API GET call to {url} failed: {e}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {url}")
        return {}
