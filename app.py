"""
Mobile Health Units Assistant (Advanced UI)
An advanced, conversational interface for healthcare organizations to explore
Metabase dashboards and gain data-driven insights with AI-generated charts.
"""

import os
import json
import logging
import time

import streamlit as st
from dotenv import load_dotenv

from metabase_client import MetabaseConfig, MetabaseClient, MetabaseMetadataProvider
from ai_agent import MetabaseLLMAgent
from ui_utils import render_chat_response

# --- Initial Configuration ---
load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('watchdog').setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Streamlit Application UI ---
@st.cache_resource
def init_metabase_client():
    """Initializes the Metabase client, handling authentication."""
    config = MetabaseConfig(
        base_url=os.getenv('METABASE_URL').rstrip('/'),
        username=os.getenv('METABASE_USERNAME'),
        password=os.getenv('METABASE_PASSWORD')
    )
    client = MetabaseClient(config)
    if not client.authenticate():
        st.error("Metabase authentication failed. Please check your credentials and connection.")
        return # Use return instead of st.stop()
    return client

def main():
    """Main function to run the Streamlit chat application."""
    st.set_page_config(page_title="Mobile Health Units Assistant", layout="wide")

    st.markdown("<h1 style='text-align: center;'>🧠 AI-Powered MHU Insights</h1>", unsafe_allow_html=True)
    

    # --- Environment Variable Check ---
    required_vars = ['METABASE_URL', 'METABASE_USERNAME', 'METABASE_PASSWORD', 'GEMINI_API_KEY', 'OPENAI_API_KEY']
    if any(not os.getenv(var) for var in required_vars):
        st.error("Error: Required environment variables are missing. Please check your `.env` file.")
        return # Use return instead of st.stop()

    # --- Initialization ---
    client = init_metabase_client()
    provider = MetabaseMetadataProvider(client.session, client.config)
    agent = MetabaseLLMAgent()

    # --- Sidebar for Dashboard Selection ---
    st.sidebar.header("Dashboard")
    dashboards = provider.get_dashboards()
    if not dashboards:
        st.sidebar.warning("No dashboards found.")
        return # Use return instead of st.stop()

    selected_dashboard_id = 8 # Hardcoded for debugging
    dashboard_info = provider.get_dashboard_details(selected_dashboard_id)
    selected_dashboard_name = dashboard_info.get("name", "Hardcoded Dashboard 8")
    st.sidebar.write(f"{selected_dashboard_name.replace(' - Dashboard', '')}")

    # Display a list of cards (questions) in the dashboard
    st.sidebar.subheader("Available Cards")
    dashcards_list = []
    for dc in dashboard_info.get("dashcards", []):
        if dc.get("card_id"): # Only consider actual cards, not just dashboard elements
            dashcards_list.append(dc.get("card", {}).get("name"))

    if dashcards_list:
        for card_name in dashcards_list:
            st.sidebar.markdown(f"- {card_name}")
    else:
        st.sidebar.info("No data cards found in this dashboard.")

    

    

    

    

    # --- Main Chat UI ---
    

    # Initialize chat history
    if "messages" not in st.session_state or st.session_state.get("current_dashboard_id") != selected_dashboard_id:
        st.session_state.messages = []
        st.session_state.current_dashboard_id = selected_dashboard_id
        st.session_state.selected_date = None # Initialize selected_date

    # Display chat messages
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            render_chat_response(message["content"])

    # Display suggested prompts as clickable links within the chat
    if st.session_state.get("suggested_prompts"):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("Here are some questions you might want to ask:")
            for prompt_text in st.session_state.suggested_prompts:
                st.markdown(f"- [{prompt_text}](/?prompt={prompt_text})")
        st.session_state.suggested_prompts = [] # Clear prompts after displaying

    # Get user input
    if prompt := st.chat_input("Ask a question...") or st.session_state.get("prompt"):
        st.session_state.prompt = None # Clear state
        st.session_state.messages.append({"role": "user", "content": {"response_parts": [{"type": "text", "content": prompt}]}})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            progress_bar = st.progress(0.0, text="Analyzing dashboard structure...")
            status_text = st.empty()
            dashboard_context = None
            
            try:
                # Step 1: Get all card names and descriptions for the AI to analyze
                all_dashcards = [dc for dc in provider.get_dashboard_details(selected_dashboard_id).get("dashcards", []) if dc.get("card_id")]
                
                status_text.text("AI is identifying relevant cards...")
                relevant_card_names = agent.identify_relevant_cards(prompt, all_dashcards)
                
                response_json = None # Initialize response_json at the beginning of the try block

                if not relevant_card_names:
                    response_json = {"response_parts": [{"type": "text", "content": "I couldn't identify any relevant cards for your question. Please try rephrasing or ask a question related to the available Metabase cards."}]}
                
                if relevant_card_names and response_json is None: # Only proceed if relevant cards were found/assigned and no fallback message yet
                    # Step 3: Fetch data only for relevant cards
                    date_filters = agent.parse_date_filters(prompt, st.session_state.selected_date)
                    date_value = date_filters.get("date_value")

                    # Pass the specific dashcard to the generator
                    for update in provider.get_dashboard_context_generator(selected_dashboard_id, relevant_card_names, date_value=date_value):
                        if update["status"] == "progress":
                            progress_bar.progress(update["progress"], text=update["message"])
                        elif update["status"] == "complete":
                            dashboard_context = update["context"]
                            progress_bar.progress(1.0, text="All relevant data loaded!")
                            time.sleep(1)
                            break
                        elif update["status"] == "error":
                            st.error(update["message"])
                            return # Use return instead of st.stop()
                    
                    progress_bar.empty()
                    status_text.empty()

                    if not dashboard_context or not dashboard_context.get("cards"):
                        response_json = {"response_parts": [{"type": "text", "content": "I found relevant cards, but couldn't retrieve data from them. They might be empty or have issues."}]}
                    else:
                        with st.spinner("🤖 AI is analyzing the data..."):
                            full_response_str = agent.answer_question(prompt, dashboard_context, st.session_state.messages, date_value)
                            try:
                                clean_str = full_response_str.strip().replace("```json", "").replace("```", "")
                                response_json = json.loads(clean_str)
                            except json.JSONDecodeError:
                                logger.error(f"Failed to decode AI response JSON: {full_response_str}")
                                response_json = {"response_parts": [{"type": "text", "content": "Sorry, I received an invalid response from the AI."}]}
                
                # Ensure response_json is set before rendering and appending to messages
                if response_json is not None:
                    render_chat_response(response_json)
                    st.session_state.messages.append({"role": "assistant", "content": response_json})

            except Exception as e:
                logger.error(f"An error occurred: {e}", exc_info=True)
                st.error(f"An unexpected error occurred: {e}")

    # Auto-scroll to the bottom of the chat
    st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

    # "New Chat" / Reset Button
    if st.button("New Chat"):
        st.session_state.messages = []
        st.session_state.suggested_prompts = []
        st.session_state.current_dashboard_id = None
        st.rerun()

if __name__ == "__main__":
    main()
