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

# Configure logging to suppress OpenAI debug messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('watchdog').setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
logging.getLogger('openai._base_client').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
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

    # Custom CSS for better styling with dark mode support
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white !important;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        color: rgba(255,255,255,0.9) !important;
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    .welcome-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border-left: 5px solid #667eea;
    }
    .welcome-box h3 {
        color: #ffffff !important;
        margin-top: 0;
    }
    .welcome-box p {
        color: #e0e0e0 !important;
        margin-bottom: 1rem;
    }
    .welcome-box ul {
        color: #e0e0e0 !important;
    }
    .welcome-box strong {
        color: #ffffff !important;
    }
    .chat-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stChatMessage[data-testid="chatMessage"] {
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    /* Dark mode text visibility */
    .stMarkdown, .stText {
        color: inherit !important;
    }
    /* Ensure all text is visible in dark mode */
    .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #ffffff !important;
    }
    .stMarkdown ul, .stMarkdown ol {
        color: #e0e0e0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Enhanced header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 AI-Powered MHU Data Assistant</h1>
        <p>Mobile Health Units • Intelligent Data Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    

    # --- Environment Variable Check ---
    required_vars = ['METABASE_URL', 'METABASE_USERNAME', 'METABASE_PASSWORD', 'GEMINI_API_KEY', 'OPENAI_API_KEY']
    if any(not os.getenv(var) for var in required_vars):
        st.error("Error: Required environment variables are missing. Please check your `.env` file.")
        return # Use return instead of st.stop()

    # --- Initialization ---
    client = init_metabase_client()
    provider = MetabaseMetadataProvider(client.session, client.config)
    
    # Share the same Metabase client with the agent to avoid duplicate authentication
    agent = MetabaseLLMAgent()
    agent.metabase_client = client  # Use the same authenticated client

    # --- Auto-select Dashboard ID 8 (No Sidebar) ---
    selected_dashboard_id = 8

    # --- Main Chat UI ---
    

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.selected_date = None # Initialize selected_date

    # Show suggested questions for new users (without welcome card)
    if not st.session_state.messages:
        # Suggested questions for new users
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Show me total patient count", use_container_width=True):
                st.session_state.prompt = "Show me the total patient count"
                st.rerun()
            if st.button("👥 What's the gender distribution?", use_container_width=True):
                st.session_state.prompt = "What's the gender distribution of patients?"
                st.rerun()
        with col2:
            if st.button("🏥 Which MHU has most patients?", use_container_width=True):
                st.session_state.prompt = "Which Mobile Health Unit has the most patients?"
                st.rerun()
            if st.button("📈 Show patient trends", use_container_width=True):
                st.session_state.prompt = "Show me patient trends over time"
                st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            render_chat_response(message["content"])

    # Display suggested prompts as clickable links within the chat (for historical messages)
    if st.session_state.get("suggested_prompts") and not st.session_state.get("prompt"):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("**💡 Suggested follow-up questions:**")
            
            # Use a more robust approach with session state tracking
            if st.session_state.suggested_prompts:
                # Initialize button state tracking
                button_state_key = f"hist_button_state_{len(st.session_state.messages)}"
                if button_state_key not in st.session_state:
                    st.session_state[button_state_key] = None
                
                # Create a container for the suggested questions
                with st.container():
                    for i, prompt_text in enumerate(st.session_state.suggested_prompts):
                        # Create a unique key for each button
                        button_key = f"hist_suggested_btn_{i}_{len(st.session_state.messages)}"
                        
                        # Check if this button was clicked
                        if st.button(f"❓ {prompt_text}", key=button_key, use_container_width=True):
                            # Set the prompt and clear suggestions
                            st.session_state.prompt = prompt_text
                            st.session_state.suggested_prompts = []
                            # Mark this button as clicked
                            st.session_state[button_state_key] = button_key
                            st.rerun()
            
            # Clear suggestions button
            clear_key = f"hist_clear_suggestions_{len(st.session_state.messages)}"
            if st.button("🗑️ Clear suggestions", key=clear_key):
                st.session_state.suggested_prompts = []
                st.rerun()

    # Chat input area with better styling
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Get user input
    if prompt := st.chat_input("💬 Ask me anything about your MHU data...") or st.session_state.get("prompt"):
        st.session_state.prompt = None # Clear state
        st.session_state.messages.append({"role": "user", "content": {"response_parts": [{"type": "text", "content": prompt}]}})
        with st.chat_message("user"):
            st.markdown(f"**{prompt}**")

        # Generate and display assistant response
        with st.chat_message("assistant"):
            # Simple progress indicator
            with st.status("🤖 AI is analyzing your question...", expanded=False) as status:
                try:
                    logger.info(f"🚀 STARTING PROCESS: Analyzing question: '{prompt}'")
                    
                    # Get dashboard details when needed (cached in session state)
                    if "dashboard_details" not in st.session_state:
                        dashboard_details = provider.get_dashboard_details(selected_dashboard_id)
                        if not dashboard_details:
                            st.error(f"Could not retrieve dashboard ID: {selected_dashboard_id}")
                            return
                        st.session_state.dashboard_details = dashboard_details
                    else:
                        dashboard_details = st.session_state.dashboard_details

                    # Create Simplified Card Details for the LLM
                    all_dashcards = [dc for dc in dashboard_details.get("dashcards", []) if dc.get("card_id")]
                    if not all_dashcards:
                        st.warning(f"No cards found in dashboard {selected_dashboard_id}. Please ensure the dashboard contains cards.")
                        return

                    simplified_card_details = []
                    for dashcard in all_dashcards:
                        card = dashcard.get("card", {})
                        simplified_info = {
                            "card_id": card.get("id"),
                            "card_name": card.get("name", "Unnamed Card"),
                            "card_description": card.get("description", "No description available"),
                            "result_metadata": card.get("schema", [])
                        }
                        simplified_card_details.append(simplified_info)

                    # Prepare date filters
                    date_filters = agent.parse_date_filters(prompt, st.session_state.selected_date)
                    date_value = date_filters.get("date_value")

                    full_response_str = agent.answer_question(
                        prompt, selected_dashboard_id, st.session_state.messages, date_value, simplified_card_details, dashboard_details
                    )
                    
                    status.update(label="✅ Analysis complete!", state="complete")

                    response_json = None
                    if full_response_str is None:
                        logger.error("agent.answer_question returned None.")
                        response_json = {"response_parts": [{"type": "text", "content": "An unexpected error occurred: The AI agent did not return a valid response."}]}
                    else:
                        logger.debug(f"Raw AI response: {full_response_str[:500]}...")

                    try:
                        # The response is already JSON from the improved answer_question method
                        response_json = json.loads(full_response_str)
                        logger.info(f"🎯 RESPONSE READY: {len(response_json.get('response_parts', []))} parts generated successfully")
                        # Validate response structure
                        if not response_json.get("response_parts"):
                            logger.warning("Response has no response_parts")
                            response_json = {"response_parts": [{"type": "text", "content": "I received an incomplete response. Please try again."}]}
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode AI response JSON: {e}")
                        logger.error(f"Response string: {full_response_str}")
                        response_json = {"response_parts": [{"type": "text", "content": "Sorry, I received an invalid response from the AI. Please try rephrasing your question."}]}
                    except Exception as e:
                        logger.error(f"Unexpected error parsing response: {e}")
                        response_json = {"response_parts": [{"type": "text", "content": "An unexpected error occurred while processing the response."}]}

                    # Ensure response_json is set before rendering and appending to messages
                    if response_json is not None:
                        render_chat_response(response_json)
                        st.session_state.messages.append({"role": "assistant", "content": response_json})
                        # Extract and store suggested questions from the response
                        if "suggested_questions" in response_json and response_json["suggested_questions"]:
                            st.session_state.suggested_prompts = response_json["suggested_questions"]
                            logger.info(f"🎯 PROCESS COMPLETE: Response delivered with {len(response_json.get('response_parts', []))} parts and {len(response_json.get('suggested_questions', []))} follow-up questions")

                    # Always display suggested follow-up questions immediately after the AI response
                    if st.session_state.get("suggested_prompts"):
                        st.markdown("**💡 Suggested follow-up questions:**")
                        with st.container():
                            for i, prompt_text in enumerate(st.session_state.suggested_prompts):
                                button_key = f"suggested_btn_{i}_{len(st.session_state.messages)}"
                                if st.button(f"❓ {prompt_text}", key=button_key, use_container_width=True):
                                    st.session_state.prompt = prompt_text
                                    st.session_state.suggested_prompts = []
                                    st.rerun()
                        clear_key = f"clear_suggestions_{len(st.session_state.messages)}"
                        if st.button("🗑️ Clear suggestions", key=clear_key):
                            st.session_state.suggested_prompts = []
                            st.rerun()

                except Exception as e:
                    logger.error(f"An error occurred: {e}", exc_info=True)
                    st.error("❌ An error occurred while processing your request. Please try again.")
                    status.update(label="❌ Error occurred", state="error")
    
    # Close chat container
    st.markdown('</div>', unsafe_allow_html=True)

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
