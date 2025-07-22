# Metabase Agent

A conversational AI-powered assistant for exploring and analyzing Metabase dashboards, built with Streamlit, OpenAI/Gemini LLMs, and advanced analytics (numpy, scikit-learn).

## Features
- **Conversational Q&A**: Ask questions about your Metabase dashboards in natural language.
- **Automatic Chart & Table Generation**: Get visualizations and tables based on your data and questions.
- **Advanced Analytics**: Summaries, trend prediction, and more using numpy and scikit-learn.
- **Hybrid Card Selection**: Uses ML (TF-IDF), keyword matching, and LLMs to find the most relevant dashboard cards.
- **Follow-up Support**: Handles "next", "continue", and other follow-up queries using conversation history.
- **Streamlit UI**: Modern, interactive web interface.

## Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/sameerhassanofficial/metabase-agent.git
   cd metabase-agent
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables**
   - Copy `.env.example` to `.env` and fill in your Metabase and LLM API keys (do NOT commit secrets).

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Usage
- Open your browser to [http://localhost:8501](http://localhost:8501)
- Select a dashboard and start asking questions!
- The agent will analyze your question, fetch relevant data, and respond with text, charts, or tables.

## Security
- **Never commit your `.env` file or API keys.**
- This repo uses GitHub push protection to prevent accidental secret leaks.

## Contributing
Pull requests and issues are welcome!

## License
MIT 