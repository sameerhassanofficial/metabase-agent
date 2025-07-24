# 🧠 AI-Powered MHU Data Assistant

A sophisticated conversational AI-powered assistant for exploring and analyzing Metabase dashboards, specifically optimized for Mobile Health Units (MHU) data analysis. Built with Streamlit, OpenAI/Gemini LLMs, and advanced analytics.

## ✨ Key Features

### 🤖 **Intelligent Data Analysis**
- **Conversational Q&A**: Ask questions about your Metabase dashboards in natural language
- **LLM-Powered Card Selection**: Advanced AI-driven selection of the most relevant dashboard cards
- **Automatic Chart & Table Generation**: Get beautiful visualizations and tables based on your data
- **Advanced Analytics**: Summaries, trend prediction, and statistical analysis using numpy and scikit-learn

### 🎨 **Modern User Interface**
- **Dark Mode Support**: Full dark mode compatibility with proper text visibility
- **Clean, Professional Design**: Modern gradient headers and streamlined interface
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile devices
- **Quick Start Buttons**: Suggested questions for immediate interaction

### ⚡ **Performance Optimizations**
- **Smart Caching**: Dashboard data cached for faster subsequent queries
- **Efficient Data Fetching**: Only loads data when needed, reducing API calls
- **Optimized Logging**: Clean, informative terminal output without clutter
- **Streamlined Processing**: Removed redundant operations for better performance

### 🔧 **Technical Excellence**
- **Hybrid Card Selection**: Combines ML (TF-IDF), keyword matching, and LLMs
- **Follow-up Support**: Handles "next", "continue", and contextual queries
- **Error Recovery**: Graceful error handling with user-friendly messages
- **Real-time Progress**: Clear status indicators during analysis

## 🚀 Quick Start

### 1. **Clone the Repository**
```bash
git clone https://github.com/sameerhassanofficial/metabase-agent.git
cd metabase-agent
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Configure Environment Variables**
Create a `.env` file with your API credentials:
```env
METABASE_URL=https://your-metabase-instance.com
METABASE_USERNAME=your_username
METABASE_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### 4. **Run the Application**
```bash
streamlit run app.py
```

### 5. **Start Analyzing**
- Open your browser to [http://localhost:8501](http://localhost:8501)
- The interface is pre-configured for Dashboard ID 8 (Mobile Health Units)
- Start asking questions or use the suggested quick-start buttons!

## 📊 Example Queries

Try these questions to get started:
- 📊 "Show me the total patient count"
- 👥 "What's the gender distribution of patients?"
- 🏥 "Which Mobile Health Unit has the most patients?"
- 📈 "Show me patient trends over time"
- 🔍 "How many patients were served in the last month?"

## 🎯 What's New

### **Latest Enhancements (v2.0)**
- ✅ **Dark Mode Support**: Perfect visibility in both light and dark themes
- ✅ **Cleaner Interface**: Removed welcome card for streamlined experience
- ✅ **Performance Boost**: Optimized data fetching and caching
- ✅ **Better UX**: Simplified progress indicators and error handling
- ✅ **Professional Design**: Modern gradients and responsive layout

### **Core Capabilities**
- ✅ **Intelligent Analysis**: AI-powered insights and recommendations
- ✅ **Visual Data**: Beautiful charts and interactive visualizations
- ✅ **Export Options**: Download data as CSV files
- ✅ **Follow-up Questions**: Context-aware conversation flow
- ✅ **Real-time Processing**: Live analysis with progress tracking

## 🔒 Security & Best Practices

- **Never commit your `.env` file or API keys**
- **Use environment variables** for all sensitive configuration
- **Regular updates** to dependencies for security patches
- **GitHub push protection** prevents accidental secret leaks

## 🛠️ Technical Architecture

```
metabase-agent/
├── app.py              # Main Streamlit application
├── ai_agent.py         # Core AI logic and LLM integration
├── metabase_client.py  # Metabase API client with caching
├── ui_utils.py         # UI rendering and chart generation
├── analytics_utils.py  # Data analysis and trend prediction
└── requirements.txt    # Python dependencies
```

## 🤝 Contributing

We welcome contributions! Please feel free to:
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 🔧 Submit pull requests
- 📚 Improve documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Metabase** for the powerful analytics platform
- **OpenAI & Google** for advanced LLM capabilities
- **Streamlit** for the excellent web framework
- **Plotly** for beautiful interactive visualizations

---

**Built with ❤️ for Mobile Health Units data analysis** 