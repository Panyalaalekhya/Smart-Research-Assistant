# Smart-Research-Assistant
A Smart Research Assistant built with LangGraph, Groq and Tavily

## Features
- Real-time web search using Tavily
- AI-powered answers using Groq's Llama model
- Self-reflection to improve answer quality
- Beautiful web interface using Streamlit
- Persistent search history
- Clickable source links

## Technologies Used
- Python
- LangGraph
- Groq AI (Llama 3.3)
- Tavily Search API
- Streamlit
- LangChain

## Setup
1. Clone this repository
2. Create virtual environment: python -m venv venv
3. Activate venv: .\venv\Scripts\activate
4. Install packages: pip install langchain langchain-groq langgraph python-dotenv tavily-python langchain-community streamlit
5. Create .env file and add:
   - GROQ_API_KEY=your_groq_key
   - TAVILY_API_KEY=your_tavily_key
6. Run terminal version: python Research_Assistant.py
7. Run web version: streamlit run app.py
