# ==================== STEP 1 : LOAD MODULES ====================

import os
import streamlit as st
import yfinance as yf
from datetime import datetime

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI


from langchain.agents.initialize import initialize_agent
from langchain.agents.agent_types import AgentType



load_dotenv()


# ==================== STEP 2 : PAGE CONFIG ====================

st.set_page_config(
    page_title="AI Stock Market Analysis Chatbot",
    layout="wide"
)

st.title("📈 AI Stock Market Analysis Chatbot")

st.sidebar.title("SET API CONFIG")

GOOGLE_API_KEY = st.sidebar.text_input(
    "GOOGLE_API_KEY",
    type="password"
)

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    st.sidebar.success("API Key Loaded!!")
else:
    st.sidebar.info("Enter API Key")


# ==================== STEP 3 : STOCK TOOL ====================

def get_stock_price(symbol: str) -> str:

    try:
        # Sanitize input: LLMs sometimes pass symbols with quotes or trailing spaces
        clean_symbol = symbol.strip().replace("'", "").replace('"', "")
        
        stock = yf.Ticker(clean_symbol.upper())
        info = stock.info

        current = info.get("currentPrice", "Not Available")
        previous = info.get("previousClose", "Not Available")
        open_price = info.get("open", "Not Available")
        high = info.get("dayHigh", "Not Available")
        low = info.get("dayLow", "Not Available")

        return f"""
Stock : {clean_symbol.upper()}
Current Price : {current}
Previous Close : {previous}
Open Price : {open_price}
Day High : {high}
Day Low : {low}
"""

    except Exception:
        return "Unable to fetch stock information. Make sure the ticker symbol is correct."


# ==================== STEP 4 : DATE TIME TOOL ====================

def current_datetime(query: str) -> str:
    # Langchain tools always pass a string input, even if it's unused. 
    now = datetime.now()

    return now.strftime(
        "Current Date : %d-%m-%Y\nCurrent Time : %H:%M:%S"
    )


# ==================== STEP 5 : CREATE TOOLS ====================

tools = [
    Tool(
        name="Stock_Price_Tool", 
        func=get_stock_price,
        description="Use this tool to get live stock price using stock symbol like AAPL, TSLA, MSFT, RELIANCE.NS."
    ),
    Tool(
        name="Current_Date_Time",
        func=current_datetime,
        description="Use this tool whenever user asks today's date or current time."
    )
]


# ==================== STEP 6 : LOAD GEMINI ====================

if GOOGLE_API_KEY:

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    # ==================== STEP 7 : CREATE AGENT ====================

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
        handle_parsing_errors=True, 
        verbose=False
    )

    # ==================== STEP 8 : USER INPUT ====================

    st.subheader("Ask Your Question")

    user_question = st.text_area(
        "Example:\nWhat is the current price of AAPL?"
    )

    if st.button("Get Answer"):

        if user_question:

            with st.spinner("Generating Answer..."):

                try:
                    # Update to 'invoke' as 'run' is deprecated
                    response = agent.invoke({"input": user_question})

                    st.success("Answer Generated")
                    st.write(response["output"])

                except Exception as e:
                    st.error("Something went wrong.")
                    st.write(f"Error Details: {e}")

        else:
            st.warning("Please enter a question.")
else:
    st.info("Please enter your Google API Key in the sidebar to start.")
