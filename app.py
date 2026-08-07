# ==================== STEP 1 : LOAD MODULES ====================

import os
import streamlit as st
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage

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
    """Fetch live stock price details using a stock ticker symbol like AAPL, TSLA, MSFT, RELIANCE.NS."""
    try:
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

def current_datetime(query: str = "") -> str:
    """Returns today's date and current time."""
    now = datetime.now()
    return now.strftime("Current Date : %d-%m-%Y\nCurrent Time : %H:%M:%S")

# ==================== STEP 5 : TOOL MAPPING ====================

tools = [get_stock_price, current_datetime]
tools_by_name = {
    "get_stock_price": get_stock_price,
    "current_datetime": current_datetime
}

# ==================== STEP 6 : LOAD GEMINI ====================

if GOOGLE_API_KEY:

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    # Bind tools directly to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # ==================== STEP 7 : USER INPUT ====================

    st.subheader("Ask Your Question")

    user_question = st.text_area(
        "Example:\nWhat is the current price of AAPL?"
    )

    if st.button("Get Answer"):

        if user_question:

            with st.spinner("Generating Answer..."):

                try:
                    messages = [HumanMessage(content=user_question)]
                    ai_msg = llm_with_tools.invoke(messages)
                    messages.append(ai_msg)

                    # Check if Gemini decided to invoke a tool
                    if ai_msg.tool_calls:
                        for tool_call in ai_msg.tool_calls:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]

                            # Execute the requested function
                            if tool_name in tools_by_name:
                                tool_output = tools_by_name[tool_name](**tool_args)
                            else:
                                tool_output = f"Tool {tool_name} not found."

                            messages.append(
                                ToolMessage(
                                    content=str(tool_output),
                                    tool_call_id=tool_call["id"]
                                )
                            )

                        # Pass tool results back to Gemini for final summary
                        final_response = llm.invoke(messages)
                        st.success("Answer Generated")
                        st.write(final_response.content)
                    else:
                        st.success("Answer Generated")
                        st.write(ai_msg.content)

                except Exception as e:
                    st.error("Something went wrong.")
                    st.write(f"Error Details: {e}")

        else:
            st.warning("Please enter a question.")
else:
    st.info("Please enter your Google API Key in the sidebar to start.")
