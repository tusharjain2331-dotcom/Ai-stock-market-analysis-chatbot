# ==================== STEP 1 : LOAD MODULES ====================

import os
import streamlit as st
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

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

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

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

# ==================== STEP 5B : HELPER TO EXTRACT PLAIN TEXT ====================

def extract_text(content):
    """
    Gemini/LangChain sometimes returns content as a string,
    and sometimes as a list of blocks like:
    [{"type": "text", "text": "...", "extras": {...}}]
    This normalizes it into a clean plain string.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
                elif "text" in block:
                    parts.append(block["text"])
        return "\n".join(parts).strip()

    return str(content)

# ==================== STEP 6 : LOAD GEMINI ====================

if GOOGLE_API_KEY:

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.3
    )

    llm_with_tools = llm.bind_tools(tools)

    # ==================== STEP 7 : CHAT STATE ====================

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": str}

    # Render existing chat history as chat bubbles
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ==================== STEP 8 : CHAT INPUT ====================

    user_question = st.chat_input("Ask about a stock... e.g. What is the current price of AAPL?")

    if user_question:

        # Show user's message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    messages = [HumanMessage(content=user_question)]
                    ai_msg = llm_with_tools.invoke(messages)
                    messages.append(ai_msg)

                    if ai_msg.tool_calls:
                        for tool_call in ai_msg.tool_calls:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]

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

                        final_response = llm.invoke(messages)
                        answer_text = extract_text(final_response.content)
                    else:
                        answer_text = extract_text(ai_msg.content)

                    st.markdown(answer_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer_text})

                except Exception as e:
                    error_text = f"⚠️ Something went wrong.\n\n**Error Details:** {e}"
                    st.markdown(error_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_text})

else:
    st.info("Please enter your Google API Key in the sidebar to start.")
