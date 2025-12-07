from itertools import product
import operator
import os
import pandas as pd

from typing import Dict, List, TypedDict, Annotated, Literal

from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# ===========================
# ENVIRONMENT SETUP
# ===========================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = ChatGroq(model="openai/gpt-oss-20b")

# ===========================
# LOAD PRODUCTS
# ===========================
PRODUCT_FILE = "products.csv"
if not os.path.exists(PRODUCT_FILE):
    raise FileNotFoundError("⚠️ Product file not found! Please ensure 'products.csv' exists.")

products_df = pd.read_csv(PRODUCT_FILE)

# ===========================
# TOOL 1 — FILTER PRODUCTS
# ===========================
@tool
def filter_products(product_type: str = None, min_rating: float = 0.0,
                    price_min: float = 0.0, price_max: float = 999999.0) -> List[Dict]:
    """
    Filter products by type, minimum rating, and price range.
    Returns top 5 recommended products with Product_ID.
    """
    df = products_df.copy()

    if product_type:
        df = df[df["Product_Type"].str.contains(product_type, case=False, na=False)]

    df = df[
        (df["Rating"] >= min_rating)
        & (df["Price"] >= price_min)
        & (df["Price"] <= price_max)
    ]

    df = df.sort_values(by=["Rating", "Price"], ascending=[False, True]).head(5)

    if df.empty:
        return [{"message": "No products found matching your criteria."}]
    return df.to_dict(orient="records")

# ===========================
# TOOL 2 — INVENTORY CHECK
# ===========================
@tool
def check_inventory(product_id: str) -> bool:
    """
    Checks if a product exists and has Stock > 0.
    Returns True if available, False otherwise.
    """
    product = products_df[products_df["Product_ID"] == product_id]
    return not product.empty and product.iloc[0]["Stock"] > 0

# ===========================
# TOOL 3 — CHECKOUT
# ===========================
@tool
def checkout(product_id: str) -> Dict:
    """
    Simulates checkout for a given product.
    Adds the item to the cart if available.
    """
    product = products_df[products_df["Product_ID"] == product_id]
    if product.empty:
        return {"success": False, "message": f"❌ Product {product_id} not found."}
    if product.iloc[0]["Stock"] <= 0:
        return {"success": False, "message": f"❌ Product {product_id} is out of stock."}

    product_data = {
        "Product_ID": str(product.iloc[0]["Product_ID"]),
        "Product_Name": product.iloc[0]["Product_Name"],
        "Price": float(product.iloc[0]["Price"]),
    }

    return {
        "success": True,
        "message": f"✅ {product_data['Product_Name']} successfully added to your cart.",
        "Product_ID": product_data["Product_ID"],
    }

# ===========================
# TOOL LIST
# ===========================
tools = [filter_products, check_inventory, checkout]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# ===========================
# LANGGRAPH STATE DEFINITION
# ===========================
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# ===========================
# NODE 1 — LLM ReasonING NODE
# ===========================
def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="""You are a sales agent for happie mobiles.
                        - Only handle mobile phone purchases.
                        - When the user provides a product type, rating, and price range, immediately call the filter_products tool with those arguments.
                        - After calling filter_products tool, give the list in tabular form.
                        - Observe for words such as buy, check, okay, nice, help me check, to call the check_inventory tool.
                        - Always use the tool when filters are clear.
                        """
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# ===========================
# NODE 2 — TOOL EXECUTION
# ===========================
def tool_node(state: dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}

# ===========================
# CONTROL FLOW
# ===========================
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue or stop"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tool_node"
    return END

# ===========================
# MEMORY (OPTION A)
# ===========================
conversation_memory: list[AnyMessage] = []  # stores all messages in memory

# ===========================
# MAIN RESPONSE FUNCTION
# ===========================
def get_response(user_input: str) -> str:
    global conversation_memory

    # Add user input to memory
    conversation_memory.append(HumanMessage(content=user_input))

    # Initialize with full conversation history
    state: MessagesState = {
        "messages": conversation_memory,
        "llm_calls": 0
    }

    # Build the LangGraph agent
    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent_builder.add_edge("tool_node", "llm_call")
    agent = agent_builder.compile()

    # Run the graph
    result_state = agent.invoke(state)

    # Get the model's reply
    final_messages = result_state.get("messages", [])
    if final_messages:
        last_msg = final_messages[-1]

        # Save LLM reply to memory
        conversation_memory.append(last_msg)

        return getattr(last_msg, "content", str(last_msg))
    return "Sorry, I couldn't process your request."

# ===========================
# TESTING / DEMO
# ===========================

response = get_response("I want to buy smartphone with greator than 4 rating and price between 800 and 1000 dollars")
print("Agent:", response)
