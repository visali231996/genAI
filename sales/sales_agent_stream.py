from itertools import product
import operator
import os

from typing import Dict, List, TypedDict, Annotated, Literal
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
import pandas as pd
# ===========================
# ENVIRONMENT SETUP
# ===========================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = ChatGroq(model="openai/gpt-oss-20b")
if "cart" not in st.session_state:
    st.session_state.cart = []
cart = st.session_state.cart


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

    # Add to cart (if not already)
    if not any(item["Product_ID"] == product_id for item in cart):
        cart.append(product_data)

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
# NODE 1 — LLM Reasoning Node
# ===========================
# ===========================
# NODE 1 — LLM Reasoning Node
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
                        — after calling filter_products tool, give the list in tabular form
                        -observe for the words such as buy,check,okay,nice,help me check, to call  check_inventory tool
                       -always use the tool when filters are clear.
                        -
                        """
                    )
                   
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

# ===========================
# NODE 2 — TOOL EXECUTION
# ===========================
from langchain.messages import ToolMessage


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
from typing import Literal
from langgraph.graph import StateGraph, START, END


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# ===========================
# BUILD THE GRAPH
# ===========================
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")
agent = agent_builder.compile()



    
# ===========================
# STREAMLIT APP
# ===========================
st.set_page_config(page_title="🤖 AI Sales Assistant", page_icon="🛍️", layout="wide")

# Create two columns: main area (chat + products) and cart
col1, col2 = st.columns([2, 1], gap="large")

# ===========================
# LEFT SIDE — CHAT + PRODUCTS
# ===========================
with col1:
    st.title("🛍️ Sales Assistant")
    st.caption("Chat with your intelligent sales agent for electronics shopping.")

    # Product CSV Viewer
    CSV_PATH = "C:/Users/Visal/OneDrive/Desktop/genai/sales/products.csv"
    df = pd.read_csv(CSV_PATH)
    with st.expander("📂 View Available Phones"):
        st.dataframe(df)

    # Chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content="You are a helpful AI sales assistant.")]

    # Chat history
    for msg in st.session_state.messages[1:]:
        if not hasattr(msg, "content"):
            continue
        role = "🧑‍💼 You" if isinstance(msg, HumanMessage) else "🤖 Assistant"
        st.markdown(f"**{role}:** {msg.content}")

    # User input
    user_input = st.chat_input("Type your message here...")

    def get_response(user_input):
        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking..."):
                result = agent.invoke({"messages": st.session_state.messages})
                st.session_state.messages = result["messages"]

                last_msg = next(
                    (msg.content for msg in reversed(st.session_state.messages)
                    if hasattr(msg, "content") and not isinstance(msg, (HumanMessage, SystemMessage))),
                    None
                )
                if last_msg:
                    st.markdown(f"**🤖 Assistant:** {last_msg}")
                else:
                    st.warning("No valid assistant response found.")

    if user_input:
        get_response(user_input)


# ===========================
# RIGHT SIDE — FIXED CART
# ===========================
with col2:
    st.markdown("### 🛒 Your Cart")

    cart = st.session_state.cart

    if len(cart) == 0:
        st.info("Your cart is empty.")
    else:
        cart_df = pd.DataFrame(cart)
        st.dataframe(cart_df, use_container_width=True)
        total = sum(item["Price"] for item in cart)
        st.success(f"💰 **Total: ${total:.2f}**")

        if st.button("🗑️ Clear Cart"):
            st.session_state.cart.clear()
            st.success("Cart cleared successfully.")
