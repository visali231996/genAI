<u><b>#Sales Agent chatbot:</b></u>
 This chatbot uses langgraph to create static AI LLM agent. static model agent meaning we define which type of LLM we are going to use in our projectand it does not change while we are running the agent.
 We send tools to the agent that perform specific tasks when called. tools hep in Multiple tool calls in sequence (triggered by a single prompt),Parallel tool calls when appropriate, Dynamic tool selection based on previous results. error handling.
-------
 <u><b>##Feautures:</b></u>
 focus on giving information about the list of products that fall under user requirements(based on price range,rating,type) and also checking inventory and checking out.

 ###<b>Scope control:</b>
 stritly focusses on smartphones that are present in user given database(products.csv)

 <b>###jail breaking:</b>
 does not hallucinate and give recommendations based on pre defined criteria..if asked about other topics...politely denies the request.

<b>### markdown:</b>
 uses markdown so that output is given in given transcript

<b> ###Loacl session memory:</b>
 Session content is stored only in Streamlit session memory. No persistent storage or logging.

 -------
<u><b>## Project structure</u></b>
 📦 sales/
│
├── 🔐 .env
├── 📄 products.csv
├── 📝 readme.txt
├── 🤖 sales_agent.py
├── ⚡ sales_agent_stream.py
├── 🌐 sales_api.py
├── 📄 tree.txt
│
└── 🐍 venv/
    ├── .gitignore
    └── pyvenv.cfg
------
<u><b>##Set Up:</b></u>
1.Create virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
2. Install dependencies
pip install -r requirements.txt
3.Configure environment variables
Create a .env file inside the root folder:
GROQ_API_KEY=your_groq_api_key_here
-------
<u><b>##Run the app:</b></u>
streamlit run sales_agent_stream.py
and open the link  http://localhost:8501

for FastAPI:
uvicorn sales_api:app --test_reload
and open the link http://127.0.0.1:8000/docs
------
<u><b>##Usage:</b></u>
1. search for the products
example:need oppo smatphone with proice less than 30000 rupees, rating above 4.5

2.ask to check for the desired phone inventory
example: check for Oppo reno 6

3. if product is available, ask to check out

4. check for message : product added to cart successfully
----
<u><b>##requirements:</b></u>
streamlit
python-dotenv
langchain
groq
pandas
optional:
matplotlib
openpyxl
----
<u><b>##Data Notes</b></u>
File	Description
products.csv	Product catalog used for recommendations
cart.json	    Tracks current user cart
.env	       Stores secret API key (not committed)
----
<u><b>##security notes:</b></u>
.env and session files are ignored via .gitignore.
All session history is saved locally (/sessions/ folder).
No external API calls beyond Groq’s secure LLM endpoint.
------

 <u><b>##Future Enhancements:</b></u>
 Add user authentication and profile-based recommendations
 Integrate with live e-commerce APIs (Shopify/Amazon mock)
 Include sentiment-based conversational upselling
 Deploy on Streamlit Cloud or HuggingFace Spaces