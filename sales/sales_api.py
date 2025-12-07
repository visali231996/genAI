from fastapi import FastAPI

from sales_agent import get_response
app = FastAPI(title="sales Agent", description="gives advice on mobile phones purchase")
@app.get("/agent/{query}")
def get_ai_response(query:str):
    response = get_response(query)
    return {'response': response}