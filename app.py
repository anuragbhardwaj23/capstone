from fastapi import FastAPI, HTTPException
from together import Together
import os


app = FastAPI(title="Travel Recommendation Chatbot", version="1.0")


TOGETHER_API_KEY = "b46e1e85bbd7faef330ac533cde023ad56c0f879ac568c3b4e9be716aa72208b"


client = Together(api_key=TOGETHER_API_KEY)


def query_together_ai(user_input):
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[{"role": "user", "content": user_input}],
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Welcome to the Travel Recommendation API"}

@app.get("/recommend")
def recommend(query: str):
    response = query_together_ai(query)
    return {"response": response}

# Run using: uvicorn app:app --reload
