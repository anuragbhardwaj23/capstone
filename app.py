from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from together import Together
from uuid import uuid4
import spacy
import re

app = FastAPI(title="Travel Recommendation Chatbot", version="1.1")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TOGETHER_API_KEY = "b46e1e85bbd7faef330ac533cde023ad56c0f879ac568c3b4e9be716aa72208b"
client = Together(api_key=TOGETHER_API_KEY)

nlp = spacy.load("en_core_web_sm")

# Session-based memory
chat_sessions = {}

# Welcome message
WELCOME_MESSAGE = "Hello! I’m your travel assistant. How can I help you today?"

# Categories we support
CATEGORY_KEYWORDS = {
    "student": ["student", "university"],
    "senior": ["senior", "elderly"],
    "army": ["army", "military", "armed forces"]
}

# Dummy flight data
dummy_data = {
    "flights": [
        {"source": "Delhi", "destination": "Bangalore", "date": "2025-04-28", "category": "student", "price": "₹7000"},
        {"source": "Bangalore", "destination": "Mumbai", "date": "2025-04-28", "category": "senior", "price": "₹4000"},
        {"source": "Mumbai", "destination": "Chennai", "date": "2025-05-10", "category": "student", "price": "₹5500"},
        {"source": "Chennai", "destination": "Hyderabad", "date": "2025-05-12", "category": "army", "price": "₹3500"},
        {"source": "Hyderabad", "destination": "Kolkata", "date": "2025-06-15", "category": "senior", "price": "₹5000"},
        {"source": "Kolkata", "destination": "Delhi", "date": "2025-06-20", "category": "student", "price": "₹6200"},
        {"source": "San Francisco", "destination": "New York", "date": "2025-07-04", "category": "student", "price": "$300"},
        {"source": "Los Angeles", "destination": "Chicago", "date": "2025-08-10", "category": "army", "price": "$250"}
    ]
}

# Preprocess and extract entities
def preprocess_and_extract_entities(text):
    clean_text = re.sub(r"[^\w\s]", "", text.lower())
    doc = nlp(clean_text)

    source = destination = travel_date = category = None

    for ent in doc.ents:
        if ent.label_ == "GPE":
            if not source:
                source = ent.text.title()
            else:
                destination = ent.text.title()
        elif ent.label_ == "DATE":
            travel_date = ent.text  # Handle date parsing here if needed

    for word in clean_text.split():
        for key, values in CATEGORY_KEYWORDS.items():
            if word in values:
                category = key

    return source, destination, travel_date, category

# Function to clean AI responses (Remove Markdown-style formatting)
def clean_response(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Remove **bold**
    text = re.sub(r"#{1,3}\s*", "", text)  # Remove #, ##, ###
    return text.strip()

# Query AI model
def query_together_ai(user_input, chat_history):
    messages = [{"role": "system", "content": "You are a helpful travel assistant that recommends flights and hotels."}]
    messages += chat_history
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=messages
        )
        if response.choices:
            raw_response = response.choices[0].message.content
            cleaned_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
            return clean_response(cleaned_response) if cleaned_response else "Sorry, I couldn't find relevant results."
        else:
            return "Sorry, I couldn't find relevant results."
    except Exception as e:
        return f"Error retrieving AI response: {str(e)}"

@app.get("/")
def home():
    return {"message": "Welcome to the Travel Recommendation API"}

@app.get("/recommend")
def recommend(query: str = "", session_id: str = None):
    session_id = session_id or str(uuid4())

    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "chat_history": [{"role": "assistant", "content": WELCOME_MESSAGE}],
            "source": None,
            "destination": None,
            "date": None,
            "category": None
        }
        return {
            "session_id": session_id,
            "response": WELCOME_MESSAGE,
            "collected": chat_sessions[session_id]
        }

    session_data = chat_sessions[session_id]

    # Extract entities from the query
    src, dest, date, cat = preprocess_and_extract_entities(query)

    if src:
        session_data["source"] = src
    if dest:
        session_data["destination"] = dest
    if date:
        session_data["date"] = date
    if cat:
        session_data["category"] = cat

    session_data["chat_history"].append({"role": "user", "content": query})

    # Check flights in dummy data
    for flight in dummy_data["flights"]:
        if (
            session_data.get("source") and session_data.get("destination") and session_data.get("date") and session_data.get("category") and
            flight["source"].lower() == session_data["source"].lower() and
            flight["destination"].lower() == session_data["destination"].lower() and
            flight["date"] == session_data["date"] and
            flight["category"] == session_data["category"]
        ):
            response_text = (
                f"Here is a {session_data['category']} discounted flight from {session_data['source']} "
                f"to {session_data['destination']} on {session_data['date']} for {flight['price']}."
            )
            session_data["chat_history"].append({"role": "assistant", "content": response_text})
            chat_sessions[session_id] = session_data
            return {
                "session_id": session_id,
                "response": response_text,
                "collected": session_data
            }

    # If no match found, call AI model
    response_text = query_together_ai(query, session_data["chat_history"])
    session_data["chat_history"].append({"role": "assistant", "content": response_text})
    chat_sessions[session_id] = session_data

    return {
        "session_id": session_id,
        "response": response_text,
        "collected": session_data
    }
