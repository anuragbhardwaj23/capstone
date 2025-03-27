from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from together import Together
from uuid import uuid4
import spacy
import re
import json

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

# Categories we support
CATEGORY_KEYWORDS = {
    "student": ["student", "university"],
    "senior": ["senior", "elderly"],
    "army": ["army", "military", "armed forces"]
}

# Dummy data
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
    ],
    "hotels": [
        {"city": "Los Angeles", "name": "Hilton LA", "rating": 4.5, "price_per_night": 120},
        {"city": "San Francisco", "name": "Marriott SF", "rating": 4.7, "price_per_night": 150},
        {"city": "Bangalore", "name": "The Leela Palace", "rating": 4.9, "price_per_night": 1800},
        {"city": "Delhi", "name": "Taj Mahal Hotel", "rating": 4.8, "price_per_night": 2200},
        {"city": "Mumbai", "name": "The Oberoi", "rating": 4.6, "price_per_night": 2500},
        {"city": "Chennai", "name": "ITC Grand Chola", "rating": 4.7, "price_per_night": 2300},
        {"city": "Kolkata", "name": "JW Marriott", "rating": 4.6, "price_per_night": 2100},
        {"city": "Hyderabad", "name": "Hyatt Hyderabad", "rating": 4.5, "price_per_night": 1900},
        {"city": "New York", "name": "Plaza Hotel", "rating": 4.9, "price_per_night": 350},
        {"city": "Chicago", "name": "Four Seasons", "rating": 4.8, "price_per_night": 320}
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

# Query AI model and filter out <think></think> content
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
            # Remove content inside <think> </think> tags
            cleaned_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
            return cleaned_response if cleaned_response else "Sorry, I couldn't find relevant results."
        else:
            return "Sorry, I couldn't find relevant results."
    except Exception as e:
        return f"Error retrieving AI response: {str(e)}"

@app.get("/")
def home():
    return {"message": "Welcome to the Travel Recommendation API"}

@app.get("/recommend")
def recommend(query: str, session_id: str = None):
    session_id = session_id or str(uuid4())

    session_data = chat_sessions.get(session_id, {
        "chat_history": [],
        "source": None,
        "destination": None,
        "date": None,
        "category": None
    })

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

    # If no match found, call Together AI
    response_text = query_together_ai(query, session_data["chat_history"])
    session_data["chat_history"].append({"role": "assistant", "content": response_text})
    chat_sessions[session_id] = session_data

    return {
        "session_id": session_id,
        "response": response_text,
        "collected": session_data
    }

@app.get("/hotels")
def hotels(city: str, min_rating: float = 0, max_price: float = float("inf")):
    matching_hotels = [
        hotel for hotel in dummy_data["hotels"]
        if hotel["city"].lower() == city.lower() and
           hotel["rating"] >= min_rating and
           hotel["price_per_night"] <= max_price
    ]
    return {"hotels": matching_hotels} if matching_hotels else {"message": f"No hotels found in {city} matching the criteria."}
