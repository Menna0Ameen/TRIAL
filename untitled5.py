import random
import json
import pandas as pd
import os
import torch
import streamlit as st
import requests
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ✅ Set Device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Load Model (Use a Smaller Model for Streamlit Deployment)
model_name = "microsoft/DialoGPT-small"  # Change if necessary
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# ✅ Create LangChain Pipeline
chatbot_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=200)
llm = HuggingFacePipeline(pipeline=chatbot_pipeline)

# ✅ Load Product Catalog
json_path = "product_catalog.json"
df_catalog = pd.read_json(json_path)  # FIXED: Removed quotes

# ✅ Streamlit App Title
st.title("🛒 AI AI-Powered E-Commerce Chatbot (Free Version)")

# ✅ Check Hugging Face API Key (Set Fallback)
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", "your_fallback_api_key")

# ✅ Function to Filter Products
def find_products(query):
    query_lower = query.lower()
    price_limit = None
    category, delivery_time, brand, min_rating, min_discount = None, None, None, None, None

    if "under $" in query_lower:
        try:
            price_limit = int(query_lower.split("under $")[1].split()[0])
        except ValueError:
            pass

    for cat in df_catalog["category"].unique():
        if cat.lower() in query_lower:
            category = cat

    for d in df_catalog["delivery_time"].unique():
        if d.lower() in query_lower:
            delivery_time = d

    for b in df_catalog["brand"].unique():
        if b.lower() in query_lower:
            brand = b

    if "rated above" in query_lower:
        try:
            min_rating = float(query_lower.split("rated above ")[1].split()[0])
        except ValueError:
            pass

    if "at least" in query_lower and "%" in query_lower:
        try:
            min_discount = int(query_lower.split("at least ")[1].split("%")[0])
        except ValueError:
            pass

    filtered = df_catalog[
        (df_catalog["price"] <= price_limit if price_limit else True) &
        (df_catalog["category"] == category if category else True) &
        (df_catalog["delivery_time"] == delivery_time if delivery_time else True) &
        (df_catalog["brand"] == brand if brand else True) &
        (df_catalog["rating"] >= min_rating if min_rating else True) &
        (df_catalog["discount"] >= min_discount if min_discount else True) &
        (df_catalog["stock"] > 0)
    ]

    return filtered.head(5)
    
def classify_query(user_query):
    """Classifies the query type based on keywords."""
    user_query = user_query.lower()

    if any(word in user_query for word in ["find", "show", "recommend", "under", "cheapest", "I need"]):
        return "product_search"
    elif any(word in user_query for word in ["stock", "available", "in stock", "in-stock"]):
        return "availability_check"
    elif any(word in user_query for word in ["deliver", "shipping", "arrive"]):
        return "delivery_check"
    else:
        return "general"

# ✅ Chat Function
def chat_with_bot(user_query):
    query_type = classify_query(user_query)
    response_text = "I'm here to help! You can ask me to find products, check availability, or delivery options."

    if query_type == "product_search":
        filtered_products = find_products(user_query)
        if not filtered_products.empty:
            product_list = "\n".join([
                f"{row['name']} - ${row['price']} - {row['category']} - {row['brand']} - {row['rating']} - {row['discount']}"
                for _, row in filtered_products.iterrows()
            ])
            response_text = f"Here are some options:\n{product_list}"
        else:
            response_text = "Sorry, no matching products found."

    # ✅ API Call to Hugging Face Model
    response_text = requests.post(
        f"https://api-inference.huggingface.co/models/{model_name}",
        headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
        json={"inputs": user_query, "parameters": {"max_new_tokens": 150}}  # FIXED: Limited tokens
    )

    response_data = response_text.json()
    if response_text.status_code == 200 and response_data:
        return response_data[0].get("generated_text", "No response generated.")
    else:
        return f"AI Error: {response_data}"

# ✅ Streamlit Chat Interface
st.subheader("Chat with your AI Assistant")
user_query = st.text_input("Ask me anything about our products:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Send") and user_query:
    response = chat_with_bot(user_query)
    st.session_state.chat_history.append(f"You: {user_query}")
    st.session_state.chat_history.append(f"AI: {response}")
    st.write(f"AI: {response}")

st.subheader("Chat History")
for msg in st.session_state.chat_history:
    st.write(msg)
