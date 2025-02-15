import random
import json
import pandas as pd
import os
import streamlit as st
import requests
from langchain import PromptTemplate, LLMChain  # Import PromptTemplate and LLMChain
from langchain.llms import HuggingFacePipeline  # Import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # Import pipeline function

# ✅ Load Model for Conversational AI
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ✅ Create AI Pipeline
chatbot_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=200)
llm = HuggingFacePipeline(pipeline=chatbot_pipeline)

# ✅ Load Product Catalog
json_path = "product_catalog.json"

if not os.path.exists(json_path):
    st.error(f"❌ ERROR: The JSON file '{json_path}' is missing.")
    st.stop()

df_catalog = pd.read_json(json_path)

# ✅ Check if the required columns exist
required_columns = {"id", "name", "brand", "price", "category", "stock", "rating", "discount", "delivery_time"}
if not required_columns.issubset(df_catalog.columns):
    st.error("❌ The product catalog is missing required columns!")
    st.stop()

# ✅ Streamlit App Title
st.title("🛒 AI-Powered E-Commerce Chatbot (Free Version)")
st.write("✅ Loaded Product Catalog:")

# ✅ Display catalog
st.dataframe(df_catalog.head())

# ✅ Check if the API key is available in secrets
if "HUGGINGFACE_API_KEY" in st.secrets:
    HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
else:
    st.error("🚨 Hugging Face API Key is missing! Add it in Streamlit Secrets.")
    st.stop()

# ✅ Query Classification Function
def classify_query(user_query):
    """Classifies the query type based on keywords."""
    user_query = user_query.lower()

    if any(word in user_query for word in ["find", "show", "recommend", "under", "cheapest", "I need"]):
        return "product_search"
    elif any(word in user_query for word in ["stock", "available", "in stock", "in-stock"]):
        return "availability_check"
    elif any(word in user_query for word in ["deliver", "shipping", "arrive"]):
        return "delivery_check"
    elif "category" in user_query or "categories" in user_query:
        return "category_check"
    else:
        return "general"

# ✅ Product Search Function
def find_products(query):
    """Filter products based on price, category, delivery_time, brand, rating, and discount."""
    query_lower = query.lower()

    price_limit = None
    if "under $" in query_lower:
        try:
            price_limit = float(query_lower.split("under $")[1].split()[0])
        except ValueError:
            pass

    category = None
    for cat in df_catalog["category"].unique():
        if cat.lower() in query_lower:
            category = cat

    brand = None
    for b in df_catalog["brand"].unique():
        if b.lower() in query_lower:
            brand = b

    min_rating = None
    if "rated above" in query_lower:
        try:
            min_rating = float(query_lower.split("rated above ")[1].split()[0])
        except ValueError:
            pass

    min_discount = None
    if "at least" in query_lower and "%" in query_lower:
        try:
            min_discount = int(query_lower.split("at least ")[1].split("%")[0])
        except ValueError:
            pass

    # ✅ Apply Filters
    filtered = df_catalog[
        (df_catalog["price"] <= price_limit if price_limit else True) &
        (df_catalog["category"] == category if category else True) &
        (df_catalog["brand"] == brand if brand else True) &
        (df_catalog["rating"] >= min_rating if min_rating else True) &
        (df_catalog["discount"] >= min_discount if min_discount else True) &
        (df_catalog["stock"] > 0)  # Ensure in-stock items
    ]

    return filtered.head(5)  # Return top 5 results

# ✅ Chatbot Logic
def chat_with_bot(user_query):
    """Handles user queries with chatbot and product recommendations."""
    query_type = classify_query(user_query)

    if query_type == "product_search":
        filtered_products = find_products(user_query)

        if not filtered_products.empty:
            product_list = "\n".join([
                f"{row['name']} - ${row['price']} - {row['category']} - {row['brand']} - {row['rating']}⭐ - {row['discount']}% off"
                for _, row in filtered_products.iterrows()
            ])
            return f"Here are some options:\n{product_list}"
        else:
            return "Sorry, no matching products found."

    elif query_type == "category_check":
        categories = df_catalog["category"].unique().tolist()
        return f"Available categories: {', '.join(categories)}"

    elif query_type == "availability_check":
        in_stock = [
            f"Checking stock for: {row['name']} - Stock: {row['stock']}"
            for _, row in df_catalog.iterrows() if row["stock"] > 0
        ]
        return "\n".join(in_stock) if in_stock else "None of these items are currently in stock."

    elif query_type == "delivery_check":
        fast_delivery = [
            row["name"] for _, row in df_catalog.iterrows()
            if row["delivery_time"] in ["Next Day", "Same-day"]
        ]
        return f"These items can be delivered today or tomorrow: {', '.join(fast_delivery)}" if fast_delivery else "None of these items can be delivered today or tomorrow."

    else:
        return "I'm here to help! You can ask me to find products, check availability, or delivery options."

    # ✅ AI Fallback for Unhandled Queries
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model_name}",
        headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
        json={"inputs": user_query, "max_new_tokens": 200}
    )
    
    response_data = response.json()
    if response.status_code == 200 and response_data:
        return response_data[0].get("generated_text", "No response generated.")
    else:
        return f"AI Error: {response_data}"

# ✅ Streamlit Chat Interface
st.subheader("Chat with your AI Assistant")
user_query = st.text_input("Ask me anything about our products:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Send"):
    if user_query:
        response = chat_with_bot(user_query)
        st.session_state.chat_history.append(f"**You:** {user_query}")
        st.session_state.chat_history.append(f"**AI:** {response}")
        st.write(f"**AI:** {response}")

# ✅ Display Chat History
st.subheader("Chat History")
for msg in st.session_state.chat_history:
    st.write(msg)
