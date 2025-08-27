import streamlit as st
import pandas as pd
import numpy as np
import requests  # Used to simulate web search
from google import genai
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Order Smart Chatbot",
    page_icon="🤖",
    layout="wide",
)

try:
    model = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("Could not initialize Gemini client. Please check GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

# --- Professional Styling (CSS) ---
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #f0f2f6;
    }
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    /* Card-like containers */
    .st-emotion-cache-1r4qj8v, .st-emotion-cache-1v0mbdj {
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 20px;
        background-color: #ffffff;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
        transition: 0.3s;
    }
    .st-emotion-cache-1r4qj8v:hover, .st-emotion-cache-1v0mbdj:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid transparent;
        padding: 0.6em 1.2em;
        font-size: 1em;
        font-weight: 500;
        background-color: #1a73e8;
        color: white;
        cursor: pointer;
        transition: border-color 0.25s;
    }
    .stButton>button:hover {
        border-color: #1a73e8;
        background-color: #ffffff;
        color: #1a73e8;
    }
    /* Title styling */
    h1 {
        color: #1a237e;
    }
    /* Chat message styling */
    .st-chat-message {
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Generation and Helper Functions ---

@st.cache_data
def load_sample_data():
    """Creates a sample DataFrame to be used if no file is uploaded."""
    df = pd.read_csv('order_data.csv')
    return df

def search_web_for_link(product_name):
    """
    Simulates a web search to find a product link.
    For demonstration, we'll use a generic Google search link.
    """
    st.toast(f"Searching online for: '{product_name}'...")
    query = product_name.replace(" ", "+")
    return f"https://www.google.com/search?q={query}&tbm=shop"

def extract_customer_info(query):
    prompt = f"""
    Extract the customer identifier from the query. It could be an email, full name, or partial name.
    Return as JSON: {{"email": "email or null", "first_name": "first name or null", "last_name": "last name or null"}}
    If full name like "John Doe", split into first and last.
    Query: {query}
    """
    try:
        resp = model.models.generate_content(
            model="gemini-2.5-flash",
            contents=[system_prompt, user_prompt],
            config={"response_mime_type": "application/json"}
        )
        
        return resp.text
    except Exception as e:
        st.error(f"Error extracting customer info: {e}")
        return {"email": None, "first_name": None, "last_name": None}

# --- GEMINI API CALL ---
def get_gemini_response(query, customer_data_str):
    system_prompt = """
    You are a smart assistant specializing in customer order management.
    Your tasks:
    - Clean and standardize product names (e.g., remove extra spaces, fix typos).
    - Summarize past orders intelligently.
    - If the query is about re-ordering, suggest based on past products, including links.
    - Include all relevant details: order dates, quantities, prices, and product links.
    - Provide a clear, concise summary.
    - Show all related information about the customer's orders.
    - Format nicely with markdown if possible.
    """
    
    user_prompt = f"""
    Customer order data (CSV format):
    {customer_data_str}
    
    User query: {query}
    """
    
    try:
        response = model.generate_content([system_prompt, user_prompt])
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response at this time."

# --- Main Application UI ---
st.title("🤖 Customer Order Smart Chatbot")
st.markdown("Chat with the AI about customer orders. Mention the customer by name or email in your query.")

# Data loading at the top
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['df'] = df
else:
    if st.button("Use Sample Data"):
        df = load_sample_data()
        st.session_state['df'] = df

if 'df' not in st.session_state:
    st.info("Please upload a CSV file or use the sample data to begin.")
    st.stop()

df = st.session_state['df']

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about a customer's orders (e.g., 'Show past orders for John Doe' or 'John Doe wants to reorder the polo shirt')"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("🧠 AI is thinking..."):
        # Extract customer info
        customer_info = extract_customer_info(prompt)
        
        filtered_df = df.copy()
        
        # Apply filters based on extracted info
        if customer_info.get("email"):
            filtered_df = filtered_df[filtered_df['Customer Email'].str.lower() == customer_info["email"].lower()]
        if customer_info.get("first_name"):
            filtered_df = filtered_df[
                filtered_df['Customer First Name'].str.contains(customer_info["first_name"], case=False, na=False)]
        if customer_info.get("last_name"):
            filtered_df = filtered_df[
                filtered_df['Customer Last Name'].str.contains(customer_info["last_name"], case=False, na=False)]
        
        if filtered_df.empty:
            response = "No customer found matching the provided information. Please try again with more details."
        else:
            # Fill missing links
            for index, row in filtered_df.iterrows():
                if pd.isna(row['links']):
                    product_name = str(row['Product Name']) if not pd.isna(row['Product Name']) else "Unknown Product"
                    product_name = product_name.strip()
                    found_link = search_web_for_link(product_name)
                    filtered_df.at[index, 'links'] = found_link
            
            # Convert to CSV string
            customer_data_for_ai = filtered_df.to_csv(index=False)
            
            # Get AI response with full query
            response = get_gemini_response(prompt, customer_data_for_ai)
            
            # Optionally, include table in response
            response += "\n\n### Filtered Order Details\n"
            response += filtered_df.to_markdown(index=False)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


#######
