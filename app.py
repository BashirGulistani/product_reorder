import streamlit as st
import pandas as pd
import numpy as np
import requests  # Used to simulate web search
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Order Smart Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
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
    /* Sidebar styling */
    .st-emotion-cache-16txtl3 {
        background-color: #ffffff;
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
</style>
""", unsafe_allow_html=True)


# --- Data Generation and Helper Functions ---

@st.cache_data
def load_sample_data():
    df = pd.DataFrame('order_data.csv')
    return df


def search_web_for_link(product_name):
    """
    Simulates a web search to find a product link.
    In a real application, this could use an API like Google Custom Search.
    For demonstration, we'll use a generic Google search link.
    """
    st.toast(f"Searching online for: '{product_name}'...")
    # Mock search: replace with real API if available
    query = product_name.replace(" ", "+")
    return f"https://www.google.com/search?q={query}&tbm=shop"


# --- GEMINI API CALL ---
def get_gemini_response(query, customer_data_str):
    system_prompt = """
    You are a smart assistant specializing in customer order management.
    Your tasks:
    - Clean and standardize product names (e.g., remove extra spaces, fix typos).
    - Summarize past orders intelligently.
    - If the query is about re-ordering, suggest based on past products.
    - Include all relevant details: order dates, quantities, prices, and product links.
    - Provide a clear, concise summary.
    - Show all related information about the customer's orders.
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
st.markdown("Filter by customer details and chat with AI for intelligent order summaries and re-order suggestions.")

# --- Data Upload and Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
    else:
        if st.button("Use Sample Data"):
            df = load_sample_data()
            st.session_state['df'] = df

    if 'df' in st.session_state:
        df = st.session_state['df']
        st.subheader("Filter Customer")

        # Create unique lists for dropdowns to avoid duplicates
        customer_emails = sorted(df['Customer Email'].unique())

        # Filter by email (more reliable)
        selected_email = st.selectbox("Select by Customer Email", options=[""] + list(customer_emails))

        # Optional: Filter by name (can combine with email)
        st.markdown("---")
        st.write("Additional name filters (applied on top of email):")
        first_name_search = st.text_input("Customer First Name (contains)")
        last_name_search = st.text_input("Customer Last Name (contains)")

# --- Main Content Area ---
if 'df' in st.session_state:
    filtered_df = df.copy()

    # Apply filters sequentially
    if selected_email:
        filtered_df = filtered_df[filtered_df['Customer Email'] == selected_email]
    if first_name_search:
        filtered_df = filtered_df[
            filtered_df['Customer First Name'].str.contains(first_name_search, case=False, na=False)]
    if last_name_search:
        filtered_df = filtered_df[
            filtered_df['Customer Last Name'].str.contains(last_name_search, case=False, na=False)]

    # Display results only if a filter has been applied and returned results
    if not filtered_df.equals(df) and not filtered_df.empty:
        st.header("Filtered Customer Orders")

        # Find and fill missing links
        for index, row in filtered_df.iterrows():
            if pd.isna(row['links']):
                product_name = row['Product Name']
                found_link = search_web_for_link(product_name)
                filtered_df.at[index, 'links'] = found_link

        # Display the data table with clickable links
        st.markdown("### Customer Order Details")
        st.dataframe(
            filtered_df,
            column_config={
                "links": st.column_config.LinkColumn("Product Link", display_text="🔗 View Product")
            },
            hide_index=True,
            use_container_width=True
        )

        # --- Chatbot Section ---
        st.markdown("---")
        st.header("💬 Chat with AI")
        st.markdown("Ask a question about the filtered orders (e.g., 'Summarize past orders' or 'Help me re-order the polo shirt').")

        user_query = st.text_input("Your question",
                                   key="chat_input")

        if st.button("Ask AI"):
            if user_query:
                with st.spinner("🧠 AI is thinking..."):
                    # Convert filtered data to a string for the AI
                    customer_data_for_ai = filtered_df.to_csv(index=False)

                    # Get AI response
                    ai_response = get_gemini_response(user_query, customer_data_for_ai)

                    # Display AI response in a styled container
                    st.markdown("### 💡 AI Summary")
                    st.info(ai_response)
            else:
                st.warning("Please enter a question for the AI.")
    elif filtered_df.empty:
        st.warning("No customers found with the specified filters.")
    else:
        st.info("👈 Please use the sidebar to filter and find a customer.")

else:
    st.info("Please upload a CSV file or use the sample data to begin.")
