import streamlit as st
import pandas as pd
import numpy as np
import io
import requests # Used to simulate web search

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Order Smart Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
def create_sample_data():
    """Creates a sample DataFrame to be used if no file is uploaded."""
    data = {
        'Customer First Name': ['John', 'Jane', 'John', 'Peter', 'Mary', 'Jane'],
        'Customer Last Name': ['Smith', 'Doe', 'Smith', 'Jones', 'Brown', 'Doe'],
        'Customer Email': ['john.smith@example.com', 'jane.doe@example.com', 'john.smith@example.com', 'peter.jones@example.com', 'mary.brown@example.com', 'jane.doe@example.com'],
        'Order ID': [101, 102, 103, 104, 105, 106],
        'Product Name': ["TM18304 Men's VEGA Performance Tech Quarter Zip", "QC300 Avalon: 5-in-1 Power Bank", "STYLE# 1578 MESH POLO", "SanMar Ladies Perfect Weight V-Neck Tee. DT5501", "Koozie® Collapsible Can Kooler", "A product with no link"],
        'Product SKU': ['TM18304', 'QC300', '1578', 'DT5501', '45448', 'XYZ-987'],
        'Supplier Name': ['SPOKE', 'PCNA', 'S&S Activewear', 'Sanmar', 'Koozie Group', 'Unknown Supplier'],
        'Quantity': [2, 1, 3, 5, 10, 1],
        'Price': [55.99, 45.00, 32.50, 12.00, 1.50, 25.00]
    }
    df = pd.DataFrame(data)
    return df

# Mapping of supplier names to their URL templates for link generation
supplier_map = {
    'SPOKE': 'https://spokeapparel.com/?s={productId}',
    'Sanmar': 'https://sanmar.com/search?text={productId}',
    'S&S Activewear': 'https://www.ssactivewear.com/ps/?q={productId}',
    'PCNA': 'https://www.pcna.com/en-us/Search?SearchTerm={productId}',
    'Koozie Group': 'https://www.kooziegroup.com/searchai/?query={productId}',
    # Add other suppliers from your list here...
}

def generate_link(row):
    """Generates a product link based on the supplier and SKU."""
    supplier = row['Supplier Name']
    sku = row['Product SKU']
    if pd.isna(supplier):
        return None
    url_template = supplier_map.get(supplier)
    if not url_template:
        return None
    if '{productId}' in url_template and not pd.isna(sku):
        return url_template.format(productId=str(sku))
    elif '{productId}' not in url_template:
        return url_template
    return None

def search_web_for_link(product_name):
    """
    Simulates a web search to find a product link.
    In a real application, this would use an API like Google Search.
    """
    st.toast(f"Searching online for: '{product_name}'...")
    # This is a mock search. Replace with a real search API call.
    # For demonstration, we'll use a generic Google search link.
    query = product_name.replace(" ", "+")
    return f"https://www.google.com/search?q={query}"

# --- MOCK GEMINI API CALL ---
def get_gemini_response(query, customer_data_str):
    """
    Mocks a call to the Gemini API.
    In a real app, you would use the actual google.generativeai library.
    """
    # This is a canned response for demonstration.
    # A real implementation would send the query and data to the AI model.
    product_lines = []
    for line in customer_data_str.strip().split('\n')[1:]: # Skip header
        parts = line.split(',')
        if len(parts) > 4:
            product_lines.append(f"- **{parts[4].strip()}** (Qty: {parts[7].strip()})")

    products_summary = "\n".join(product_lines)

    response = f"""
    Based on the customer's order history, here is a summary:

    The customer has placed **{len(product_lines)}** order(s) for the following items:
    {products_summary}

    If they wish to re-order, you can use the links provided in the data table below to find the products.
    """
    return response


# --- Main Application UI ---
st.title("🤖 Customer Order Smart Chatbot")
st.markdown("Upload your order data, filter by customer, and ask the AI for an order summary.")

# --- Data Upload and Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        if st.button("Use Sample Data"):
            df = create_sample_data()
            st.session_state['df'] = df

    if 'df' in st.session_state:
        df = st.session_state['df']
        st.subheader("Filter Customer")
        
        # Create unique lists for dropdowns to avoid duplicates
        customer_emails = df['Customer Email'].unique()
        
        # Filter by email (more reliable)
        selected_email = st.selectbox("Select by Customer Email", options=[""] + list(customer_emails))

        # Optional: Filter by name
        st.markdown("---")
        st.write("Or filter by name:")
        first_name_search = st.text_input("Customer First Name (contains)")
        last_name_search = st.text_input("Customer Last Name (contains)")

# --- Main Content Area ---
if 'df' in st.session_state:
    filtered_df = df.copy()

    # Apply filters
    if selected_email:
        filtered_df = filtered_df[filtered_df['Customer Email'] == selected_email]
    elif first_name_search or last_name_search:
        if first_name_search:
            filtered_df = filtered_df[filtered_df['Customer First Name'].str.contains(first_name_search, case=False, na=False)]
        if last_name_search:
            filtered_df = filtered_df[filtered_df['Customer Last Name'].str.contains(last_name_search, case=False, na=False)]

    # Display results only if a filter has been applied and returned results
    if not filtered_df.equals(df) and not filtered_df.empty:
        st.header("Filtered Customer Orders")
        
        # Generate initial links
        filtered_df['links'] = filtered_df.apply(generate_link, axis=1)

        # Find and fill missing links
        for index, row in filtered_df.iterrows():
            if pd.isna(row['links']):
                product_name = row['Product Name']
                found_link = search_web_for_link(product_name)
                filtered_df.loc[index, 'links'] = found_link
        
        # Display the data table with clickable links
        st.markdown("### Customer Order Details")
        st.dataframe(
            filtered_df,
            column_config={
                "links": st.column_config.LinkColumn("Product Link", display_text="🔗 View Product")
            },
            hide_index=True
        )

        # --- Chatbot Section ---
        st.markdown("---")
        st.header("💬 Chat with AI")
        st.markdown("Ask a question about the filtered orders above.")

        user_query = st.text_input("Your question (e.g., 'Summarize past orders' or 'Help me re-order the polo shirt')", key="chat_input")

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
    elif not filtered_df.empty:
        st.info("👈 Please use the sidebar to filter and find a customer.")
    else:
        st.warning("No customers found with the specified filters.")

else:
    st.info("Please upload a CSV file or use the sample data to begin.")
