
import streamlit as st
import pandas as pd
import json
from google import genai
from openai import OpenAI

# --- Page Configuration ---
st.set_page_config(
    page_title="Past Customer Orders Chatbot",
    page_icon="🤖",
    layout="wide",
)



# --- Gemini client ---
try:
    model = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("Could not initialize Gemini client. Please set GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

st.markdown("""
<style>
    /* General App Background */
    .stApp {
        background-color: #f9fafc;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title in center */
    h1 {
        text-align: center;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    .stCaption {
        text-align: center !important;
        display: block;
        margin-bottom: 2rem;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 16px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        font-size: 1rem;
        line-height: 1.5;
    }
    .stChatMessage[data-testid="stChatMessage-user"] {
        background: #e6f0ff;
        border: 1px solid #c2dbff;
        margin-left: auto;
        margin-right: 0;
        max-width: 75%;
    }
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background: #ffffff;
        border: 1px solid #e6e6e6;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
        margin-right: auto;
        margin-left: 0;
        max-width: 75%;
    }

    /* Chat input styled like ChatGPT */
    
    div[data-baseweb="input"]:focus-within {
        border: 1px solid #1a73e8 !important;
        box-shadow: 0 0 0 2px rgba(26,115,232,0.2);
    }

    /* Chat input placeholder */
    div[data-baseweb="input"] input {
        font-size: 1rem !important;
    }

    /* Button styles */
    .stButton>button {
        border-radius: 30px;
        padding: 0.6em 1.4em;
        font-size: 1rem;
        font-weight: 500;
        background-color: #1a73e8;
        color: white !important;
        border: none;
        cursor: pointer;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #155fc9;
    }

    /* Previous styles can go here if needed */





    /*
    This targets the actual text input area within the container.
    We make it transparent so it doesn't look like a box inside a box.
    */
    div[data-testid="stChatInput"] textarea {
        background-color: transparent;
        border: none;
        box-shadow: none;
        height: 100px; /* Adjust the height as you like */
        font-size: 1rem;
        color: #1a1a1a; /* Set text color */
    }

    /* Style for the placeholder text */
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #888888;
    }
</style>
""", unsafe_allow_html=True)



# --- Data ---
@st.cache_data
def load_sample_data() -> pd.DataFrame:
    # Read as strings to avoid unintended numeric coercions (tracking, zip, SKUs, etc.)
    #df = pd.read_csv("ALLORDERS_FNAL.csv", dtype=str, keep_default_na=True, na_values=["", "NA", "NaN", "nan"])
    df = pd.read_parquet('PastOrders.parquet')
    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    return df

df = None
try:
    df = load_sample_data()
except FileNotFoundError:
    st.error("Sample data file 'order_data.csv' not found in the app directory.")
    st.stop()

# Conversation context for follow-ups
if "current_context" not in st.session_state:
    st.session_state.current_context = None  # {"customer": {...}, "orders": [...], "df": DataFrame}



import re



from textwrap import dedent






# --- Helpers ---
import re
from typing import Optional, Dict

# Precompile email regex (strict-enough for app use)
_EMAIL_START_RE = re.compile(
    r"^\s*([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b",
    re.IGNORECASE
)

# Matches a leading single-quoted name:  'Steven Henderson' wants to reorder...
# Captures the inside of the first quote pair.
import re
from typing import Dict, Optional

# Regex for a full email address
_EMAIL_START_RE = re.compile(r"^\s*([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b", re.I)

# 
_QUOTED_NAME_START_RE = re.compile(r"^\s*'([^']+?)'")

# NEW
_DOMAIN_START_RE = re.compile(r"^\s*@([A-Z0-9.-]+)", re.I)

def extract_customer_info(query: str) -> Dict[str, Optional[str]]:
    """
    Extracts customer info from a query string.
    The query can be a full email, a domain/brand name, or a full name.
    """
    query = query.strip()
    
    # Return dictionary structure
    result = {"email": None, "first_name": None, "last_name": None, "domain": None}

    # 1. Check for a full email address (most specific)
    m_email = _EMAIL_START_RE.match(query)
    if m_email:
        result["email"] = m_email.group(1).lower()
        return result

    # 2. Check for a domain name (e.g., @honehealth.com)
    m_domain = _DOMAIN_START_RE.match(query)
    if m_domain:
        result["domain"] = m_domain.group(1).lower()
        return result

    # 3. Check for a quoted full name
    m_name = _QUOTED_NAME_START_RE.match(query)
    if m_name:
        name = m_name.group(1).strip()
        if "," in name:  # "Last, First"
            last, first, *_ = [p.strip() for p in name.split(",")]
            result["first_name"] = first or None
            result["last_name"] = last or None
        else:
            parts = [p for p in name.split() if p]
            if len(parts) >= 2:
                result["first_name"] = parts[0]
                result["last_name"] = parts[-1]
            elif len(parts) == 1:
                result["first_name"] = parts[0]
        return result

    # 4. If nothing matches, return the empty dictionary
    return result









#def filter_customer(df: pd.DataFrame, email: str | None, first_name: str | None, last_name: str | None, domain: str | None) -> pd.DataFrame:
#    out = df.copy()
#    if email:
        # exact match on email (case-insensitive)
#        out = out[out["Customer Email"].str.lower() == email.strip().lower()]
#    elif domain:
#        out = out[out["Customer Email"].str.contains(domain.strip(), case=False, na=False)]
#    else:
#        if first_name:
#            out = out[out["Customer First Name"].str.lower() == first_name.strip().lower()]
#        if last_name:
#            out = out[out["Customer Last Name"].str.lower() == last_name.strip().lower()]
#    return out




def filter_customer(
    df: pd.DataFrame,
    email: str | None,
    first_name: str | None,
    last_name: str | None,
    domain: str | None,
    query: str | None  # New parameter for product search
) -> pd.DataFrame:

    out = df.copy()

    # --- Step 1: Original Customer Filtering Logic ---
    if email:
        # Exact match on email (case-insensitive)
        out = out[out["Customer Email"].str.lower() == email.strip().lower()]
    elif domain:
        out = out[out["Customer Email"].str.contains(domain.strip(), case=False, na=False)]
    else:
        if first_name:
            out = out[out["Customer First Name"].str.lower() == first_name.strip().lower()]
        if last_name:
            out = out[out["Customer Last Name"].str.lower() == last_name.strip().lower()]

    # --- Step 2: New Conditional Product Name Filtering Logic ---
    if query and not out.empty:
        # Prepare the query by splitting it into a set of unique, lowercase words
        query_words = set(query.strip().lower().split())

        # Get all unique words present in the ProductName column of the current results
        # .dropna() handles cases where ProductName might be empty
        # .str.split(expand=True).stack() is a robust way to get all words
        all_product_words = set(
            out['Product Name'].dropna().str.lower().str.split(expand=True).stack().unique()
        )

        # Find the words that are in both the query and the product names
        matching_words = query_words.intersection(all_product_words)

        # **Only filter if there is at least one matching word**
        if matching_words:
            # Create a regex pattern to match any of the matching words (e.g., 'word1|word2')
            # re.escape handles any special characters in the words safely
            regex_pattern = '|'.join(re.escape(word) for word in matching_words)
            out = out[out['Product Name'].str.contains(regex_pattern, case=False, na=False)]

    return out


import re
from decimal import Decimal, InvalidOperation

def _to_decimal(x):
    if x is None: 
        return None
    s = str(x)
    # keep digits, dots, and minus sign (strip $ and commas etc.)
    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return Decimal(s) if s else None
    except InvalidOperation:
        return None

def _to_int(x):
    try:
        return int(str(x).strip())
    except Exception:
        return None

def compact_orders_for_llm(filtered: pd.DataFrame) -> dict:
    """
    Build a compact, LLM-friendly JSON with brand and line_total per item.
    Brand preference: Supplier Name -> Manufacturer Name (fallback).
    """
    if filtered.empty:
        return {"customer": None, "orders": []}

    f = filtered.fillna("")
    first = f.iloc[0]
    customer = {
        "first_name": first.get("Customer First Name", ""),
        "last_name": first.get("Customer Last Name", ""),
        "email": first.get("Customer Email", ""),
        "phone": first.get("Customer Phone", ""),
        "company": first.get("Company Name", ""),
    }

    orders = []
    for order_id, group in f.groupby("Order ID", dropna=False):
        g0 = group.iloc[0]
        items = []
        for _, r in group.iterrows():
            unit_price = _to_decimal(r.get("Item Product Unit Price", ""))
            qty = _to_int(r.get("Quantity", ""))
            subtotal = _to_decimal(r.get("Item Subtotal", ""))

            supplier = r.get("Supplier Name", "") or None
            link = r.get("links", "") or None
            category = r.get("Category", "") or None
            items.append({
                "product_name": r.get("Product Name", ""),
                "supplier": supplier or "",
                "sku": r.get("Product SKU", ""),
                "manufacturer": r.get("Manufacturer Name", ""),
                "color": r.get("Product Color", ""),
                "size": r.get("Product Size", ""),
                "quantity": qty if qty is not None else r.get("Quantity", ""),
                "unit_price": str(unit_price) if unit_price is not None else r.get("Item Product Unit Price", ""),
                "subtotal": str(subtotal) if subtotal is not None else r.get("Item Subtotal", ""),     
                "category": category,
                "link": link
            })
        orders.append({
            "order_id": order_id,
            "status": g0.get("Order Status", ""),
            "date_ordered": g0.get("Date Ordered", ""),
            "date_completed": g0.get("Date Completed", ""),
            "delivery_method": g0.get("Delivery Method", ""),
            "tracking_numbers": g0.get("Tracking Number(s)", ""),
            "totals": {
                "subtotal": g0.get("Order Subtotal", ""),
                "delivery_total": g0.get("Delivery Total", ""),
                "tax": g0.get("Order Sales Tax", ""),
                "discount": g0.get("Discount Total", ""),
                "order_total": g0.get("Order Total", "")
            },
            "shipping": {
                "name": f"{g0.get('Shipping First Name','')} {g0.get('Shipping Last Name','')}".strip(),
                "company": g0.get("Shipping Company Name", ""),
                "street1": g0.get("Shipping Street 1", ""),
                "street2": g0.get("Shipping Street 2", ""),
                "city": g0.get("Shipping City", ""),
                "state": g0.get("Shipping State", ""),
                "zip": g0.get("Shipping Zip", ""),
                "country": g0.get("Shipping Country", ""),
                "phone": g0.get("Shipping Phone", "")
            },
            "items": items
        })

    return {"customer": customer, "orders": orders}




def get_gemini_summary(user_query: str, compact_json: dict) -> str:
    system_prompt = """
{
  "instructions": {
    "role": "assistant",
    "purpose": "Answer customer order inquiries precisely and concisely.",
    "inputs": {
      "data": "Compact JSON for a SINGLE customer's orders (already filtered).",
      "query": "User's request (may be a follow-up)."
    },
    "rules": {
      "data_scope": [
        "Work ONLY with `data`.",
        "Do NOT search the web.",
        "Do NOT fabricate links."
      ],
      "semantic_filtering": [
          "If the user specifies a product type (e.g., 'bottles'), return ONLY items that semantically match that category.",
          "Eliminate irrelevant items (e.g., Sweatshirts if user asked for bottles).",
          "Err on the side of precision: exclude items that clearly do not fit the requested product type."
    ],
    "performance": [
      "Perform semantic filtering quickly without over-explaining.",
      "Do not hallucinate new products; only use items from `data`."
    ]
      "order_selection": [
        "If user references a specific order id, focus only on that order.",
        "If the user asks for past orders, return ALL past orders from `data`.",
        "If the user asks for a specific product (e.g., bottles), only return orders where Product Name matches. If Product Name did not match, look at category if it exists."
      ],
      "item_display_format": [
        "For each item in any order, list details in this exact order:",
        "1) Link (if present), product name as markdown link.",
        "2) Product Name.",
        "3) Supplier Name.",
        "4) Product SKU."
        "4) Product Size.",
        "5) Quantity.",
        "6) Item Product Unit Price.",
        "7) Item Subtotal."
      ]
    },
    "style": {
      "formatting": [
        "Use Markdown bullets.",
        "Use bold where helpful.",
        "No raw JSON.",
        "No tables."
      ]
    }
  }
}

"""

    # Helpful sort: most-recent-first
    try:
        orders = compact_json.get("orders", [])
        def safe_key(o): return o.get("date_ordered") or ""
        compact_json["orders"] = sorted(orders, key=safe_key, reverse=True)
    except Exception:
        pass

    payload = {"query": user_query, "data": compact_json}
    try:
        #resp = model.models.generate_content(
        #    model="gemini-2.5-flash",
        #    contents=[system_prompt, json.dumps(payload, ensure_ascii=False)],
        #    config={"response_mime_type": "text/plain"}
        #)
        resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            response_format={"type": "text"}  # plain text response
        )
        return resp.choices[0].message.content.strip()
        #return resp.text.strip()
    except Exception as e:
        st.error(f"Error generating AI response: {e}")
        return "Sorry, I couldn't generate a response right now."



# --- UI ---
st.title("Past Customer Orders Chatbot")
#st.caption("Ask about a customer's orders by name or email. The AI will summarize the essentials and include product links when available.")


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("e.g., \"'Steven Henderson' wants to reorder a T-shirt\" or \"john@acme.com show past orders\"  \"@acme wants to reorder bottles\"")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        ctx = st.session_state.get("current_context")
        ident = extract_customer_info(prompt)
        has_identifier = any([ident.get("email"), ident.get("first_name"), ident.get("last_name"), ident.get("domain")])

        if has_identifier:
            filtered = filter_customer(
                df,
                ident.get("email"),
                ident.get("first_name"),
                ident.get("last_name"),
                ident.get("domain"),
                prompt
            )

            if filtered.empty:
                response = "I couldn’t find a matching customer. Start with an email (e.g., john@acme.com), a domain (e.g., @acme.com), or a quoted full name (e.g., 'Steven Henderson')."

            # CASE 1: Query starts with @domain and contains 'past orders' → show table
            elif ident.get("domain") and "past orders" in prompt.lower():
                st.session_state.current_context = {"customer": None, "orders": [], "df": filtered.copy()}
                cols_to_show = [
                    "Date Ordered",
                    "Order ID",
                    "Customer First Name",
                    "Customer Last Name",
                    "Product Name",
                    "Supplier Name",
                    "Product SKU",
                    "Product Size",
                    "Quantity",
                    "Item Product Unit Price",
                    "Item Subtotal",
                    "links"
                ]
                display_df = filtered[cols_to_show].fillna("")
                st.markdown(f"### All Past Orders for @{ident['domain']}")
                st.dataframe(display_df, use_container_width=True)
                response = f"I’ve listed all past orders for @{ident['domain']} above."

            # CASE 2: @domain but not just 'past orders' → go to AI (e.g. reorder bottles)
            else:
                compact = compact_orders_for_llm(filtered)
                st.session_state.current_context = {
                    "customer": compact["customer"],
                    "orders": compact["orders"],
                    "df": filtered.copy()
                }
                response = get_gemini_summary(
                    prompt,
                    {"customer": compact["customer"], "orders": compact["orders"]}
                )
        else:
            if not ctx:
                response = ("Please begin with an email, domain (e.g., `@acme.com`), "
                            "or a quoted full name (e.g., 'Steven Henderson').")
            else:
                response = get_gemini_summary(prompt, {"customer": ctx["customer"], "orders": ctx["orders"]})

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

