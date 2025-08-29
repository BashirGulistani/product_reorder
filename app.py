
import streamlit as st
import pandas as pd
import json
from google import genai

# --- Page Configuration ---
st.set_page_config(
    page_title="Past Customer Orders Chatbot",
    page_icon="🤖",
    layout="wide",
)



# --- Gemini client ---
try:
    model = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
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

    /* New styles for the chat input box */
    /* Targets the container to control width and centering */
    div[data-testid="stChatInput"] {
        width: 70% !important;
        margin: 0 auto !important; /* Centers the reduced-width input */
        background-color: transparent;
    }

    /* Targets the actual text area for height and styling */
    div[data-testid="stChatInput"] textarea {
        height: 100px; /* Increase the height of the text area */
        border-radius: 30px;
        border: 3px solid #d9d9d9;
        padding-top: 0.8rem;
        padding-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)



# --- Data ---
@st.cache_data
def load_sample_data() -> pd.DataFrame:
    # Read as strings to avoid unintended numeric coercions (tracking, zip, SKUs, etc.)
    df = pd.read_csv("ALLORDERS_FNAL.csv", dtype=str, keep_default_na=True, na_values=["", "NA", "NaN", "nan"])
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









def filter_customer(df: pd.DataFrame, email: str | None, first_name: str | None, last_name: str | None, domain: str | None) -> pd.DataFrame:
    out = df.copy()
    if email:
        # exact match on email (case-insensitive)
        out = out[out["Customer Email"].str.lower() == email.strip().lower()]
    elif domain:
        out = out[out["Customer Email"].str.contains(domain.strip(), case=False, na=False)]
    else:
        if first_name:
            out = out[out["Customer First Name"].str.lower() == first_name.strip().lower()]
        if last_name:
            out = out[out["Customer Last Name"].str.lower() == last_name.strip().lower()]
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

            brand = r.get("Supplier Name", "") or r.get("Manufacturer Name", "")
            link = r.get("links", "") or None
            items.append({
                "product_name": r.get("Product Name", ""),
                "brand": brand or "",
                "sku": r.get("Product SKU", ""),
                "manufacturer": r.get("Manufacturer Name", ""),
                "color": r.get("Product Color", ""),
                "size": r.get("Product Size", ""),
                "quantity": qty if qty is not None else r.get("Quantity", ""),
                "unit_price": str(unit_price) if unit_price is not None else r.get("Item Product Unit Price", ""),
                "unit_price": str(subtotal) if subtotal is not None else r.get("Item Subtotal", ""),
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
You are a precise assistant for customer order inquiries.

INPUTS YOU GET:
- `data`: compact JSON for a SINGLE customer's orders (already filtered).
- `query`: the user's request (may be a follow-up like "details for order 1074026", "show tracking", "reorder", etc.).

STRICT RULES:
- Work ONLY with `data`. Do NOT search the web. Do NOT fabricate links.
- If the user references a specific order id, focus on that order; otherwise prefer the most recent order (by Date Ordered if available).
- When listing items for any order, the FIRST thing shown must be, per item and in this exact order:
  1) Link (if present), shown by making the product name a markdown link;
  2) Product Name;
  3) Product Brand (use `item.brand`; if missing, omit "Brand:");
  4) Product Size;
  5) Quantity;
  6) Item Product Unit Price;
  7) Item Subtotal;

RECOMMENDED ITEM LINE FORMAT (Markdown list):
- [Product Name](link) — Brand: BRAND • Qty: Q • Line total: $T
- If `link` is null, do not create a link; just show: Product Name — Brand: BRAND • Qty: Q • Line total: $T
- If BRAND or Line total are missing, omit those segments but keep the rest.

STRUCTURE YOUR ANSWER:
1) A short order header per relevant order (e.g., **Order 12345** — Status; Date Ordered).
2) Then an **Items** section that lists each item using the format above (one bullet per item).
3) Then any requested extras (tracking, totals, shipping) as concise bullets.
4) End with: "If you want more details (like prices, shipping, or tracking), please let me know."

LENGTH & STYLE:
- Be concise (≈150 words when possible). Use Markdown bullets and bold where helpful. No raw JSON or tables.
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
        resp = model.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[system_prompt, json.dumps(payload, ensure_ascii=False)],
            config={"response_mime_type": "text/plain"}
        )
        return resp.text.strip()
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
prompt = st.chat_input("e.g., \"'Steven Henderson' wants to reorder a T-shirt\" or \"john@acme.com show past orders\"")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        ctx = st.session_state.get("current_context")
        ident = extract_customer_info(prompt)
        has_identifier = any([ident.get("email"), ident.get("first_name"), ident.get("last_name"), ident.get("domain")])

        if has_identifier:
            # Filter dataset once per customer selection
            filtered = filter_customer(df, ident.get("email"), ident.get("first_name"), ident.get("last_name"), ident.get("domain"))
            if filtered.empty:
                response = "I couldn’t find a matching customer. Start with an email (e.g., john@acme.com) or a quoted full name (e.g., 'Steven Henderson')."
            else:
                compact = compact_orders_for_llm(filtered)
                st.session_state.current_context = {"customer": compact["customer"], "orders": compact["orders"], "df": filtered.copy()}
                response = get_gemini_summary(prompt, {"customer": compact["customer"], "orders": compact["orders"]})
        else:
            if not ctx:
                response = ("Please begin with an email or a quoted full name so I know the customer.\n"
                            "Examples:\n- `john@acme.com show past orders`\n- `'Steven Henderson' wants to reorder a T-shirt`")
            else:
                # Pure LLM follow-up using saved (already filtered) context
                response = get_gemini_summary(prompt, {"customer": ctx["customer"], "orders": ctx["orders"]})

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

