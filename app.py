
import streamlit as st
import pandas as pd
import json
from google import genai

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Order Chatbot",
    page_icon="🤖",
    layout="wide",
)



# --- Gemini client ---
try:
    model = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("Could not initialize Gemini client. Please set GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

# --- Styling ---
st.markdown("""
<style>
    * { color: black !important; }
    .stApp { background-color: #f0f2f6; }
    .main .block-container { padding: 2rem 5rem; }
    .st-emotion-cache-1r4qj8v, .st-emotion-cache-1v0mbdj {
        border: 1px solid #e6e6e6; border-radius: 10px; padding: 20px; background: #fff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: 0.3s;
    }
    .st-emotion-cache-1r4qj8v:hover, .st-emotion-cache-1v0mbdj:hover {
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .stButton>button {
        border-radius: 8px; border: 1px solid transparent; padding: 0.6em 1.2em;
        font-size: 1em; font-weight: 500; background-color: #1a73e8; color: white !important;
        cursor: pointer; transition: border-color 0.25s;
    }
    .stButton>button:hover { border-color: #1a73e8; background: #ffffff; color: black !important; }
    .st-chat-message { border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Data ---
@st.cache_data
def load_sample_data() -> pd.DataFrame:
    # Read as strings to avoid unintended numeric coercions (tracking, zip, SKUs, etc.)
    df = pd.read_csv("order_data.csv", dtype=str, keep_default_na=True, na_values=["", "NA", "NaN", "nan"])
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
from typing import Optional, Dict

_EMAIL_START_RE = re.compile(r"^\s*([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b", re.I)
_QUOTED_NAME_START_RE = re.compile(r"^\s*'([^']+?)'")

def extract_customer_info(query: str) -> Dict[str, Optional[str]]:
    # Leading email wins
    m_email = _EMAIL_START_RE.match(query.strip())
    if m_email:
        return {"email": m_email.group(1).lower(), "first_name": None, "last_name": None}

    # Leading 'Full Name'
    m_name = _QUOTED_NAME_START_RE.match(query.strip())
    if m_name:
        name = m_name.group(1).strip()
        if "," in name:  # "Last, First"
            last, first, *_ = [p.strip() for p in name.split(",")]
            return {"email": None, "first_name": first or None, "last_name": last or None}
        parts = [p for p in name.split() if p]
        if len(parts) >= 2:
            return {"email": None, "first_name": parts[0], "last_name": parts[-1]}
        elif len(parts) == 1:
            return {"email": None, "first_name": parts[0], "last_name": None}

    return {"email": None, "first_name": None, "last_name": None}









def filter_customer(df: pd.DataFrame, info: dict) -> pd.DataFrame:
    out = df.copy()
    if info.get("email"):
        out = out[out["Customer Email"].str.lower() == info["email"].lower()]
    if info.get("first_name"):
        out = out[out["Customer First Name"].str.contains(info["first_name"], case=False, na=False)]
    if info.get("last_name"):
        out = out[out["Customer Last Name"].str.contains(info["last_name"], case=False, na=False)]
    return out

def compact_orders_for_llm(df: pd.DataFrame) -> dict:
    """
    Build a concise JSON snapshot. No raw CSV dump.
    Group by Order ID and summarize key fields and line items (with links if present).
    """
    # Columns we might use (handle missing gracefully)
    def g(col, row):
        return (row.get(col) if isinstance(row, dict) else row[col]) if col in df.columns else None

    # Build per-order structure
    orders = []
    if df.empty:
        return {"customer": None, "orders": []}

    # Use first non-null customer-level fields
    first = df.iloc[0].fillna("")
    customer = {
        "first_name": first.get("Customer First Name", ""),
        "last_name": first.get("Customer Last Name", ""),
        "email": first.get("Customer Email", ""),
        "phone": first.get("Customer Phone", ""),
        "company": first.get("Company Name", ""),
    }

    for order_id, group in df.groupby("Order ID", dropna=False):
        g0 = group.iloc[0].fillna("")
        items = []
        for _, r in group.fillna("").iterrows():
            items.append({
                "product_name": r.get("Product Name", ""),
                "sku": r.get("Product SKU", ""),
                "manufacturer": r.get("Manufacturer Name", ""),
                "color": r.get("Product Color", ""),
                "size": r.get("Product Size", ""),
                "quantity": r.get("Quantity", ""),
                "unit_price": r.get("Item Product Unit Price", ""),
                "link": r.get("links", "") or None  # keep None if blank
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
                "order_total": g0.get("Order Total", ""),
                "payment_total": g0.get("Payment Total", "")
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

You will receive:
- `data`: a compact JSON of a SINGLE customer's orders (already filtered by the app).
- `query`: the user's request (may be a follow-up like "details for order 1074026", "show tracking", "reorder", etc.).

Your job:
- Work ONLY with the provided JSON. Do NOT search the web. Do NOT fabricate links.
- Understand follow-ups naturally (order IDs, tracking, items, totals, shipping, reorder suggestions).
- Prefer the most recent order when the user does not specify.
- If an order id is referenced, focus on that order.
- Summaries should be concise (≈150 words), using Markdown bullets when useful.
- If `links` exist on items, include them inline. If not, omit.
- End with: "If you want more details (like prices, shipping, or tracking), please let me know."
"""

    # Sort orders by Date Ordered desc to help the model pick "most recent"
    try:
        orders = compact_json.get("orders", [])
        def safe_key(o): return o.get("date_ordered") or ""
        compact_json["orders"] = sorted(orders, key=safe_key, reverse=True)
    except Exception:
        pass

    payload = {"query": user_query, "data": compact_json}
    try:
        resp = model.models.generate_content(
            model="gemini-2.5-flash",
            contents=[system_prompt, json.dumps(payload, ensure_ascii=False)],
            config={"response_mime_type": "text/plain"}
        )
        return resp.text.strip()
    except Exception as e:
        st.error(f"Error generating AI response: {e}")
        return "Sorry, I couldn't generate a response right now."


# --- UI ---
st.title("Customer Order Smart Chatbot")
st.caption("Ask about a customer's orders by name or email. The AI will summarize the essentials and include product links when available.")


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
        has_identifier = any([ident.get("email"), ident.get("first_name"), ident.get("last_name")])

        if has_identifier:
            # Filter dataset once per customer selection
            filtered = filter_customer(df, ident.get("email"), ident.get("first_name"), ident.get("last_name"))
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

