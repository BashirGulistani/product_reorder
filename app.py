
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

# --- Styling ---
st.title("Past Customer Orders Chatbot")
st.caption("Ask about a customer's orders by name or email. The AI will summarize the essentials and include product links when available.")

# Top-centered input styled like a large rounded search bar
st.markdown("""
<style>
/* Center the input area and make it big & pill-shaped */
.hero-wrap { display:flex; justify-content:center; margin: 1.25rem 0 0.5rem 0; }
.hero { width:min(900px, 92vw); }
.hero .stTextInput>div>div {
    border-radius: 999px !important;
    border: 1px solid #d9d9d9 !important;
    padding: 0.25rem 1.25rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.hero .stTextInput input {
    height: 56px !important;
    font-size: 1.05rem !important;
}
.hero .stButton>button {
    height: 48px; border-radius: 999px; padding: 0 1.25rem;
}
</style>
""", unsafe_allow_html=True)

# Keep a place to show validation/errors nicely
feedback = st.empty()

# Centered input form at the top.
st.markdown('<div class="hero-wrap"><div class="hero">', unsafe_allow_html=True)
with st.form("hero_query", clear_on_submit=False):
    top_prompt = st.text_input(
        label="Ask about past orders",
        placeholder="e.g., \"'Steven Henderson' wants to reorder a T-shirt\" or \"john@acme.com show past orders\"",
        label_visibility="collapsed",
        key="top_query",
    )
    cols = st.columns([6, 1.5])
    with cols[1]:
        submitted = st.form_submit_button("Ask")
st.markdown('</div></div>', unsafe_allow_html=True)

# Process submit BEFORE rendering chat history, so chat appears below the input
if submitted and top_prompt.strip():
    with st.chat_message("user"):
        st.markdown(top_prompt)
    st.session_state.messages.append({"role": "user", "content": top_prompt})

    with st.spinner("Thinking..."):
        ctx = st.session_state.get("current_context")
        ident = extract_customer_info(top_prompt)
        has_identifier = any([ident.get("email"), ident.get("first_name"), ident.get("last_name")])

        if has_identifier:
            filtered = filter_customer(df, ident.get("email"), ident.get("first_name"), ident.get("last_name"))
            if filtered.empty:
                response = ("I couldn’t find a matching customer. Start with an email (e.g., john@acme.com) "
                            "or a quoted full name (e.g., 'Steven Henderson').")
            else:
                compact = compact_orders_for_llm(filtered)
                st.session_state.current_context = {
                    "customer": compact["customer"],
                    "orders": compact["orders"],
                    "df": filtered.copy()
                }
                response = get_gemini_summary(top_prompt, {"customer": compact["customer"], "orders": compact["orders"]})
        else:
            if not ctx:
                response = ("Please begin with an email or a quoted full name so I know the customer.\n"
                            "Examples:\n- `john@acme.com show past orders`\n- `'Steven Henderson' wants to reorder a T-shirt`")
            else:
                response = get_gemini_summary(top_prompt, {"customer": ctx["customer"], "orders": ctx["orders"]})

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
elif submitted:
    feedback.warning("Please type a question first.")

st.markdown("----")

# Conversation history renders BELOW the centered input
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



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









def filter_customer(df: pd.DataFrame, email: str | None, first_name: str | None, last_name: str | None) -> pd.DataFrame:
    out = df.copy()
    if email:
        # exact match on email (case-insensitive)
        out = out[out["Customer Email"].str.lower() == email.strip().lower()]
    else:
        if first_name:
            out = out[out["Customer First Name"].str.contains(first_name.strip(), case=False, na=False)]
        if last_name:
            out = out[out["Customer Last Name"].str.contains(last_name.strip(), case=False, na=False)]
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
    line_total = quantity * unit_price (precomputed).
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
            line_total = (unit_price * qty) if (unit_price is not None and qty is not None) else None

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
                "line_total": str(line_total) if line_total is not None else None,
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
  4) Size;
  5) Quantity;
  6) Line Total (use `item.line_total` if present; otherwise omit).

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
            model="gemini-2.5-flash",
            contents=[system_prompt, json.dumps(payload, ensure_ascii=False)],
            config={"response_mime_type": "text/plain"}
        )
        return resp.text.strip()
    except Exception as e:
        st.error(f"Error generating AI response: {e}")
        return "Sorry, I couldn't generate a response right now."



# --- UI ---
st.title("Past Customer Orders Chatbot")
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

