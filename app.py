
import streamlit as st
import pandas as pd
import json
from google import genai

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Order Smart Chatbot",
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

def detect_followup_intent(msg: str) -> dict:
    """
    Returns one of:
      {"type": "details", "order_id": "123"}  # drill into a specific order
      {"type": "show", "what": "tracking|prices|items|shipping"}  # generic details
      {"type": "reorder"}  # reorder intent
      {"type": "switch"}  # indicates new customer present
      {"type": "unknown"}
    """
    m = re.search(r"\border\s*#?\s*([A-Za-z0-9\-]+)\b", msg, re.I)
    if m:
        return {"type": "details", "order_id": m.group(1)}

    if re.search(r"\b(track|tracking|status)\b", msg, re.I):
        return {"type": "show", "what": "tracking"}
    if re.search(r"\b(price|total|subtotal|tax|discount)\b", msg, re.I):
        return {"type": "show", "what": "prices"}
    if re.search(r"\b(items?|line items?|products?)\b", msg, re.I):
        return {"type": "show", "what": "items"}
    if re.search(r"\b(ship|address|delivery)\b", msg, re.I):
        return {"type": "show", "what": "shipping"}
    if re.search(r"\bre-?order|order again|same as last time\b", msg, re.I):
        return {"type": "reorder"}

    return {"type": "unknown"}


from textwrap import dedent

def format_order_brief(o: dict) -> str:
    lines = []
    lines.append(f"**Order {o.get('order_id','')}** — {o.get('status','')}")
    if o.get("date_ordered"):
        lines.append(f"- Date Ordered: {o['date_ordered']}")
    if o.get("date_completed"):
        lines.append(f"- Date Completed: {o['date_completed']}")
    t = o.get("totals") or {}
    if any(t.values()):
        lines.append(f"- Totals: Subtotal {t.get('subtotal','')}, Delivery {t.get('delivery_total','')}, Tax {t.get('tax','')}, Discount {t.get('discount','')}, **Order Total {t.get('order_total','')}**")
    return "\n".join(lines)

def format_items(o: dict) -> str:
    lines = ["**Items**"]
    for it in (o.get("items") or []):
        row = f"- {it.get('product_name','(Unnamed)')} x{it.get('quantity','')}"
        if it.get("unit_price"):
            row += f" @ {it['unit_price']}"
        if it.get("sku"):
            row += f" (SKU: {it['sku']})"
        if it.get("link"):
            row += f" — [link]({it['link']})"
        lines.append(row)
    return "\n".join(lines)

def format_tracking(o: dict) -> str:
    tr = (o.get("tracking_numbers") or "").strip()
    if tr:
        return f"**Tracking**: {tr}"
    return "_No tracking numbers recorded for this order._"

def format_shipping(o: dict) -> str:
    s = o.get("shipping") or {}
    addr = ", ".join([v for v in [s.get("street1",""), s.get("street2",""), s.get("city",""), s.get("state",""), s.get("zip",""), s.get("country","")] if v])
    name = s.get("name","").strip() or "(Name not set)"
    out = [f"**Ship To**: {name}"]
    if s.get("company"): out.append(f"- Company: {s['company']}")
    if addr: out.append(f"- Address: {addr}")
    if s.get("phone"): out.append(f"- Phone: {s['phone']}")
    return "\n".join(out)




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
_QUOTED_NAME_START_RE = re.compile(r"^\s*'([^']+?)'")

def extract_customer_info(query: str) -> Dict[str, Optional[str]]:
    """
    Deterministic extractor that reads the *beginning* of the query.
    Supported forms at the start:
      1) EMAIL            -> e.g., "jane@company.com wants to reorder..."
      2) 'FIRST LAST'     -> e.g., "'Steven Henderson' wants to reorder..."

    Returns:
        {
          "email": email or None,
          "first_name": first name or None,
          "last_name": last name or None
        }
    """
    if not isinstance(query, str):
        return {"email": None, "first_name": None, "last_name": None}

    s = query.strip()

    # 1) Leading email wins
    m_email = _EMAIL_START_RE.match(s)
    if m_email:
        return {"email": m_email.group(1).lower(), "first_name": None, "last_name": None}

    # 2) Leading single-quoted full name
    m_name = _QUOTED_NAME_START_RE.match(s)
    if m_name:
        name = m_name.group(1).strip()

        # Optional: handle "Last, First" if user types it
        if "," in name:
            last, first, *rest = [p.strip() for p in name.split(",")]
            # Ignore middle names if present
            return {"email": None, "first_name": first or None, "last_name": last or None}

        # Default: "First [Middle ...] Last"
        parts = [p for p in name.split() if p]
        if len(parts) >= 2:
            first = parts[0]
            last = parts[-1]
            return {"email": None, "first_name": first, "last_name": last}
        elif len(parts) == 1:
            # In case someone provides only one token in quotes
            return {"email": None, "first_name": parts[0], "last_name": None}

    # If nothing valid at the start, return nulls
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
    """
    Ask Gemini for a concise natural-language summary.
    It should *not* regurgitate the raw data; it should summarize.
    """
    system_prompt = """
You are a precise assistant for customer order inquiries.

Write a short, friendly summary of the customer's past orders.
Rules:
- DO NOT dump raw tables or CSV.
- Summarize in <= 150 words when possible.
- Mention what the customer ordered (product names), quantities, and key dates.
- Include any item-level link URLs **only if present** in the input (no searching, no fabricated links).
- If there are multiple orders, summarize the most recent first (by Date Ordered if present).
- End with: "If you want more details (like prices, shipping, or tracking), please let me know."
- Use clear Markdown bullets when helpful.
"""

    # Sort orders by date desc if dates exist (light Python sort to help the model)
    try:
        orders = compact_json.get("orders", [])
        def safe_key(o):
            return o.get("date_ordered") or ""
        compact_json["orders"] = sorted(orders, key=safe_key, reverse=True)
    except Exception:
        pass

    user_payload = {
        "query": user_query,
        "data": compact_json
    }

    try:
        resp = model.models.generate_content(
            model="gemini-2.5-flash",
            contents=[system_prompt, json.dumps(user_payload, ensure_ascii=False)],
            config={"response_mime_type": "text/plain"}
        )
        return resp.text.strip()
    except Exception as e:
        st.error(f"Error generating AI summary: {e}")
        return "Sorry, I couldn't generate a summary right now."

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
prompt = st.chat_input("e.g., “Show past orders for John Doe” or “john@example.com wants to reorder hoodies”")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("🧠 Thinking..."):
        # Try to see if the user mentioned a *new* customer
        info = extract_customer_info(prompt)
        is_new_customer = any([info.get("email"), info.get("first_name"), info.get("last_name")])

        if is_new_customer:
            # Switch context to the new customer
            filtered = filter_customer(df, info)
            if filtered.empty:
                response = ("I couldn’t find a matching customer. "
                            "Try including an email address or full name (e.g., “john@example.com” or “John Doe”).")
            else:
                compact = compact_orders_for_llm(filtered)
                st.session_state.current_context = {"customer": compact["customer"], "orders": compact["orders"], "df": filtered.copy()}
                response = get_gemini_summary(prompt, {"customer": compact["customer"], "orders": compact["orders"]})

        else:
            # No new customer specified → treat as follow-up if we have context
            ctx = st.session_state.current_context
            if not ctx:
                response = ("I don’t have a customer in context yet. "
                            "Please mention a customer name or email to begin.")
            else:
                intent = detect_followup_intent(prompt)
                orders = ctx["orders"] or []

                if intent["type"] == "details":
                    # Try to find that order
                    oid = intent["order_id"]
                    match = next((o for o in orders if str(o.get("order_id","")).lower() == oid.lower()), None)
                    if match:
                        parts = [format_order_brief(match), "", format_items(match), "", format_shipping(match), "", format_tracking(match)]
                        response = "\n".join([p for p in parts if p])
                    else:
                        response = f"I couldn’t find order **{oid}** for this customer. You can ask me for 'recent orders' or specify another order number."

                elif intent["type"] == "show":
                    # Show generic slices for the most recent order
                    if not orders:
                        response = "There are no orders on record for this customer."
                    else:
                        o = orders[0]  # most recent (we sorted in get_gemini_summary; if you want, sort here too)
                        if intent["what"] == "tracking":
                            response = "\n".join([format_order_brief(o), "", format_tracking(o)])
                        elif intent["what"] == "prices":
                            response = format_order_brief(o)
                        elif intent["what"] == "items":
                            response = "\n".join([format_order_brief(o), "", format_items(o)])
                        elif intent["what"] == "shipping":
                            response = "\n".join([format_order_brief(o), "", format_shipping(o)])
                        else:
                            response = get_gemini_summary(prompt, {"customer": ctx["customer"], "orders": orders})

                elif intent["type"] == "reorder":
                    # Simple reorder suggestion using the most recent order's items
                    if not orders:
                        response = "No past items found to reorder."
                    else:
                        o = orders[0]
                        lines = ["Here’s a quick reorder from the most recent order:"]
                        for it in (o.get("items") or []):
                            line = f"- {it.get('product_name','(Unnamed)')} x{it.get('quantity','')}"
                            if it.get("link"):
                                line += f" — [link]({it['link']})"
                            lines.append(line)
                        lines.append("\nIf you want to change quantities or pick a different order, let me know.")
                        response = "\n".join(lines)

                else:
                    # Unknown follow-up → let LLM summarize within current context
                    response = get_gemini_summary(prompt, {"customer": ctx["customer"], "orders": orders})

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
