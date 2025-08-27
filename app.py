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

# --- Helpers ---
def extract_customer_info(query: str) -> dict:
    """
    Use LLM to extract email / first / last in robust JSON.
    """
    prompt = f"""
You will extract a customer's identifier from the user's query.

Return STRICT JSON with keys exactly:
{{
  "email": string or null,
  "first_name": string or null,
  "last_name": string or null
}}

Rules:
- If a full name is present (e.g., "John Q. Doe"), split into first_name="John" and last_name="Doe" (ignore middle).
- If only one token like "John" appears, set first_name="John", last_name=null.
- If an email appears, set email to the lowercase email.
- If nothing identifiable, return all nulls.

Query: {query}
"""
    try:
        resp = model.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config={"response_mime_type": "application/json"}
        )
        data = json.loads(resp.text)
        # Safety net keys
        return {
            "email": (data.get("email") or None),
            "first_name": (data.get("first_name") or None),
            "last_name": (data.get("last_name") or None),
        }
    except Exception as e:
        st.warning(f"Extractor fallback (couldn't parse JSON): {e}")
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
st.title("🤖 Customer Order Smart Chatbot")
st.caption("Ask about a customer's orders by name or email. The AI will summarize the essentials and include product links when available.")

# (Optional) small peek of data shape so users know it's loaded — no raw dump in chat
with st.expander("Dataset info"):
    st.write(f"Rows: {len(df):,}  •  Columns: {len(df.columns)}")
    st.write("Key columns used:", [
        "Order ID","Order Status","Date Ordered","Date Completed",
        "Customer First Name","Customer Last Name","Customer Email","Customer Phone",
        "Product Name","Product SKU","Quantity","Item Product Unit Price","links"
    ])

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

    with st.spinner("🧠 Summarizing the customer’s orders..."):
        # 1) Extract who to search for
        info = extract_customer_info(prompt)

        # 2) Filter rows for that customer (email wins if present; names are partial match)
        filtered = filter_customer(df, info)

        if filtered.empty:
            response = ("I couldn’t find a matching customer. "
                        "Try including an email address or full name (e.g., “john@example.com” or “John Doe”).")
        else:
            # 3) Build compact JSON snapshot (no web searches; keep links only if present in data)
            compact = compact_orders_for_llm(filtered)

            # 4) Ask Gemini for a short human summary
            response = get_gemini_summary(prompt, compact)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
