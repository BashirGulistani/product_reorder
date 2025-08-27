import streamlit as st
import pandas as pd
import json
import re
from textwrap import dedent
from google import genai

# --- Page Configuration ---
st.set_page_config(page_title="Customer Order Smart Chatbot", page_icon="🤖", layout="wide")

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
    .pill { display:inline-block; padding:4px 10px; border-radius:999px; background:#eef3ff; border:1px solid #c6d4ff; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

# --- Data ---
@st.cache_data
def load_sample_data() -> pd.DataFrame:
    df = pd.read_csv("order_data.csv", dtype=str, keep_default_na=True, na_values=["", "NA", "NaN", "nan"])
    df.columns = [c.strip() for c in df.columns]
    return df

try:
    df = load_sample_data()
except FileNotFoundError:
    st.error("Sample data file 'order_data.csv' not found in the app directory.")
    st.stop()

# --- Conversation state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_context" not in st.session_state:
    st.session_state.current_context = None  # {"customer": {...}, "orders": [...], "df": filtered_df}

# --- Helpers (no customer extraction LLM) ---
def filter_customer(df: pd.DataFrame, email: str | None, first_name: str | None, last_name: str | None) -> pd.DataFrame:
    out = df.copy()
    # email exact (case-insensitive) wins
    if email:
        out = out[out["Customer Email"].str.lower() == email.strip().lower()]
    else:
        if first_name:
            out = out[out["Customer First Name"].str.contains(first_name.strip(), case=False, na=False)]
        if last_name:
            out = out[out["Customer Last Name"].str.contains(last_name.strip(), case=False, na=False)]
    return out

def compact_orders_for_llm(filtered: pd.DataFrame) -> dict:
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
            items.append({
                "product_name": r.get("Product Name", ""),
                "sku": r.get("Product SKU", ""),
                "manufacturer": r.get("Manufacturer Name", ""),
                "color": r.get("Product Color", ""),
                "size": r.get("Product Size", ""),
                "quantity": r.get("Quantity", ""),
                "unit_price": r.get("Item Product Unit Price", ""),
                "link": r.get("links", "") or None
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
Write a short, friendly summary of the customer's past orders.
Rules:
- DO NOT dump raw tables or CSV.
- Summarize in <= 150 words when possible.
- Mention what the customer ordered (product names), quantities, and key dates.
- Include item-level link URLs only if present (no searching).
- If multiple orders, summarize the most recent first (by Date Ordered if present).
- End with: "If you want more details (like prices, shipping, or tracking), please let me know."
- Use clear Markdown bullets when helpful.
"""
    try:
        orders = compact_json.get("orders", [])
        def safe_key(o): return o.get("date_ordered") or ""
        compact_json["orders"] = sorted(orders, key=safe_key, reverse=True)
    except Exception:
        pass

    user_payload = {"query": user_query, "data": compact_json}
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

def detect_followup_intent(msg: str) -> dict:
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

def format_order_brief(o: dict) -> str:
    lines = [f"**Order {o.get('order_id','')}** — {o.get('status','')}"]
    if o.get("date_ordered"): lines.append(f"- Date Ordered: {o['date_ordered']}")
    if o.get("date_completed"): lines.append(f"- Date Completed: {o['date_completed']}")
    t = o.get("totals") or {}
    if any(t.values()):
        lines.append(f"- Totals: Subtotal {t.get('subtotal','')}, Delivery {t.get('delivery_total','')}, "
                     f"Tax {t.get('tax','')}, Discount {t.get('discount','')}, **Order Total {t.get('order_total','')}**")
    return "\n".join(lines)

def format_items(o: dict) -> str:
    lines = ["**Items**"]
    for it in (o.get("items") or []):
        row = f"- {it.get('product_name','(Unnamed)')} x{it.get('quantity','')}"
        if it.get("unit_price"): row += f" @ {it['unit_price']}"
        if it.get("sku"): row += f" (SKU: {it['sku']})"
        if it.get("link"): row += f" — [link]({it['link']})"
        lines.append(row)
    return "\n".join(lines)

def format_tracking(o: dict) -> str:
    tr = (o.get("tracking_numbers") or "").strip()
    return f"**Tracking**: {tr}" if tr else "_No tracking numbers recorded for this order._"

def format_shipping(o: dict) -> str:
    s = o.get("shipping") or {}
    addr = ", ".join([v for v in [s.get("street1",""), s.get("street2",""), s.get("city",""), s.get("state",""), s.get("zip",""), s.get("country","")] if v])
    name = s.get("name","").strip() or "(Name not set)"
    out = [f"**Ship To**: {name}"]
    if s.get("company"): out.append(f"- Company: {s['company']}")
    if addr: out.append(f"- Address: {addr}")
    if s.get("phone"): out.append(f"- Phone: {s['phone']}")
    return "\n".join(out)

# --- UI ---
st.title("🤖 Customer Order Smart Chatbot")
st.caption("Select a customer via email/name, then ask what you want (e.g., 'show tracking', 'list items', 'reorder'). No web search; links shown only if present in your data.")

# Customer selector row (email wins; otherwise partial name match)
c1, c2, c3 = st.columns([2.2, 1.5, 1.5])
with c1:
    email_in = st.text_input("Customer Email (exact match)", placeholder="e.g., jane@company.com")
with c2:
    first_in = st.text_input("First Name (partial ok)", placeholder="e.g., Jane")
with c3:
    last_in = st.text_input("Last Name (partial ok)", placeholder="e.g., Doe")

# Live filter + set context
filtered = filter_customer(df, email_in.strip() or None, first_in.strip() or None, last_in.strip() or None)
if not (email_in or first_in or last_in):
    st.info("Enter an email or name to select a customer.")
else:
    if filtered.empty:
        st.warning("No matching customer found.")
    else:
        # Update context
        compact = compact_orders_for_llm(filtered)
        st.session_state.current_context = {"customer": compact["customer"], "orders": compact["orders"], "df": filtered.copy()}

        # Current customer pill + basic stats
        cust = compact["customer"] or {}
        display_name = " ".join([cust.get("first_name",""), cust.get("last_name","")]).strip() or "(no name)"
        st.markdown(f"**Current Customer:** <span class='pill'>{display_name}</span>  &nbsp;&nbsp; **Orders:** {len(compact['orders'])}", unsafe_allow_html=True)

        with st.expander("(Optional) See which rows matched", expanded=False):
            st.write(f"Matched rows: {len(filtered)}")
            st.dataframe(filtered[[
                "Order ID","Date Ordered","Order Status","Customer Email","Customer First Name","Customer Last Name","Product Name","Quantity","links"
            ]].head(50), use_container_width=True)

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat (intent only; uses current_context)
prompt = st.chat_input("Ask about the selected customer's orders (e.g., 'show items', 'order 12345', 'show tracking', 'reorder')")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("🧠 Thinking..."):
        ctx = st.session_state.current_context
        if not ctx or not (email_in or first_in or last_in) or filtered.empty:
            response = "Please select a customer first (email or name) so I know whose orders to reference."
        else:
            intent = detect_followup_intent(prompt)
            orders = ctx["orders"] or []

            if intent["type"] == "details":
                oid = intent["order_id"]
                match = next((o for o in orders if str(o.get("order_id","")).lower() == oid.lower()), None)
                if match:
                    parts = [format_order_brief(match), "", format_items(match), "", format_shipping(match), "", format_tracking(match)]
                    response = "\n".join([p for p in parts if p])
                else:
                    response = f"I couldn’t find order **{oid}** for this customer."

            elif intent["type"] == "show":
                if not orders:
                    response = "There are no orders on record for this customer."
                else:
                    o = orders[0]  # most recent
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
                if not orders:
                    response = "No past items found to reorder."
                else:
                    o = orders[0]
                    lines = ["Here’s a quick reorder from the most recent order:"]
                    for it in (o.get("items") or []):
                        line = f"- {it.get('product_name','(Unnamed)')} x{it.get('quantity','')}"
                        if it.get("link"): line += f" — [link]({it['link']})"
                        lines.append(line)
                    lines.append("\nIf you want to change quantities or pick a different order, let me know.")
                    response = "\n".join(lines)

            else:
                # Unknown follow-up → concise LLM summary within current context
                response = get_gemini_summary(prompt, {"customer": ctx["customer"], "orders": orders})

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
