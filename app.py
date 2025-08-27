import streamlit as st
import pandas as pd
import json
import re
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

# --- Data (always local sample CSV) ---
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

# --- State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_context" not in st.session_state:
    st.session_state.current_context = None  # {"customer": {...}, "orders": [...], "df": filtered_df}

# --- Helpers (no LLM extraction) ---
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)

def parse_identifier_from_text(msg: str) -> tuple[str|None, str|None, str|None]:
    """
    From a free-form chat message, try to extract:
    - email (wins if present), or
    - full name → first/last (best-effort)
    Heuristics:
      1) email via regex
      2) if message contains quoted name "Jane Doe" use that
      3) look for name after 'for ' / 'for customer ' / 'for client ' (two words)
      4) fallback: two capitalized words anywhere
    Returns (email, first_name, last_name)
    """
    if not msg:
        return None, None, None

    # 1) Email wins
    m = EMAIL_RE.search(msg)
    if m:
        return m.group(0).lower(), None, None

    # normalize spaces
    s = " ".join(msg.strip().split())

    # 2) Quoted full name
    mq = re.search(r'"([^"]+)"', s)
    if mq:
        name = mq.group(1).strip()
        parts = [p for p in name.split() if p]
        if len(parts) >= 2:
            return None, parts[0], parts[-1]
        elif len(parts) == 1:
            return None, parts[0], None

    # 3) After keywords like "for", "for customer", "for client"
    mk = re.search(r"\bfor(?:\s+(?:customer|client))?\s+([A-Za-z][A-Za-z'\-]+)\s+([A-Za-z][A-Za-z'\-]+)\b", s, re.I)
    if mk:
        return None, mk.group(1), mk.group(2)

    # 4) Any two capitalized words next to each other
    mc = re.search(r"\b([A-Z][a-zA-Z'\-]+)\s+([A-Z][a-zA-Z'\-]+)\b", s)
    if mc:
        return None, mc.group(1), mc.group(2)

    # 5) Single token (treat as first name partial)
    ms = re.search(r"\b([A-Za-z][A-Za-z'\-]+)\b", s)
    if ms:
        return None, ms.group(1), None

    return None, None, None

def filter_customer(df: pd.DataFrame, email: str | None, first_name: str | None, last_name: str | None) -> pd.DataFrame:
    out = df.copy()
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
- Mention product names, quantities, key dates.
- Include item link URLs only if present. Do not search the web.
- If multiple orders, summarize the most recent first (by Date Ordered).
- End with: "If you want more details (like prices, shipping, or tracking), please let me know."
- Use Markdown bullets when helpful.
"""
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
        st.error(f"Error generating AI summary: {e}")
        return "Sorry, I couldn't generate a summary right now."

def detect_followup_intent(msg: str) -> dict:
    m = re.search(r"\border\s*#?\s*([A-Za-z0-9\-]+)\b", msg, re.I)
    if m: return {"type": "details", "order_id": m.group(1)}
    if re.search(r"\b(track|tracking|status)\b", msg, re.I): return {"type": "show", "what": "tracking"}
    if re.search(r"\b(price|total|subtotal|tax|discount)\b", msg, re.I): return {"type": "show", "what": "prices"}
    if re.search(r"\b(items?|line items?|products?)\b", msg, re.I): return {"type": "show", "what": "items"}
    if re.search(r"\b(ship|address|delivery)\b", msg, re.I): return {"type": "show", "what": "shipping"}
    if re.search(r"\bre-?order|order again|same as last time\b", msg, re.I): return {"type": "reorder"}
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
st.caption("Type an email or full name in your message (e.g., “Show past orders for john@acme.com” or “Show past orders for Jane Doe”). No web search; links shown only if present in your data.")

# Show tiny dataset info (optional)
with st.expander("Dataset info"):
    st.write(f"Rows: {len(df):,}  •  Columns: {len(df.columns)}")
    st.write("Key columns used:", [
        "Order ID","Order Status","Date Ordered","Date Completed","Customer Email",
        "Customer First Name","Customer Last Name","Product Name","Quantity","links"
    ])

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("e.g., “Show past orders for john@acme.com” or “Show tracking for Jane Doe’s last order”")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        # Try to extract a customer identifier (email/full name) from THIS message
        # Try to extract a customer identifier (email/full name) from THIS message
        email, first_name, last_name = parse_identifier_from_text(prompt)
        
        ctx = st.session_state.current_context
        response = None
        
        if any([email, first_name, last_name]):
            # User gave a new identifier → rebuild context
            filtered = filter_customer(df, email, first_name, last_name)
            if filtered.empty:
                response = "I couldn’t find a matching customer. Try an email (e.g., john@acme.com) or full name (e.g., John Doe)."
            else:
                compact = compact_orders_for_llm(filtered)
                st.session_state.current_context = {
                    "customer": compact["customer"],
                    "orders": compact["orders"],
                    "df": filtered.copy()
                }
                # Summarize using LLM
                response = get_gemini_summary(prompt, {"customer": compact["customer"], "orders": compact["orders"]})
        
                # Show current customer pill
                cust = compact["customer"] or {}
                display_name = " ".join([cust.get("first_name",""), cust.get("last_name","")]).strip() or cust.get("email","(no name)")
                response = f"**Current Customer:** <span class='pill'>{display_name}</span>\n\n" + response
        
        else:
            # No identifier → follow-up using existing context
            if not ctx:
                response = ("Please include an email or full name in your first message. "
                            "For example: `Show past orders for john@acme.com` or `Show past orders for Jane Doe`.")
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
                    response = get_gemini_summary(prompt, {"customer": ctx["customer"], "orders": orders})



    

    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": response})
