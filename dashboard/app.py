"""Streamlit dashboard for the watch listing anomaly detector.

Two scoring modes:
- If API_URL is set, the dashboard calls the FastAPI /score endpoint (local dev,
  demonstrates the API).
- Otherwise it imports the scoring module and scores in-process. This is what runs
  on Streamlit Community Cloud, which serves only the Streamlit app (no separate
  API server).
"""
import os
import sys
import streamlit as st

# make the api package importable when running from the repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_URL = os.environ.get("API_URL")  # set this only for local API mode

if API_URL:
    import requests

    def get_score(payload):
        resp = requests.post(f"{API_URL}/score", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
else:
    from api import scoring

    def get_score(payload):
        return scoring.score_listing(payload)

BRANDS = ["Rolex", "Omega", "Seiko", "Breitling", "Cartier", "Longines",
          "Audemars Piguet", "TAG Heuer", "Hublot", "Patek Philippe",
          "IWC", "Tudor", "Panerai", "Other"]
MOVEMENTS = ["", "Automatic", "Quartz", "Manual winding"]
CONDITIONS = ["", "Unworn", "New", "Very good", "Good", "Fair", "Poor", "Incomplete"]

st.set_page_config(page_title="Watch Listing Anomaly Detector", page_icon=None, layout="centered")
st.title("Watch listing anomaly detector")
st.caption("Scores how unusual a luxury watch listing is across price, spec "
           "consistency, and completeness. The score is a signal to investigate, "
           "not a fraud verdict.")

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", BRANDS)
    model = st.text_input("Model", placeholder="e.g. Submariner Date")
    price = st.number_input("Price (USD)", min_value=1.0, value=8500.0, step=100.0)
    ref = st.text_input("Reference", placeholder="e.g. 126610LN")
with col2:
    mvmt = st.selectbox("Movement", MOVEMENTS)
    condition = st.selectbox("Condition", CONDITIONS)
    yop = st.text_input("Year of production", placeholder="e.g. 2021")
    size = st.text_input("Case size", placeholder="e.g. 41 mm")

with st.expander("Optional spec fields"):
    casem = st.text_input("Case material", placeholder="e.g. Steel")
    bracem = st.text_input("Bracelet material", placeholder="e.g. Steel")

if st.button("Score listing", type="primary"):
    payload = {
        "brand": brand, "price": float(price),
        "model": model or None, "ref": ref or None,
        "mvmt": mvmt or None, "condition": condition or None,
        "yop": yop or None, "size": size or None,
        "casem": casem or None, "bracem": bracem or None,
    }
    try:
        result = get_score(payload)
    except Exception as e:
        st.error(f"Scoring failed: {e}")
        st.stop()

    risk = result["risk_score"]
    band = result["risk_band"]
    color = {"low": "green", "medium": "orange", "high": "red"}[band]

    st.markdown(f"### Risk score: :{color}[{risk} / 100]  ({band})")
    st.progress(min(int(risk), 100))

    st.markdown("**Why:**")
    for r in result["reasons"]:
        st.markdown(f"- {r}")

    st.markdown("**Score breakdown**")
    b = result["breakdown"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Underpriced", f"{b['underpriced_score']:.0f}")
    c2.metric("Spec anomaly", f"{b['spec_anomaly_score']:.0f}")
    c3.metric("Completeness", f"{b['completeness_score']:.0f}")
    st.json(b, expanded=False)
