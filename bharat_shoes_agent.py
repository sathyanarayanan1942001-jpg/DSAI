"""
BHARAT SHOES — Autonomous Customer Intelligence Agent
------------------------------------------------------
Run:
    pip install -r requirements.txt
    streamlit run bharat_shoes_agent.py

Optional LLM:
    set OPENAI_API_KEY=your_key
The application remains functional without an API key using deterministic
diagnostic logic; with an API key, the LLM interprets the evidence and
generates richer recommendations.
"""

import os, io, json, textwrap
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(
    page_title="Bharat Shoes | AI Customer Intelligence",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Visual system ----------
st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
.metric-card {
    padding: 16px 18px; border-radius: 14px; background: #f7f9fc;
    border: 1px solid #e6eaf0; min-height: 105px;
}
.ai-box {
    padding: 18px; border-radius: 14px; background: #eef7ff;
    border-left: 5px solid #1769aa; margin: 8px 0 18px 0;
}
.alert-box {
    padding: 18px; border-radius: 14px; background: #fff6e6;
    border-left: 5px solid #e28b00; margin: 8px 0 18px 0;
}
.small-note {color:#667085; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

# ---------- Data ----------
FEATURES = [
    "Purchases", "Total_Spend_INR", "Avg_Order_Value_INR",
    "Website_Visits", "Discount_Uses", "Cart_Additions",
    "Cart_Abandonments", "Returns", "Email_Open_Rate_Pct",
    "Social_Engagement", "Days_Since_Last_Purchase",
    "Campaign_Conversions", "Lifetime_Value_INR"
]

DEMO_FILE_XLSX = "bharat_shoes_100_customers.xlsx"
DEMO_FILE_CSV = "bharat_shoes_100_customers.csv"

def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)
    if os.path.exists(DEMO_FILE_XLSX):
        return pd.read_excel(DEMO_FILE_XLSX)
    if os.path.exists(DEMO_FILE_CSV):
        return pd.read_csv(DEMO_FILE_CSV)
    st.error("Upload a CSV or Excel file to begin.")
    st.stop()

def clean_data(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def make_clusters(df, k):
    X = df[FEATURES].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(Xs)
    out = df.copy()
    out["Cluster"] = labels
    score = silhouette_score(Xs, labels) if k > 1 else 0
    return out, scaler, model, score

def cluster_profiles(df):
    numeric = df.groupby("Cluster")[FEATURES].mean()
    overall = df[FEATURES].mean()
    profiles = {}
    for cluster, r in numeric.iterrows():
        signals = []
        if r["Total_Spend_INR"] > overall["Total_Spend_INR"] * 1.35:
            signals.append("high spending")
        if r["Purchases"] > overall["Purchases"] * 1.30:
            signals.append("high purchase frequency")
        if r["Discount_Uses"] > overall["Discount_Uses"] * 1.40:
            signals.append("high discount dependence")
        if r["Website_Visits"] > overall["Website_Visits"] * 1.35:
            signals.append("high digital activity")
        if r["Cart_Abandonments"] > overall["Cart_Abandonments"] * 1.40:
            signals.append("high cart abandonment")
        if r["Days_Since_Last_Purchase"] > overall["Days_Since_Last_Purchase"] * 1.45:
            signals.append("long purchase recency")
        if r["Email_Open_Rate_Pct"] > overall["Email_Open_Rate_Pct"] * 1.20:
            signals.append("high email engagement")
        if r["Returns"] > overall["Returns"] * 1.50 + 0.2:
            signals.append("high returns")

        s = " ".join(signals)
        if "high spending" in s and "high purchase frequency" in s:
            name = "High-Value Loyal"
        elif "long purchase recency" in s:
            name = "At-Risk"
        elif "high cart abandonment" in s and "high digital activity" in s:
            name = "High-Intent Non-Converted"
        elif "high discount dependence" in s:
            name = "Discount-Sensitive"
        else:
            name = "Emerging / Mixed Behaviour"
        profiles[int(cluster)] = {"name": name, "signals": signals}
    return profiles, numeric

def diagnose(df, profiles):
    issues = []
    total = len(df)

    at_risk = df["Days_Since_Last_Purchase"] >= 90
    high_abandon = (df["Cart_Abandonments"] >= df["Cart_Additions"].replace(0, 1) * 0.5) & (df["Website_Visits"] >= 20)
    discount_heavy = df["Discount_Uses"] > df["Purchases"] * 1.5
    low_engagement = (df["Email_Open_Rate_Pct"] < 25) & (df["Website_Visits"] < 10)

    if at_risk.mean() >= 0.15:
        issues.append({
            "severity": "HIGH",
            "issue": "Customer inactivity / retention crisis",
            "evidence": f"{at_risk.mean():.1%} of customers have not purchased for 90+ days.",
            "solution": "Launch a win-back journey prioritising high-LTV inactive customers."
        })
    if high_abandon.mean() >= 0.12:
        issues.append({
            "severity": "HIGH",
            "issue": "Conversion / cart-abandonment crisis",
            "evidence": f"{high_abandon.mean():.1%} show high-intent behaviour with significant abandonment.",
            "solution": "Use product-level retargeting, cart recovery and friction diagnostics."
        })
    if discount_heavy.mean() >= 0.15:
        issues.append({
            "severity": "MEDIUM",
            "issue": "Discount dependency",
            "evidence": f"{discount_heavy.mean():.1%} use discounts at a high rate relative to purchases.",
            "solution": "Replace blanket discounts with segment-specific bundles, loyalty rewards and controlled offers."
        })
    if low_engagement.mean() >= 0.15:
        issues.append({
            "severity": "MEDIUM",
            "issue": "Low campaign engagement",
            "evidence": f"{low_engagement.mean():.1%} have both low email engagement and low site activity.",
            "solution": "Test channel preference, message relevance and reactivation journeys."
        })

    if not issues:
        issues.append({
            "severity": "WATCH",
            "issue": "No single dominant crisis detected",
            "evidence": "The diagnostic thresholds did not identify a concentrated failure mode.",
            "solution": "Continue monitoring segment-level conversion, retention and campaign response."
        })
    return issues

def llm_analyse(context, question=None):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    prompt = f"""
You are the autonomous Customer Intelligence Agent for Bharat Shoes, an Indian
running-shoe brand. Use ONLY the supplied evidence. Do not invent facts.
Diagnose the marketing problem, explain likely behavioural causes, and propose
specific, measurable interventions. Distinguish observed evidence from
hypotheses. Avoid giving the same campaign to every segment.

DATA:
{context}

USER QUESTION:
{question or "Provide the overall diagnosis, crisis priorities, and segment-specific marketing actions."}

Return:
1. Diagnosis
2. Evidence
3. Root-cause hypotheses
4. Recommended actions
5. KPI to monitor
6. What should NOT be done
"""
    try:
        # Support both Responses API and standard Chat Completions
        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            response = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                input=prompt
            )
            return response.output_text
        else:
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
    except Exception as e:
        # Fallback to chat completions if responses API fails
        try:
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e2:
            return f"LLM unavailable; deterministic engine used. Technical note: {e2}"

# ---------- Sidebar ----------
st.sidebar.title("👟 Bharat Shoes")
st.sidebar.caption("Autonomous Customer Intelligence Agent")

uploaded = st.sidebar.file_uploader(
    "Drop your customer file here",
    type=["xlsx", "xls", "csv"],
    help="Required columns can be inferred; missing numeric columns default to 0."
)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Executive Overview", "Campaign Diagnosis", "Behavioural Segments",
     "Customer 360", "AI Marketing Strategist", "Campaign Control Room", "Data"]
)

k = st.sidebar.slider("Number of behavioural clusters", 3, 7, 5)

raw = load_data(uploaded)
df = clean_data(raw)
clustered, scaler, model, sil = make_clusters(df, k)
profiles, cluster_means = cluster_profiles(clustered)
issues = diagnose(clustered, profiles)

# ---------- Derived metrics ----------
at_risk_count = int((clustered["Days_Since_Last_Purchase"] >= 90).sum())
high_value_count = int((clustered["Lifetime_Value_INR"] >= clustered["Lifetime_Value_INR"].quantile(.75)).sum())
conversion_rate = clustered["Campaign_Conversions"].sum() / max(clustered["Purchases"].sum(), 1)
avg_ltv = clustered["Lifetime_Value_INR"].mean()

# ---------- PAGE 1 ----------
if page == "Executive Overview":
    st.title("Customer Intelligence Command Centre")
    st.caption("Bharat Shoes | Running footwear | Behaviour-first marketing")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Customers", f"{len(clustered):,}")
    c2.metric("At-Risk", f"{at_risk_count:,}")
    c3.metric("High-LTV", f"{high_value_count:,}")
    c4.metric("Avg. LTV", f"₹{avg_ltv:,.0f}")
    c5.metric("Cluster Quality", f"{sil:.2f}")

    st.markdown("### AI Diagnosis")
    top = issues[0]
    st.markdown(
        f'<div class="ai-box"><b>{top["severity"]}: {top["issue"]}</b><br>'
        f'{top["evidence"]}<br><br><b>Recommended direction:</b> {top["solution"]}</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        seg_counts = clustered["Cluster"].value_counts().sort_index()
        seg_names = [profiles[int(x)]["name"] for x in seg_counts.index]
        fig = px.bar(
            x=seg_names, y=seg_counts.values,
            labels={"x":"Behavioural Segment", "y":"Customers"},
            title="Customer Behavioural Segments"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.scatter(
            clustered, x="Website_Visits", y="Total_Spend_INR",
            size="Lifetime_Value_INR", color="Cluster",
            hover_data=["Customer_ID", "Purchases", "Days_Since_Last_Purchase"],
            title="Behaviour vs Value"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Crisis Watchlist")
    crisis_df = pd.DataFrame(issues)
    st.dataframe(crisis_df, use_container_width=True, hide_index=True)

# ---------- PAGE 2 ----------
elif page == "Campaign Diagnosis":
    st.title("🔎 Campaign Diagnosis")
    st.caption("Answers: Where is the campaign leaking value, and why?")

    # Funnel proxy from available customer interaction fields.
    total = len(clustered)
    visited = int((clustered["Website_Visits"] > 0).sum())
    carts = int((clustered["Cart_Additions"] > 0).sum())
    converted = int((clustered["Campaign_Conversions"] > 0).sum())

    funnel = pd.DataFrame({
        "Stage": ["Customers", "Website Engagers", "Cart Adders", "Campaign Converters"],
        "Customers": [total, visited, carts, converted]
    })
    fig = px.funnel(funnel, y="Stage", x="Customers", title="Customer Conversion Funnel")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected problems")
    for issue in issues:
        if issue["severity"] == "HIGH":
            st.markdown(
                f'<div class="alert-box"><b>🔴 {issue["issue"]}</b><br>'
                f'{issue["evidence"]}<br><br><b>Action:</b> {issue["solution"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="ai-box"><b>🟡 {issue["issue"]}</b><br>'
                f'{issue["evidence"]}<br><br><b>Action:</b> {issue["solution"]}</div>',
                unsafe_allow_html=True
            )

    st.subheader("Segment-level campaign risk")
    seg_table = clustered.groupby("Cluster").agg(
        Customers=("Customer_ID","count"),
        Avg_Spend=("Total_Spend_INR","mean"),
        Avg_Recency=("Days_Since_Last_Purchase","mean"),
        Avg_Abandon=("Cart_Abandonments","mean"),
        Avg_Discount=("Discount_Uses","mean"),
        Avg_Engagement=("Email_Open_Rate_Pct","mean")
    ).reset_index()
    seg_table["Segment"] = seg_table["Cluster"].map(lambda x: profiles[int(x)]["name"])
    st.dataframe(seg_table.drop(columns="Cluster").round(2), use_container_width=True, hide_index=True)

# ---------- PAGE 3 ----------
elif page == "Behavioural Segments":
    st.title("🧠 Behavioural Segmentation")
    st.caption("Machine learning discovers groups from behaviour; the LLM interprets them.")

    st.info(f"Current model: K-Means | K={k} | Silhouette score={sil:.2f}")

    seg_summary = clustered.groupby("Cluster").agg(
        Customers=("Customer_ID","count"),
        Spend=("Total_Spend_INR","mean"),
        Purchases=("Purchases","mean"),
        Visits=("Website_Visits","mean"),
        Discount_Uses=("Discount_Uses","mean"),
        Abandonments=("Cart_Abandonments","mean"),
        Recency=("Days_Since_Last_Purchase","mean"),
        LTV=("Lifetime_Value_INR","mean")
    ).reset_index()

    seg_summary["Segment"] = seg_summary["Cluster"].map(lambda x: profiles[int(x)]["name"])
    st.dataframe(seg_summary.drop(columns="Cluster").round(2), use_container_width=True, hide_index=True)

    selected = st.selectbox(
        "Inspect a segment",
        list(seg_summary["Segment"])
    )
    row = seg_summary[seg_summary["Segment"] == selected].iloc[0]
    cluster_id = int(row["Cluster"])
    p = profiles[cluster_id]

    a,b,c = st.columns(3)
    a.metric("Customers", int(row["Customers"]))
    b.metric("Avg Spend", f"₹{row['Spend']:,.0f}")
    c.metric("Avg LTV", f"₹{row['LTV']:,.0f}")

    st.markdown("### AI segment interpretation")
    signals = ", ".join(p["signals"]) if p["signals"] else "mixed behavioural pattern"
    st.markdown(
        f'<div class="ai-box"><b>{selected}</b><br>'
        f'The cluster is characterised by: {signals}.<br><br>'
        f'<b>Marketing implication:</b> Do not treat this group as interchangeable '
        f'with other customers merely because of age or location.</div>',
        unsafe_allow_html=True
    )

# ---------- PAGE 4 ----------
elif page == "Customer 360":
    st.title("👤 Customer 360")
    customer_id = st.selectbox("Select Customer ID", clustered["Customer_ID"].tolist())
    r = clustered[clustered["Customer_ID"] == customer_id].iloc[0]
    seg = profiles[int(r["Cluster"])]["name"]

    st.markdown(f"### {customer_id} — {seg}")
    a,b,c,d = st.columns(4)
    a.metric("Lifetime Value", f"₹{r['Lifetime_Value_INR']:,.0f}")
    b.metric("Total Spend", f"₹{r['Total_Spend_INR']:,.0f}")
    c.metric("Purchases", int(r["Purchases"]))
    d.metric("Days Since Purchase", int(r["Days_Since_Last_Purchase"]))

    left,right = st.columns(2)
    with left:
        customer_metrics = pd.DataFrame({
            "Metric": ["Website Visits","Cart Additions","Cart Abandonments",
                       "Discount Uses","Returns","Email Open Rate","Social Engagement"],
            "Value": [r["Website_Visits"],r["Cart_Additions"],r["Cart_Abandonments"],
                      r["Discount_Uses"],r["Returns"],r["Email_Open_Rate_Pct"],
                      r["Social_Engagement"]]
        })
        fig = px.bar(customer_metrics, x="Metric", y="Value", title="Behaviour Signals")
        fig.update_xaxes(tickangle=35)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Journey diagnosis")
        if r["Days_Since_Last_Purchase"] >= 90:
            journey = "Previously acquired → purchase history → declining activity → AT RISK"
        elif r["Cart_Abandonments"] >= 5 and r["Website_Visits"] >= 20:
            journey = "High browsing → product interest → cart activity → NON-CONVERTED"
        elif r["Discount_Uses"] > r["Purchases"] * 1.5:
            journey = "Browsing → promotion exposure → discount-led purchase behaviour"
        elif r["Purchases"] >= 8 and r["Total_Spend_INR"] >= clustered["Total_Spend_INR"].median():
            journey = "Repeat purchase → high value → LOYAL"
        else:
            journey = "Acquisition → developing relationship"
        st.markdown(f'<div class="ai-box">{journey}</div>', unsafe_allow_html=True)

# ---------- PAGE 5 ----------
elif page == "AI Marketing Strategist":
    st.title("🤖 AI Marketing Strategist")
    st.caption("Ask the agent to diagnose, explain and recommend actions from the uploaded data.")

    context = {
        "customer_count": len(clustered),
        "issues": issues,
        "segments": [
            {"name": profiles[int(c)]["name"], "cluster": int(c)}
            for c in sorted(profiles)
        ],
        "metrics": {
            "at_risk": at_risk_count,
            "high_value": high_value_count,
            "avg_ltv": round(float(avg_ltv),2),
            "silhouette": round(float(sil),3)
        }
    }

    question = st.text_area(
        "Ask the Customer AI",
        placeholder="Example: Why are our campaigns inefficient and what should we do about the at-risk customers?"
    )
    if st.button("Run Autonomous Analysis", type="primary"):
        llm_answer = llm_analyse(json.dumps(context, indent=2), question)
        if llm_answer:
            st.markdown(llm_answer)
        else:
            st.markdown(
                f'<div class="ai-box"><b>Deterministic AI-agent diagnosis</b><br>'
                f'The current evidence identifies {at_risk_count} at-risk customers and '
                f'{high_value_count} high-LTV customers. The main action is to separate '
                f'retention, conversion, discount and loyalty interventions instead of '
                f'using one campaign for all customers.</div>',
                unsafe_allow_html=True
            )

# ---------- PAGE 6 ----------
elif page == "Campaign Control Room":
    st.title("🎯 Campaign Control Room")
    st.caption("Translate diagnosis into differentiated actions.")

    # Distinct segment options across discovered profiles
    unique_segments = sorted(list({p["name"] for p in profiles.values()}))
    segment = st.selectbox("Target segment", unique_segments)

    objectives = ["Retention", "Conversion", "Loyalty", "Reactivation", "New-customer nurturing"]
    objective = st.selectbox("Campaign objective", objectives)

    channels = st.multiselect(
        "Preferred channels",
        ["Email", "WhatsApp", "SMS", "Instagram", "Website / App"],
        default=["Email", "WhatsApp"]
    )

    if st.button("Generate Campaign Plan", type="primary"):
        strategy = {
            "High-Value Loyal": (
                "Loyalty and premium-access journey; avoid unnecessary blanket discounts.",
                "Repeat purchase, LTV, loyalty engagement"
            ),
            "Discount-Sensitive": (
                "Targeted bundles, controlled promotions and loyalty rewards rather than blanket discounts.",
                "Conversion, margin, repeat purchase"
            ),
            "High-Intent Non-Converted": (
                "Product retargeting, cart recovery, reviews/social proof and checkout-friction testing.",
                "Cart-to-purchase conversion"
            ),
            "At-Risk": (
                "Win-back journey prioritising high-LTV inactive customers; test personalized incentives.",
                "Reactivation and repeat purchase"
            ),
            "Emerging / Mixed Behaviour": (
                "Nurture with product education, recommendations and behavioural observation.",
                "Engagement and second purchase"
            )
        }
        action, kpi = strategy.get(segment, strategy["Emerging / Mixed Behaviour"])
        st.markdown(
            f'<div class="ai-box"><h4>Recommended intervention</h4>{action}<br><br>'
            f'<b>Selected objective:</b> {objective}<br>'
            f'<b>Channels:</b> {", ".join(channels) if channels else "Select at least one"}<br>'
            f'<b>Primary KPI:</b> {kpi}</div>',
            unsafe_allow_html=True
        )

        st.subheader("Autonomous campaign checklist")
        for item in [
            "Check eligibility using current behavioural signals",
            "Suppress customers already converted by the campaign",
            "Personalize creative/product recommendations by segment",
            "Apply frequency caps to avoid message fatigue",
            "Track delivery → engagement → conversion",
            "Feed campaign response back into the next segmentation cycle"
        ]:
            st.checkbox(item, value=True, disabled=True)

# ---------- PAGE 7 ----------
elif page == "Data":
    st.title("📊 Data & Upload Centre")
    st.caption("Upload a customer Excel file or use the fabricated Bharat Shoes dataset.")

    st.markdown(
        '<div class="ai-box"><b>Excel drop zone</b><br>'
        'Upload .xlsx/.xls customer transaction and interaction data from the sidebar. '
        'The agent cleans the data, derives behavioural features, clusters customers, '
        'diagnoses campaign problems and produces recommendations.</div>',
        unsafe_allow_html=True
    )

    st.write(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")
    st.dataframe(df.head(25), use_container_width=True)

    csv = clustered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download analysed customer data (CSV)",
        csv,
        "bharat_shoes_analysed_customers.csv",
        "text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.caption("Demo dataset: 100 fabricated Bharat Shoes customers.")
st.sidebar.caption("Use real customer data only with appropriate privacy, consent and access controls.")
