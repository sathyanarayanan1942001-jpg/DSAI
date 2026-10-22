"""
Employee Attrition Root-Cause Intelligence (EARCI)
Main Application & Control Portal
"""

import os
import sys
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="EARCI — Attrition Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.orchestrator import Orchestrator
from src.data_generator import save_sample_data
from src.database import get_agent_logs, get_audit_logs, clear_actions
from src.utils import format_currency_inr, DATA_DICTIONARY

# Ensure session state initialization
if "orchestrator" not in st.session_state:
    st.session_state["orchestrator"] = Orchestrator()
    
if "pipeline_state" not in st.session_state:
    st.session_state["pipeline_state"] = None
    
if "data_source" not in st.session_state:
    st.session_state["data_source"] = "Synthetic Cohort (1,500 employees)"

if "budget_limit" not in st.session_state:
    st.session_state["budget_limit"] = 500000.0

orch = st.session_state["orchestrator"]

def load_or_run_default():
    sample_path = os.path.join(current_dir, "data", "sample_employees.csv")
    if not os.path.exists(sample_path):
        save_sample_data(sample_path)
    df = pd.read_csv(sample_path)
    with st.spinner("🤖 Orchestrator Agent initializing multi-agent intelligence pipeline..."):
        state = orch.run_full_pipeline(
            df=df,
            budget=st.session_state["budget_limit"],
            source_label="Synthetic Sample Dataset"
        )
        st.session_state["pipeline_state"] = state
    return state

# Run initially if not already executed
if st.session_state["pipeline_state"] is None:
    load_or_run_default()

state = st.session_state["pipeline_state"]

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/businesswoman.png", width=64)
    st.title("EARCI Intelligence")
    st.caption("Employee Attrition Root-Cause Intelligence v2.4")
    
    st.markdown("---")
    st.subheader("Data Configuration")
    st.info(f"**Active Source:**\n{st.session_state['data_source']}")
    
    budget_input = st.number_input(
        "Intervention Budget Cap (INR)",
        min_value=50000.0,
        max_value=10000000.0,
        value=float(st.session_state["budget_limit"]),
        step=50000.0,
        format="%.0f"
    )
    if budget_input != st.session_state["budget_limit"]:
        st.session_state["budget_limit"] = budget_input
        st.rerun()

    if st.button("🔄 Re-run Agent Pipeline", use_container_width=True):
        if state and state.get("raw_df") is not None:
            with st.spinner("Re-evaluating pipeline across agents..."):
                st.session_state["pipeline_state"] = orch.run_full_pipeline(
                    state["raw_df"],
                    budget=st.session_state["budget_limit"],
                    source_label=st.session_state["data_source"]
                )
            st.success("Pipeline refreshed successfully!")
            st.rerun()

    st.markdown("---")
    st.caption("🛡️ **Governance Status**: Non-punitive policies active. Human signoff required for consequential decisions.")

# -------------------------------------------------------------
# MAIN CONTENT
# -------------------------------------------------------------
st.title("🏢 Employee Attrition Root-Cause Intelligence (EARCI)")
st.markdown(
    "An enterprise-grade, agentic AI platform that links predictive retention modeling, "
    "empirical root-cause discovery, algorithmic fairness audits, and human-supervised intervention tracking."
)

tab_overview, tab_data_mgmt, tab_agents, tab_data_dict = st.tabs([
    "🚀 Platform Overview",
    "📁 Dataset Ingestion & Upload",
    "🤖 Agent Activity & Audit Trail",
    "📖 Data Dictionary"
])

with tab_overview:
    # Top KPI Metrics Cards
    if state and state.get("status") == "COMPLETED":
        q_rep = state["quality_report"]
        mod_m = state["model_metrics"]
        plan = state["retention_plan"]
        
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Tracked Employees", f"{q_rep.get('unique_employees', 0):,}")
        with c2:
            st.metric("Observed Exits", f"{q_rep.get('total_exits', 0):,}", f"{q_rep.get('attrition_rate', 0):.1%} rate")
        with c3:
            st.metric("Champion ML Model", mod_m.get("best_model_name", "N/A"), f"ROC-AUC: {mod_m.get('test_roc_auc', 0):.2f}")
        with c4:
            st.metric("Validated Root Causes", len(state.get("approved_findings", [])))
        with c5:
            st.metric("Funded Interventions", len(plan.get("selected_interventions", [])), f"₹{plan.get('total_allocated', 0):,.0f}")

    st.markdown("---")
    st.subheader("System Architecture & Multipage Navigation")
    
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        st.markdown("""
        #### Available Intelligence Modules:
        1. **[1_Executive_Overview](pages/1_Executive_Overview.py)**
           - Strategic enterprise dashboard, turnover rate trends, departmental heatmaps, and replacement cost avoidance.
        2. **[2_Root_Cause_Explorer](pages/2_Root_Cause_Explorer.py)**
           - Rigorous hypothesis testing, effect sizes (Cohen's d, Cramer's V, Odds Ratios), confidence intervals, and confounding factor analysis.
        3. **[3_Attrition_Risk_Analytics](pages/3_Attrition_Risk_Analytics.py)**
           - Probabilistic 90-day risk scoring for active staff, SHAP global feature associations, calibration curves, and individual risk factor breakdowns.
        """)
    with col_nav2:
        st.markdown("""
        4. **[4_AI_Retention_Copilot](pages/4_AI_Retention_Copilot.py)**
           - Natural language inquiry interface powered by controlled analytical tools. Answers complex HR questions grounded strictly in validated metrics.
        5. **[5_Retention_Action_Tracker](pages/5_Retention_Action_Tracker.py)**
           - Human-in-the-loop intervention governance with SQLite persistence. Approve, reject, track implementation status, and monitor pre- vs post-intervention shifts.
        """)

    st.markdown("---")
    st.subheader("Seven-Stage Agentic AI Coordination")
    agents_summary = [
        {"Stage": "1. Ingestion", "Agent": "Orchestrator & Data Quality Agent", "Task": "Schema validation, deduplication, outlier bounding, leakage isolation."},
        {"Stage": "2. Statistics", "Agent": "Root-Cause Investigation Agent", "Task": "Two-sample Welch tests, Chi-squared, Odds Ratios, Benjamini-Hochberg FDR correction."},
        {"Stage": "3. Modeling", "Agent": "Attrition Analytics Agent", "Task": "Time-aware split, probability calibration, threshold optimization, SHAP/importance."},
        {"Stage": "4. Strategy", "Agent": "Retention Strategy Agent", "Task": "Multi-criteria ranking (Impact, Feasibility, Urgency, Cost) and budget optimization."},
        {"Stage": "5. Ethics", "Agent": "Validation & Governance Agent", "Task": "Disparate impact fairness auditing (80% rule), sample size safety checks."},
        {"Stage": "6. Workflow", "Agent": "Human-in-the-Loop HR Partner", "Task": "Supervisory approval, modification, or rejection before execution."},
        {"Stage": "7. Monitoring", "Agent": "Monitoring & Drift Agent", "Task": "Baseline vs follow-up evaluation and continuous Kolmogorov-Smirnov drift testing."}
    ]
    st.dataframe(pd.DataFrame(agents_summary), use_container_width=True, hide_index=True)

with tab_data_mgmt:
    st.subheader("Dataset Ingestion & Management")
    st.markdown("Switch between the default synthetic benchmark or upload an enterprise CSV dataset.")
    
    col_u1, col_u2 = st.columns([1.2, 1])
    with col_u1:
        uploaded_file = st.file_uploader("Upload Authorized HR CSV File", type=["csv"])
        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(user_df):,} rows from '{uploaded_file.name}'.")
                
                # Preview uploaded columns
                st.write("**Detected Columns:**", list(user_df.columns))
                
                # Column mapping helper
                st.markdown("##### Optional Column Mapping:")
                c_m1, c_m2 = st.columns(2)
                with c_m1:
                    map_emp = st.selectbox("Employee ID Column", ["(Auto-Detect)"] + list(user_df.columns))
                    map_dept = st.selectbox("Department Column", ["(Auto-Detect)"] + list(user_df.columns))
                with c_m2:
                    map_sal = st.selectbox("Salary Column", ["(Auto-Detect)"] + list(user_df.columns))
                    map_att = st.selectbox("Attrition Status Column", ["(Auto-Detect)"] + list(user_df.columns))
                    
                custom_map = {}
                if map_emp != "(Auto-Detect)": custom_map[map_emp] = "employee_id"
                if map_dept != "(Auto-Detect)": custom_map[map_dept] = "department"
                if map_sal != "(Auto-Detect)": custom_map[map_sal] = "monthly_salary"
                if map_att != "(Auto-Detect)": custom_map[map_att] = "attrition_status"
                
                if st.button("Apply Uploaded Dataset & Execute Pipeline", type="primary"):
                    orch.processor.column_mapping = custom_map
                    with st.spinner("Validating and executing pipeline on uploaded data..."):
                        new_state = orch.run_full_pipeline(
                            df=user_df,
                            budget=st.session_state["budget_limit"],
                            source_label=f"Uploaded: {uploaded_file.name}"
                        )
                        st.session_state["pipeline_state"] = new_state
                        st.session_state["data_source"] = f"Uploaded ({uploaded_file.name})"
                    st.success("Uploaded dataset processed successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error reading uploaded CSV: {e}")

    with col_u2:
        st.markdown("##### Reset to Default Synthetic Benchmark")
        st.write("Restore the standard 1,500-employee longitudinal benchmark dataset with 18 monthly observation cycles.")
        
        if st.button("⚠️ Reset to Clean Synthetic Benchmark", use_container_width=True):
            clear_actions()
            load_or_run_default()
            st.session_state["data_source"] = "Synthetic Cohort (1,500 employees)"
            st.success("Reset completed. Synthetic cohort loaded and evaluated.")
            st.rerun()

    # Data Quality Report Card
    if state and "quality_report" in state and state["quality_report"]:
        st.markdown("---")
        st.subheader("Data Quality & Integrity Report")
        qr = state["quality_report"]
        
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("Validation Status", "PASSED" if qr.get("is_valid") else "FAILED")
        qc2.metric("Clean Records", f"{qr.get('clean_record_count', 0):,}")
        qc3.metric("Deduplicated", f"{qr.get('duplicates_removed', 0):,}")
        qc4.metric("Date Span", f"{qr.get('date_range', ['N/A', 'N/A'])[0]} to {qr.get('date_range', ['N/A', 'N/A'])[1]}")
        
        with st.expander("🔍 View Detailed Preprocessing & Outlier Audit"):
            st.json(qr)

with tab_agents:
    st.subheader("Real-Time Agent Orchestration Activity Log")
    st.markdown("Transparent audit trail logging every agent's operational task, retrieved evidence, and output.")
    
    agent_logs = get_agent_logs(limit=30)
    if agent_logs:
        df_logs = pd.DataFrame(agent_logs)
        st.dataframe(
            df_logs[["timestamp", "agent_name", "task_name", "status", "evidence_summary", "result_summary"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No agent activity logs recorded yet.")

with tab_data_dict:
    st.subheader("EARCI Canonical Data Dictionary")
    st.markdown("Standardized definitions for organizational, operational, and predictive fields.")
    dict_df = pd.DataFrame([{"Field Name": k, "Description": v} for k, v in DATA_DICTIONARY.items()])
    st.dataframe(dict_df, use_container_width=True, hide_index=True)
    
    csv_bytes = dict_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Data Dictionary (CSV)",
        data=csv_bytes,
        file_name="earci_data_dictionary.csv",
        mime="text/csv"
    )
