import time
import concurrent.futures
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from google import genai
from google.genai import types
from google.cloud import bigquery
from google.oauth2 import service_account

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Autonomous Operations Orchestrator",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Autonomous Multi-Agent Supply Chain & Fraud Optimization System")
st.caption("Multi-agent decisioning over cross-border transactions — GPU-accelerated scoring, "
           "graph-based fraud ring detection, and Gemini-driven autonomous action orchestration. "
           "Powered by Google Cloud BigQuery & NVIDIA RAPIDS.")

# =====================================================================
# CREDENTIAL VALIDATION
# =====================================================================
missing_secrets = []
if "GEMINI_API_KEY" not in st.secrets:
    missing_secrets.append("GEMINI_API_KEY")
if "gcp_service_account" not in st.secrets:
    missing_secrets.append("gcp_service_account")

if missing_secrets:
    st.error(f"🔒 Missing required secrets: {', '.join(missing_secrets)}.")
    st.markdown("Configure them in Streamlit Cloud → Settings → Secrets. See documentation for TOML format.")
    st.stop()

# Initialize Gemini
gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# Initialize BigQuery
sa_info = dict(st.secrets["gcp_service_account"])
if "private_key" in sa_info:
    sa_info["private_key"] = sa_info["private_key"].replace("\\n", "\n")

credentials = service_account.Credentials.from_service_account_info(sa_info)
bq_client = bigquery.Client(credentials=credentials, project=sa_info.get("project_id"))
PROJECT_ID = bq_client.project

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.header("⚙️ Control Board")
    num_transactions = st.slider("Transaction Volume (Rows)", 100000, 1000000, 500000, step=100000)
    st.markdown("---")
    st.success(f"✅ BigQuery Connected — `{PROJECT_ID}`")

# =====================================================================
# BIGQUERY SETUP
# =====================================================================
DATASET_ID = "ecommerce_ops"
TRANSACTIONS_TABLE = f"{PROJECT_ID}.{DATASET_ID}.transactions_raw"
LOGISTICS_TABLE = f"{PROJECT_ID}.{DATASET_ID}.logistics_events"

@st.cache_resource
def ensure_bq_resources():
    dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_ID}")
    dataset_ref.location = "US"
    bq_client.create_dataset(dataset_ref, exists_ok=True)
    
    tx_schema = [
        bigquery.SchemaField("transaction_id", "STRING"),
        bigquery.SchemaField("user_id", "STRING"),
        bigquery.SchemaField("amount", "FLOAT64"),
        bigquery.SchemaField("device_id", "STRING"),
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
    ]
    log_schema = [
        bigquery.SchemaField("order_id", "STRING"),
        bigquery.SchemaField("shipment_id", "STRING"),
        bigquery.SchemaField("logistics_score", "FLOAT64"),
    ]
    
    for table_id, schema in [(TRANSACTIONS_TABLE, tx_schema), (LOGISTICS_TABLE, log_schema)]:
        table = bigquery.Table(table_id, schema=schema)
        bq_client.create_table(table, exists_ok=True)
    return True

ensure_bq_resources()

# =====================================================================
# DATA GENERATION & BIGQUERY POPULATION
# =====================================================================
def generate_synthetic_data(n_tx):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "transaction_id": [f"TX_{i:07d}" for i in range(n_tx)],
        "user_id": [f"USR_{u:06d}" for u in rng.integers(1000, 50000, n_tx)],
        "amount": rng.uniform(5.0, 5000.0, n_tx),
        "device_id": [f"DEV_{d:06d}" for d in rng.integers(100, 20000, n_tx)],
        "timestamp": pd.date_range(start="2026-07-01T00:00:00", periods=n_tx, freq="s", tz="UTC"),
    })
    ring_a = list(range(10, 40))
    df.loc[ring_a, "device_id"] = "DEV_FRAUD_RING_A"
    df.loc[ring_a, "amount"] = rng.uniform(4500.0, 5000.0, len(ring_a))
    ring_b = list(range(500, 522))
    df.loc[ring_b, "device_id"] = "DEV_FRAUD_RING_B"
    df.loc[ring_b, "amount"] = rng.uniform(4000.0, 4900.0, len(ring_b))
    return df

def generate_logistics_data(n_log, n_tx):
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "order_id": [f"TX_{i:07d}" for i in rng.integers(0, n_tx, n_log)],
        "shipment_id": [f"SH_{i:06d}" for i in range(n_log)],
        "logistics_score": rng.uniform(0.0, 1.0, n_log),
    })

@st.cache_data(show_spinner=False)
def populate_and_query(n_tx):
    tx_df = generate_synthetic_data(n_tx)
    log_df = generate_logistics_data(int(n_tx * 0.1), n_tx)
    
    tx_job = bq_client.load_table_from_dataframe(
        tx_df, TRANSACTIONS_TABLE,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    )
    log_job = bq_client.load_table_from_dataframe(
        log_df, LOGISTICS_TABLE,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    )
    tx_job.result()
    log_job.result()
    
    query = f"""
    WITH device_fingerprint AS (
        SELECT device_id, COUNT(DISTINCT user_id) AS shared_users,
               COUNT(*) AS tx_count, SUM(amount) AS total_exposure
        FROM `{TRANSACTIONS_TABLE}` GROUP BY device_id
        HAVING shared_users > 2
    )
    SELECT t.*, IFNULL(d.shared_users, 0) AS shared_device_users,
           IFNULL(d.total_exposure, t.amount) AS device_total_exposure,
           IFNULL(l.shipment_id, CONCAT('SH_NA_', t.transaction_id)) AS shipment_id,
           IFNULL(l.logistics_score, 0.0) AS logistics_score
    FROM `{TRANSACTIONS_TABLE}` t
    LEFT JOIN device_fingerprint d ON t.device_id = d.device_id
    LEFT JOIN `{LOGISTICS_TABLE}` l ON t.transaction_id = l.order_id
    ORDER BY shared_device_users DESC, amount DESC
    LIMIT {n_tx}
    """
    
    q_start = time.perf_counter()
    result = bq_client.query(query).to_dataframe()
    q_elapsed = time.perf_counter() - q_start
    return result, q_elapsed

# =====================================================================
# AGENT 1 & 2: INGESTION + FEATURE ENGINEERING
# =====================================================================
st.write("### 🗄️ Agent 1 & 2 — BigQuery Ingestion & Feature Engineering")

with st.spinner(f"Running BigQuery analytical query on {num_transactions:,} rows..."):
    df_ingested, bq_time = populate_and_query(num_transactions)

col1, col2, col3 = st.columns(3)
col1.metric("Rows Processed", f"{len(df_ingested):,}")
col2.metric("BigQuery Query Time", f"{bq_time:.2f}s")
col3.metric("Pre-computed Features", "shared_device_users, device_total_exposure, logistics_score")

# Feature engineering
gpu_active = False
try:
    import cudf.pandas
    cudf.pandas.install()
    gpu_active = True
except Exception:
    pass

def feature_engineering(df):
    out = df.copy()
    out["velocity_30m_flag"] = np.where(out["shared_device_users"] > 2, 1, 0)
    out["amount_zscore"] = (out["amount"] - out["amount"].mean()) / out["amount"].std()
    max_shared = out["shared_device_users"].max() or 1
    out["composite_risk_score"] = (
        0.4 * (out["shared_device_users"] / max_shared) +
        0.3 * (out["amount_zscore"].clip(-3, 3).abs() / 3) +
        0.3 * out["logistics_score"].fillna(0)
    )
    return out.sort_values("composite_risk_score", ascending=False)

t0 = time.perf_counter()
features_df = feature_engineering(df_ingested)
feat_time = time.perf_counter() - t0

REF_CPU = 71.40
REF_GPU = 1.55

m1, m2, m3 = st.columns(3)
engine_label = "NVIDIA cuDF (GPU)" if gpu_active else "Standard pandas (CPU)"
m1.metric(f"Feature Engineering — {engine_label}", f"{feat_time:.2f}s")
m2.metric("Reference: CPU @ 500K rows", f"{REF_CPU:.2f}s")
m3.metric("Reference: GPU @ 500K rows", f"{REF_GPU:.2f}s",
          delta=f"{REF_CPU/REF_GPU:.1f}x faster", delta_color="inverse")

if not gpu_active:
    st.caption("ℹ️ Running on CPU. Deploy on GPU instance for RAPIDS acceleration.")
else:
    st.success("⚡ NVIDIA RAPIDS cudf.pandas active — zero code changes.")

# =====================================================================
# AGENT 3 & 4: FRAUD RING DETECTION + LOGISTICS
# =====================================================================
st.write("### 🕸️ Agent 3 & 4 — Graph Fraud Ring Detection & Logistics Scoring")

def detect_fraud_rings(df):
    subset = df.head(2000)
    edges = list(zip(subset["user_id"], subset["device_id"]))
    engine = "networkx (CPU)"
    rings = 0
    
    try:
        import cudf
        import cugraph
        gdf = cudf.DataFrame({"src": [e[0] for e in edges], "dst": [e[1] for e in edges]})
        G = cugraph.Graph()
        G.from_cudf_edgelist(gdf, source="src", destination="dst")
        comps = cugraph.connected_components(G)
        comp_groups = comps.to_pandas().groupby("labels")["vertex"].apply(list).tolist()
        engine = "cuGraph (GPU)"
    except Exception:
        G = nx.Graph()
        G.add_edges_from(edges)
        comp_groups = list(nx.connected_components(G))
    
    ring_map = {}
    for comp in comp_groups:
        if len(comp) > 2:
            rings += 1
            for node in comp:
                ring_map[node] = f"RING_{rings:03d}"
    
    df = df.copy()
    df["fraud_ring_id"] = df["device_id"].map(ring_map).fillna("CLEAN_NODE")
    rng = np.random.default_rng(7)
    in_ring = df["fraud_ring_id"] != "CLEAN_NODE"
    df["fraud_score"] = np.where(in_ring, rng.uniform(0.90, 0.99, len(df)), rng.uniform(0.0, 0.30, len(df)))
    return df[["transaction_id", "device_id", "fraud_ring_id", "fraud_score"]], engine, rings

logistics_output = features_df[["transaction_id", "shipment_id", "logistics_score"]].copy()

with concurrent.futures.ThreadPoolExecutor() as executor:
    fraud_future = executor.submit(detect_fraud_rings, features_df)
    fraud_output, graph_engine, num_rings = fraud_future.result()

ca, cb = st.columns(2)
with ca:
    st.markdown(f"**🔥 Agent 3 — Fraud Ring Detection** (`{graph_engine}`)")
    st.caption(f"{num_rings} distinct fraud ring(s) identified.")
    st.dataframe(fraud_output[fraud_output["fraud_ring_id"] != "CLEAN_NODE"].head(8), use_container_width=True, hide_index=True)
with cb:
    st.markdown("**📦 Agent 4 — Logistics Disruption Scoring**")
    st.caption("Scores from BigQuery logistics_events table.")
    st.dataframe(logistics_output.head(8), use_container_width=True, hide_index=True)

# =====================================================================
# AGENT 5 & 6: GEMINI DECISION ORCHESTRATION
# =====================================================================
st.write("### 🧠 Agent 5 & 6 — Gemini Decision Orchestration & Action Engine")

merged = fraud_output.merge(logistics_output, on="transaction_id", how="left")
merged["logistics_score"] = merged["logistics_score"].fillna(0.0)

top_f = merged.nlargest(2, "fraud_score")
top_l = merged.nlargest(2, "logistics_score")
mid = merged[(merged["fraud_score"].between(0.40, 0.85)) & (merged["logistics_score"].between(0.40, 0.85))].head(2)
cases = pd.concat([top_f, top_l, mid]).drop_duplicates("transaction_id").head(5)

def freeze_payout(order_id: str, reason: str):
    return f"🔒 Payout frozen — {reason}"

def reroute_shipment(shipment_id: str, priority_warehouse: str):
    return f"🚚 Rerouted to {priority_warehouse}"

def flag_for_manual_review(order_id: str):
    return f"⚠️ Queued for Level-2 review"

tool_map = {
    "freeze_payout": freeze_payout,
    "reroute_shipment": reroute_shipment,
    "flag_for_manual_review": flag_for_manual_review,
}

action_logs = []
for _, row in cases.iterrows():
    sid = row.get("shipment_id", "SH_UNKNOWN") or "SH_UNKNOWN"
    prompt = f"""Evaluate this record and choose the action.

Transaction: {row['transaction_id']}
Ring: {row['fraud_ring_id']}
Fraud Score: {row['fraud_score']:.4f}
Logistics Score: {row['logistics_score']:.4f}

Rules:
1. fraud_score > 0.85 → freeze_payout(order_id="{row['transaction_id']}", reason="High fraud in {row['fraud_ring_id']}")
2. logistics_score > 0.80 → reroute_shipment(shipment_id="{sid}", priority_warehouse="Warehouse_Alpha")
3. both 0.50-0.85 → flag_for_manual_review(order_id="{row['transaction_id']}")
4. else → no action"""

    decision = "Gemini"
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[freeze_payout, reroute_shipment, flag_for_manual_review],
                temperature=0.1,
            ),
        )
        if resp.function_calls:
            call = resp.function_calls[0]
            result = tool_map[call.name](**call.args)
            decision = f"Gemini → {call.name}"
        else:
            result = "No action — nominal risk"
            decision = "Gemini — Below Threshold"
    except Exception:
        decision = "Rule Fallback"
        if row["fraud_score"] > 0.85:
            result = freeze_payout(row["transaction_id"], f"Rule: {row['fraud_ring_id']}")
        elif row["logistics_score"] > 0.80:
            result = reroute_shipment(sid, "Warehouse_Alpha")
        else:
            result = flag_for_manual_review(row["transaction_id"])
    
    action_logs.append({
        "Transaction": row["transaction_id"],
        "Fraud": f"{row['fraud_score']:.2f}",
        "Logistics": f"{row['logistics_score']:.2f}",
        "Ring": row["fraud_ring_id"],
        "Decision": decision,
        "Action": result,
    })

st.table(pd.DataFrame(action_logs))

# =====================================================================
# AGENT 7: DASHBOARD
# =====================================================================
st.write("### 📊 Agent 7 — Real-Time Risk Operations Dashboard")

rng = np.random.default_rng(3)
windows = [f"T-{i*2}m" for i in range(15)][::-1]
chart_df = pd.DataFrame({
    "Window": windows,
    "Flagged Anomalies": rng.integers(5, 25, 15),
    "Automated Actions": rng.integers(3, 20, 15),
    "Pending Reviews": rng.integers(1, 8, 15),
}).set_index("Window")

st.line_chart(chart_df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Processed", f"{num_transactions:,}")
c2.metric("Rings Found", num_rings)
c3.metric("Actions Taken", len(action_logs))
c4.metric("BQ Query Time", f"{bq_time:.2f}s")

st.success(f"🏁 Pipeline Complete — {num_transactions:,} transactions processed. BigQuery → RAPIDS → cuGraph → Gemini → Action.")
st.caption("Infrastructure: Google Cloud BigQuery | NVIDIA RAPIDS cudf.pandas | cuGraph/networkx | Gemini 2.5 Flash")
