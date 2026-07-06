
### Agent Architecture (7 Specialized Agents)

| Agent | Role | Technology |
|-------|------|------------|
| **Agent 1** | BigQuery Data Ingestion | `google-cloud-bigquery`, Service Account Auth |
| **Agent 2** | GPU-Accelerated Feature Engineering | NVIDIA RAPIDS `cudf.pandas`, Multi-variable composite risk scoring |
| **Agent 3** | Graph Fraud Ring Detection | `cuGraph` / `networkx` Connected Components |
| **Agent 4** | Logistics Disruption Scoring | Concurrent `ThreadPoolExecutor` |
| **Agent 5** | Gemini Decision Orchestration | `gemini-2.5-flash` with Function Calling |
| **Agent 6** | Business Action Execution | `freeze_payout`, `reroute_shipment`, `flag_for_manual_review` |
| **Agent 7** | Real-Time Risk Monitoring | Streamlit Dashboard + Time-Series Charts |

---

## ⚡ Performance Benchmarks

| Metric | Standard CPU (pandas) | NVIDIA RAPIDS GPU (cudf) | Improvement |
|--------|----------------------|--------------------------|-------------|
| Feature Engineering (500K rows) | 71.40 seconds | 1.55 seconds | **46.1x faster** |
| Latency Reduction | Baseline | -97.8% | Near real-time |
| Code Changes Required | — | **Zero** (cudf.pandas proxy) | Drop-in acceleration |

*Reference benchmarks from controlled NVIDIA RAPIDS test environment. The system uses cudf.pandas zero-code-change proxy — identical pandas API with GPU execution when available.*

---

## 🛠️ Technology Stack

### Google Cloud Platform
- **BigQuery** — Data warehouse for transaction and logistics tables, analytical SQL queries with device fingerprinting aggregation
- **Service Account Authentication** — Secure credential-based access via Streamlit Secrets
- **Gemini 2.5 Flash** — Autonomous decision orchestration with structured function calling

### NVIDIA Accelerated Computing
- **RAPIDS cudf.pandas** — Zero-code-change GPU acceleration proxy for pandas workloads
- **cuGraph** — GPU-accelerated graph analytics for connected-components fraud ring detection

### Additional Technologies
- **Streamlit** — Interactive operations dashboard
- **networkx** — CPU fallback for graph analysis
- **Python concurrent.futures** — Parallel agent execution

---

## 📊 Key Features

### 1. Fraud Ring Detection via Graph Topology
Looks beyond individual transactions to identify colluding networks. Connected-components analysis on user-device relationship graphs reveals organized fraud rings sharing device fingerprints — something simple rule-based systems miss entirely.

### 2. Multi-Variable Composite Risk Scoring
Combines fraud indicators, statistical deviations (z-score normalization), and logistics disruption metrics into a single mathematical risk score for holistic decision-making.

### 3. Autonomous Decision Execution
Gemini 2.5 Flash evaluates risk profiles and selects the correct business action from a structured tool matrix — freezing payouts for high-fraud rings, rerouting shipments for logistics exceptions, or queuing for manual review.

### 4. Graceful Degradation with Rule Engine Fallback
If Gemini API limits or network issues occur, a built-in deterministic rule engine ensures the system remains operational and protective.

### 5. Real-Time Operations Dashboard
Live metrics, fraud ring tables, decision audit logs, and time-series anomaly tracking provide complete operational visibility.

---

)

