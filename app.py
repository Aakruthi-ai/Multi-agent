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
import json

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
# CREDENTIAL VALIDATION (Fail-fast, clear error messages)
# =====================================================================
missing_secrets = []
if "GEMINI_API_KEY" not in st.secrets:
    missing_secrets.append("GEMINI_API_KEY")
if "gcp_service_account" not in st.secrets:
    missing_secrets.append("gcp_service_account (JSON key)")

if missing_secrets:
    st.error(f"🔒 Missing required secrets: {', '.join(missing_secrets)}. Please configure them in Streamlit Cloud → Settings → Secrets.")
    st.markdown("""
   
