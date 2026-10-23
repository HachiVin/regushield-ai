"""Streamlit entry point. Run with ``streamlit run app.py``."""

import streamlit as st

from views import common

st.set_page_config(
    page_title="ReguShield — Regulatory Q&A",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

page = st.navigation(
    [
        st.Page(common.PAGE_ASSISTANT, title="Assistant", icon=":material/forum:", default=True),
        st.Page(common.PAGE_DOCUMENTS, title="Dokumen", icon=":material/description:"),
        st.Page(common.PAGE_MONITORING, title="Monitoring", icon=":material/monitoring:"),
    ]
)
common.apply_theme()
page.run()
