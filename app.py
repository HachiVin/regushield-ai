import os
import time
import json
import streamlit as st
from src.rag_chain import FinancialRAGChain
from src.ingestion import ingest_single_file

# 1. Konfigurasi Halaman & Metadata Resmi
st.set_page_config(
    page_title="ReguShield — Regulatory & Compliance Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 95%;
    }

    /* Sidebar Clean Corporate */
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #21262d;
    }
    
    .sidebar-brand {
        padding: 6px 0 16px 0;
        margin-bottom: 12px;
        border-bottom: 1px solid #21262d;
    }
    
    .sidebar-brand-title {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #f0f6fc;
    }
    
    .sidebar-brand-desc {
        font-size: 0.76rem;
        color: #8b949e;
        margin-top: 2px;
    }

    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 8px;
    }

    /* Main Header */
    .main-header {
        margin-bottom: 24px;
        padding-top: 6px;
        padding-bottom: 16px;
        border-bottom: 1px solid #21262d;
    }
    
    .main-title {
        color: #ffffff;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        line-height: 1.4;
        margin: 0 0 8px 0;
        padding: 0;
        overflow: visible;
    }
    
    .main-subtitle {
        color: #8b949e;
        font-size: 0.92rem;
        line-height: 1.5;
        margin: 0;
    }

    /* Metrics Cards */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 18px;
    }
    .metric-label {
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #58a6ff;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #8b949e;
        margin-top: 4px;
    }

    /* Citation Block */
    .citation-block {
        background-color: #0d1117;
        border-left: 3px solid #1f6feb;
        border-top: 1px solid #30363d;
        border-right: 1px solid #30363d;
        border-bottom: 1px solid #30363d;
        border-radius: 0 6px 6px 0;
        padding: 12px 16px;
        margin-top: 8px;
        margin-bottom: 10px;
        font-size: 0.88rem;
        color: #c9d1d9;
    }
    
    .citation-meta-bar {
        font-size: 0.76rem;
        font-weight: 600;
        color: #8b949e;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #21262d;
        padding-bottom: 6px;
    }
    
    .audit-tag {
        font-size: 0.68rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .tag-high { background: rgba(46, 160, 67, 0.15); color: #3fb950; border: 1px solid rgba(46, 160, 67, 0.3); }
    .tag-mid  { background: rgba(210, 153, 34, 0.15); color: #d29922; border: 1px solid rgba(210, 153, 34, 0.3); }
    .tag-low  { background: rgba(248, 81, 73, 0.15); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.3); }

    .stChatMessage {
        border-radius: 8px;
        border: 1px solid #30363d;
        margin-bottom: 12px;
        background-color: #161b22;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model Engine Cache
@st.cache_resource(show_spinner="Menghubungkan ke Hybrid Vector Index & FlashRank...")
def load_rag_chain():
    return FinancialRAGChain(model_name="openai/gpt-oss-20b")

if "messages_by_doc" not in st.session_state:
    st.session_state.messages_by_doc = {}

if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {
        "latency": 0.0,
        "confidence": 0.0,
        "chunks_evaluated": 0
    }

# 4. Sidebar: Workspace Control
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">ReguShield Workspace</div>
        <div class="sidebar-brand-desc">Document Management & Compliance Audit</div>
    </div>
    """, unsafe_allow_html=True)

    # Ingestion Section
    st.markdown('<div class="section-label">Upload Dokumen</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Dokumen PDF", 
        type=["pdf"], 
        help="Unggah berkas SOP, POJK, atau regulasi internal dalam format PDF.",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        save_dir = os.path.join("data", "raw")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, uploaded_file.name)
        
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner(f"Mengindeks dokumen: {uploaded_file.name}..."):
                num_chunks = ingest_single_file(file_path)
                st.success(f"Dokumen berhasil diindeks ({num_chunks} segment teks).")
                st.cache_resource.clear()
                st.rerun()

    st.markdown("<div style='margin: 18px 0; border-bottom: 1px solid #21262d;'></div>", unsafe_allow_html=True)

    # Scope Selection
    st.markdown('<div class="section-label">Fokus Dokumen (Scope)</div>', unsafe_allow_html=True)
    raw_files = [f for f in os.listdir("data/raw") if f.endswith(".pdf")] if os.path.exists("data/raw") else []
    
    if raw_files:
        selected_doc = st.selectbox(
            "Pilih Dokumen:", 
            ["Semua Dokumen"] + raw_files,
            label_visibility="collapsed"
        )
        st.caption(f"{len(raw_files)} dokumen aktif terdaftar.")
    else:
        st.warning("Belum ada dokumen PDF di sistem.")
        selected_doc = "Semua Dokumen"

    if selected_doc not in st.session_state.messages_by_doc:
        st.session_state.messages_by_doc[selected_doc] = []

    st.markdown("<div style='margin: 18px 0; border-bottom: 1px solid #21262d;'></div>", unsafe_allow_html=True)

    # Audit & Log Export
    st.markdown('<div class="section-label">Aksi & Ekspor Sesi</div>', unsafe_allow_html=True)
    current_chat = st.session_state.messages_by_doc[selected_doc]
    
    if current_chat:
        audit_payload = {
            "document_scope": selected_doc,
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_queries": len([m for m in current_chat if m["role"] == "user"]),
            "chat_history": current_chat
        }
        json_string = json.dumps(audit_payload, indent=2, ensure_ascii=False, default=float)
        st.download_button(
            label="Download Audit Log (.json)",
            data=json_string,
            file_name=f"audit_{selected_doc.replace('.pdf', '')}_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True
        )

    short_name = (selected_doc[:16] + "..") if len(selected_doc) > 18 else selected_doc
    if st.button(f"Hapus Riwayat Chat ({short_name})", use_container_width=True):
        st.session_state.messages_by_doc[selected_doc] = []
        st.rerun()

# 5. Main Header
st.markdown("""
<div class="main-header">
    <div class="main-title">Regulatory Intelligence Platform</div>
    <div class="main-subtitle">
        Analisis regulasi dan verifikasi kepatuhan hukum berbasis pencarian hibrida presisi dengan audit trail otomatis.
    </div>
</div>
""", unsafe_allow_html=True)

# 6. Real-time Telemetry Metrics
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Target Dokumen</div>
        <div class="metric-val" style="font-size:0.92rem; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;" title="{selected_doc}">{selected_doc}</div>
        <div class="metric-sub">{len(raw_files)} dokumen terindeks</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi2:
    lat = st.session_state.last_metrics["latency"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Latensi Inferensi</div>
        <div class="metric-val">{lat:.2f}s</div>
        <div class="metric-sub">Hybrid + LLM Speed</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi3:
    conf = st.session_state.last_metrics["confidence"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rerank Confidence</div>
        <div class="metric-val">{conf:.1f}%</div>
        <div class="metric-sub">Skor kecocokan dokumen teratas</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Grounding Integrity</div>
        <div class="metric-val" style="color:#3fb950;">ACTIVE</div>
        <div class="metric-sub">Pencegahan halusinasi ketat</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# 7. Memuat Mesin RAG
try:
    rag_chain = load_rag_chain()
except Exception as e:
    st.error(f"Gagal menghubungkan ke mesin RAG: {e}")
    st.stop()

def render_audit_citations(sources_list):
    if not sources_list:
        return
    with st.expander("Bukti Sitasi & Kutipan Dokumen", expanded=False):
        for idx, src in enumerate(sources_list, 1):
            score = float(src.get("relevance_score", 0.0))
            if score >= 0.80:
                badge_class = "tag-high"
                badge_label = f"SANGAT TINGGI ({score:.4f})"
            elif score >= 0.45:
                badge_class = "tag-mid"
                badge_label = f"RELEVAN ({score:.4f})"
            else:
                badge_class = "tag-low"
                badge_label = f"RENDAH ({score:.4f})"

            st.markdown(f"""
            <div class="citation-block">
                <div class="citation-meta-bar">
                    <span>BUKTI #{idx} — DOKUMEN: <code>{src['source']}</code> | HALAMAN: <strong>{src['page']}</strong></span>
                    <span class="audit-tag {badge_class}">{badge_label}</span>
                </div>
                <div>"{src['snippet']}..."</div>
            </div>
            """, unsafe_allow_html=True)

# 8. Render Pesan Riwayat Dokumen
current_messages = st.session_state.messages_by_doc[selected_doc]
for msg in current_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if "sources" in msg and msg["sources"]:
            render_audit_citations(msg["sources"])

# 9. Input Chat Pertanyaan & Efek Streaming
if user_query := st.chat_input(f"Ajukan pertanyaan mengenai isi {selected_doc}..."):
    st.session_state.messages_by_doc[selected_doc].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner("Mencari pasal relevan & memverifikasi sitasi..."):
            start_time = time.time()
            response_data = rag_chain.answer_query(
                query=user_query, 
                filter_source=selected_doc
            )
            latency = time.time() - start_time
            
            answer = response_data["answer"]
            sources = response_data["sources"]
            
            top_score = float(sources[0].get("relevance_score", 0.0)) if sources else 0.0

            st.session_state.last_metrics = {
                "latency": latency,
                "confidence": top_score * 100,
                "chunks_evaluated": len(sources)
            }

        streamed_response = ""
        for chunk in answer.split(" "):
            streamed_response += chunk + " "
            message_placeholder.markdown(streamed_response + "▌", unsafe_allow_html=True)
            time.sleep(0.03) 
        message_placeholder.markdown(answer, unsafe_allow_html=True)
        
        render_audit_citations(sources)

    # Simpan ke riwayat memori percakapan
    st.session_state.messages_by_doc[selected_doc].append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    
    st.rerun()