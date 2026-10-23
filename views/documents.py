"""Documents: upload PDFs to index them, see what is indexed, remove documents."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from regushield.engine import IngestError
from views import common

UPLOADER_VERSION = "uploader_version"
REPORT = "ingest_report"

STATUS_TEXT = {
    "indexed": "diindeks",
    "replaced": "diperbarui (versi lama diganti)",
    "unchanged": "tidak berubah, dilewati",
}


def ingest(engine, files) -> None:
    messages = []
    progress = st.progress(0.0, text="Mengindeks dokumen...")
    for done, file in enumerate(files, start=1):
        try:
            report = engine.ingest(file.name, file.getvalue())
            messages.append(
                (
                    "success",
                    f"**{report.source}**: {STATUS_TEXT[report.status]} — "
                    f"{report.pages} halaman, {report.chunks} potongan, "
                    f"{report.tables} tabel.",
                )
            )
        except IngestError as exc:
            messages.append(("error", f"**{file.name}**: {exc}"))
        progress.progress(done / len(files), text=f"Mengindeks dokumen... ({done}/{len(files)})")
    st.session_state[REPORT] = messages
    st.session_state[UPLOADER_VERSION] = st.session_state.get(UPLOADER_VERSION, 0) + 1
    st.rerun()


def show_report() -> None:
    for kind, text in st.session_state.pop(REPORT, []):
        (st.success if kind == "success" else st.error)(text)


def main() -> None:
    engine = common.get_engine()
    common.header("Dokumen", "PDF yang diindeks menjadi sumber jawaban Assistant.")
    show_report()

    files = st.file_uploader(
        "Unggah PDF (SOP, POJK, kode etik, peraturan internal)",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.get(UPLOADER_VERSION, 0)}",
    )
    if st.button("Indeks dokumen", icon=":material/upload:", type="primary", disabled=not files):
        with st.spinner(
            "Membaca dan mengindeks PDF (unduhan model pertama kali bisa beberapa menit)..."
        ):
            ingest(engine, files)
    st.caption(
        f"Maksimal {engine.settings.max_file_mb} MB per file. PDF hasil scan (gambar) belum "
        "didukung karena tidak ada OCR."
    )

    st.divider()
    documents = engine.documents()
    st.subheader(f"Terindeks ({len(documents)})")
    if not documents:
        st.caption("Belum ada dokumen.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Dokumen": d.source,
                    "Halaman": d.pages,
                    "Potongan": d.chunks,
                    "Tabel": d.tables,
                    "Diindeks": d.ingested_at[:16].replace("T", " "),
                }
                for d in documents
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    with st.popover("Hapus dokumen", icon=":material/delete:"):
        target = st.selectbox("Dokumen", [d.source for d in documents], key="delete_target")
        confirm = st.checkbox("Ya, hapus dari indeks", key="delete_confirm")
        if st.button("Hapus", type="primary", disabled=not confirm, key="delete_button"):
            engine.delete(target)
            st.session_state[REPORT] = [("success", f"**{target}** dihapus dari indeks.")]
            st.rerun()
    st.caption(
        "Di Streamlit Community Cloud penyimpanan bersifat sementara: dokumen terhapus saat "
        "app di-restart. Untuk penyimpanan permanen jalankan dengan Docker dan volume."
    )


main()
