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
                    f"{report.tables} tabel, {report.ocr_images} gambar dibaca dengan OCR.",
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
            "Membaca dan mengindeks PDF. Gambar berisi teks dibaca dengan OCR, jadi PDF dengan "
            "banyak gambar bisa butuh beberapa menit..."
        ):
            ingest(engine, files)
    if engine.ocr is not None:
        ocr_note = (
            "OCR aktif: tabel atau teks berbentuk gambar dan PDF hasil scan ikut dibaca. Hasil "
            "OCR ditandai di sumber jawaban karena bisa salah baca."
        )
    elif engine.settings.ocr_enabled:
        ocr_note = (
            "OCR tidak tersedia (paket rapidocr-onnxruntime belum terpasang), jadi teks di dalam "
            "gambar tidak terbaca."
        )
    else:
        ocr_note = "OCR dimatikan (REGUSHIELD_OCR=false): teks di dalam gambar tidak terbaca."
    st.caption(f"Maksimal {engine.settings.max_file_mb} MB per file. {ocr_note}")

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
                    "Potongan OCR": d.ocr_chunks,
                    "Diindeks": d.ingested_at[:16].replace("T", " "),
                }
                for d in documents
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    legacy = [
        d.source for d in documents if not d.ordered or d.index_version != engine.index_version
    ]
    if legacy:
        st.info(
            f"{', '.join(legacy)} diindeks dengan pengaturan lama (ukuran potongan, OCR, atau "
            "urutan potongan). Unggah ulang file yang sama untuk mengindeks ulang.",
            icon=":material/update:",
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