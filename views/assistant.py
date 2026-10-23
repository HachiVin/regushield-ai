"""Assistant: ask questions about the indexed PDFs; answers stream in with cited sources."""

from __future__ import annotations

import json
import time

import streamlit as st

from regushield.engine import Answer
from views import common

CHATS = "chats_by_scope"
AVATARS = {"user": ":material/person:", "assistant": ":material/shield:"}


def render_sources(answer: Answer) -> None:
    if not answer.sources:
        return
    cited = sum(s.cited for s in answer.sources)
    label = f"Sumber ({len(answer.sources)} kutipan, {cited} dikutip di jawaban)"
    with st.expander(label, icon=":material/menu_book:"):
        for s in answer.sources:
            with st.container(border=True, key=f"source_{answer.id}_{s.number}"):
                tags = [f"**[{s.number}]** {s.source}", f"halaman {s.page}"]
                if s.kind == "table":
                    tags.append("tabel")
                if s.relevance is not None:
                    tags.append(f"relevansi {s.relevance:.2f}")
                if s.cited:
                    tags.append(":green-badge[dikutip]")
                st.markdown(" · ".join(tags))
                st.markdown(s.text)


def render_answer(answer: Answer) -> None:
    render_sources(answer)
    st.caption(common.answer_caption(answer))
    if answer.status == "answered" and not answer.cited:
        st.caption(
            ":orange[Jawaban ini tidak menyebut nomor kutipan; periksa sumber secara manual.]"
        )
    st.feedback("thumbs", key=f"feedback_{answer.id}", on_change=save_feedback, args=(answer.id,))


def save_feedback(answer_id: str) -> None:
    value = st.session_state.get(f"feedback_{answer_id}")
    if value is not None:
        common.get_engine().feedback(answer_id, positive=value == 1)


EXAMPLES = [
    "Apa saja jenis sanksi bagi pelanggaran kode etik?",
    "Bagaimana aturan penerimaan hadiah atau gratifikasi?",
    "Kepada siapa benturan kepentingan wajib dilaporkan?",
]
STEPS = [
    (":material/search:", "Cari", "BM25 + vector search menemukan pasal yang relevan."),
    (":material/sort:", "Urutkan", "Reranker menilai ulang tiap kutipan terhadap pertanyaan."),
    (":material/fact_check:", "Jawab", "LLM menjawab hanya dari kutipan, dengan nomor sumber."),
]


def onboarding() -> str | None:
    """How it works + example questions, shown while the conversation is empty."""
    columns = st.columns(3)
    for column, (icon, title, text) in zip(columns, STEPS, strict=True):
        with column.container(border=True):
            st.markdown(f"{icon} **{title}**")
            st.caption(text)
    st.caption("Contoh pertanyaan:")
    for i, example in enumerate(EXAMPLES):
        if st.button(example, key=f"example_{i}", icon=":material/arrow_outward:"):
            return example
    return None


def sidebar(engine, documents: list[str]) -> str | None:
    with st.sidebar:
        st.markdown("**Ruang lingkup**")
        scope = st.selectbox(
            "Ruang lingkup", [common.ALL_DOCUMENTS, *documents], label_visibility="collapsed"
        )
        st.caption(f"{len(documents)} dokumen · {engine.store.count()} potongan teks terindeks")
        history = st.session_state[CHATS].setdefault(scope, [])
        if history:
            log = {
                "scope": scope,
                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "messages": [
                    m["answer"].to_dict()
                    if "answer" in m
                    else {"role": "user", "content": m["content"]}
                    for m in history
                ],
            }
            st.download_button(
                "Unduh audit log (JSON)",
                json.dumps(log, indent=2, ensure_ascii=False),
                file_name=f"regushield_audit_{int(time.time())}.json",
                mime="application/json",
                icon=":material/download:",
                width="stretch",
            )
            if st.button("Hapus percakapan", icon=":material/delete_sweep:", width="stretch"):
                st.session_state[CHATS][scope] = []
                st.rerun()
        st.divider()
        settings = engine.settings
        model = engine.chat.model if engine.chat else "belum dikonfigurasi"
        reranker = settings.reranker_model if engine.retriever.reranker else "nonaktif"
        st.caption(
            f"LLM: `{model}`  \nEmbedding: `{settings.embedding_model.split('/')[-1]}`  \n"
            f"Reranker: `{reranker}`"
        )
    return None if scope == common.ALL_DOCUMENTS else scope


def main() -> None:
    st.session_state.setdefault(CHATS, {})
    engine = common.get_engine()
    documents = [d.source for d in engine.documents()]
    scope = sidebar(engine, documents)

    common.header(
        "ReguShield",
        "Tanya jawab dokumen regulasi, kepatuhan, dan SOP. Setiap jawaban menyebut pasal dan "
        'halaman sumbernya, dan sistem menjawab "tidak ditemukan" bila dokumen tidak memuatnya.',
    )
    if engine.chat is None:
        st.warning(
            "GROQ_API_KEY belum diatur: sistem tetap mencari kutipan, tetapi tidak "
            "menyusun jawaban.",
            icon=":material/key_off:",
        )
    if not documents:
        st.info(
            "Belum ada dokumen. Unggah PDF regulasi di halaman **Dokumen** terlebih dahulu.",
            icon=":material/upload_file:",
        )
        if st.button("Buka halaman Dokumen", icon=":material/arrow_forward:"):
            st.switch_page(common.PAGE_DOCUMENTS)
        return

    history = st.session_state[CHATS].setdefault(scope or common.ALL_DOCUMENTS, [])
    for message in history:
        with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
            if "answer" in message:
                st.markdown(message["answer"].text)
                render_answer(message["answer"])
            else:
                st.markdown(message["content"])

    placeholder = f"Tanya tentang {scope}..." if scope else "Tanya tentang semua dokumen..."
    question = st.chat_input(placeholder, max_chars=2000)
    if not history:
        slot = st.empty()
        with slot.container():
            picked = onboarding()
        question = question or picked
        if question:
            slot.empty()  # the conversation has started: hide the introduction
    if not question:
        return
    history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        stream = engine.ask(question, scope)
        with st.spinner("Mencari pasal yang relevan..."):
            st.write_stream(stream)
        render_answer(stream.result)
    history.append({"role": "assistant", "answer": stream.result})


main()