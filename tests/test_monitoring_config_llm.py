from types import SimpleNamespace

import pytest

from regushield.config import Settings, load_settings
from regushield.llm import Completion, Usage, _usage_from
from regushield.monitoring import QueryLog, estimate_cost, percentile, summarize


# --- monitoring -----------------------------------------------------------------------------
def record(status="answered", total=1000.0, cited=True, cost=0.001, tokens=(100, 20), **extra):
    return {
        "type": "query",
        "status": status,
        "latency_ms": {"total": total},
        "ttft_ms": total / 4 if status == "answered" else None,
        "cited": cited,
        "cost_usd": cost,
        "prompt_tokens": tokens[0],
        "completion_tokens": tokens[1],
        **extra,
    }


def test_percentile_nearest_rank():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == 50 and percentile(values, 95) == 95
    assert percentile([], 50) is None
    assert percentile([7.0], 95) == 7.0


def test_summary_rates_latency_and_cost():
    records = [
        record(total=1000),
        record(total=3000, cited=False),
        record(status="not_found", total=200, cost=None, tokens=(0, 0)),
        record(status="error", total=50, cost=None, tokens=(0, 0)),
    ]
    s = summarize(records)
    assert s["queries"] == 4 and s["answered"] == 2
    assert s["not_found_rate"] == 0.25 and s["error_rate"] == 0.25
    assert s["uncited_rate"] == 0.5
    assert s["latency_p50_ms"] == 1000 and s["latency_p95_ms"] == 3000  # errors excluded
    assert s["total_cost_usd"] == pytest.approx(0.002)
    assert s["avg_tokens"] == 120


def test_empty_summary_has_no_numbers():
    s = summarize([])
    assert s["queries"] == 0 and s["latency_p50_ms"] is None and s["not_found_rate"] is None


def test_log_round_trip_skips_bad_lines_and_attaches_feedback(tmp_path):
    log = QueryLog(tmp_path / "logs" / "q.jsonl")
    log.log_query(record(id="a"))
    log.log_query(record(id="b"))
    log.path.open("a").write("{bukan json\n")
    log.log_feedback("a", positive=True)
    log.log_feedback("zzz", positive=False)  # unknown id is ignored
    rows = {r["id"]: r for r in log.read()}
    assert set(rows) == {"a", "b"}
    assert rows["a"]["feedback"] is True and "feedback" not in rows["b"]
    assert summarize(list(rows.values()))["positive_feedback_rate"] == 1.0


def test_missing_log_reads_empty(tmp_path):
    assert QueryLog(tmp_path / "none.jsonl").read() == []


def test_cost_estimate():
    assert estimate_cost(1_000_000, 1_000_000, (0.075, 0.30)) == pytest.approx(0.375)
    assert estimate_cost(10, 10, None) is None


# --- config ---------------------------------------------------------------------------------
def test_defaults_without_environment():
    s = load_settings({})
    assert s.groq_api_key is None and s.rerank_enabled
    assert s.llm_model == "openai/gpt-oss-20b"
    assert s.prices_for("openai/gpt-oss-20b") == (0.075, 0.30)
    assert s.prices_for("unknown/model") is None


def test_environment_overrides(tmp_path):
    s = load_settings(
        {
            "GROQ_API_KEY": " gsk_x ",
            "REGUSHIELD_DATA_DIR": str(tmp_path),
            "REGUSHIELD_TOP_K": "6",
            "REGUSHIELD_MIN_RELEVANCE": "0.3",
            "REGUSHIELD_RERANK": "false",
            "REGUSHIELD_PRICE_INPUT_PER_M": "1",
            "REGUSHIELD_PRICE_OUTPUT_PER_M": "2",
            "REGUSHIELD_LOG_QUESTIONS": "no",
        }
    )
    assert s.groq_api_key == "gsk_x" and s.data_dir == tmp_path
    assert s.top_k == 6 and s.min_relevance == 0.3 and not s.rerank_enabled
    assert s.prices_for("anything") == (1.0, 2.0) and not s.log_questions
    assert s.index_dir == tmp_path / "index" and s.log_path.name == "queries.jsonl"


def test_collection_name_depends_on_the_embedding_model():
    a = Settings(embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    b = Settings(embedding_model="intfloat/multilingual-e5-large")
    assert a.collection_name != b.collection_name
    assert (
        all(ch.isalnum() or ch == "-" for ch in a.collection_name) and len(a.collection_name) <= 60
    )


# --- llm streaming --------------------------------------------------------------------------
def test_completion_tracks_text_ttft_and_usage():
    completion = Completion(iter([("Halo ", None), ("dunia", None), ("", Usage(10, 2))]), 40)
    assert "".join(completion) == "Halo dunia"
    assert completion.text == "Halo dunia" and completion.ttft_ms is not None
    assert completion.usage == Usage(10, 2)


def test_completion_estimates_usage_when_the_provider_sends_none():
    completion = Completion(iter([("x" * 40, None)]), prompt_chars=400)
    list(completion)
    assert completion.usage == Usage(100, 10, estimated=True)


@pytest.mark.parametrize(
    "chunk",
    [
        SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3), x_groq=None),
        SimpleNamespace(
            usage=None,
            x_groq=SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3)),
        ),
    ],
)
def test_groq_usage_is_read_from_either_field(chunk):
    assert _usage_from(chunk) == Usage(7, 3)


def test_chunks_without_usage_give_none():
    assert _usage_from(SimpleNamespace(usage=None, x_groq=SimpleNamespace(usage=None))) is None


# --- Groq client (network replaced by a fake SDK client) ------------------------------------
def groq_chat(create):
    from regushield.llm import GroqChat

    chat = GroqChat("gsk_test", "openai/gpt-oss-20b")
    chat._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return chat


def chunk(text, usage=None):
    delta = SimpleNamespace(content=text)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta)], usage=None, x_groq=SimpleNamespace(usage=usage)
    )


def test_groq_stream_yields_text_and_usage_and_sends_low_reasoning():
    calls = {}

    def create(**kwargs):
        calls.update(kwargs)
        return iter(
            [
                chunk("Halo "),
                chunk("[1]."),
                chunk("", SimpleNamespace(prompt_tokens=50, completion_tokens=4)),
            ]
        )

    completion = groq_chat(create).stream([{"role": "user", "content": "x"}])
    assert "".join(completion) == "Halo [1]."
    assert completion.usage == Usage(50, 4)
    assert calls["stream"] is True and calls["temperature"] == 0.0
    assert calls["reasoning_effort"] == "low"


def test_reasoning_effort_is_not_sent_to_other_models():
    from regushield.llm import GroqChat

    assert GroqChat("gsk_test", "llama-3.3-70b-versatile")._extra == {}


@pytest.mark.parametrize(
    ("status", "message"),
    [(401, "tidak valid"), (429, "Batas request"), (404, "tidak tersedia"), (503, "HTTP 503")],
)
def test_groq_errors_become_readable_messages(status, message):
    import groq
    import httpx

    from regushield.llm import LLMError

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request)
    error_cls = {
        401: groq.AuthenticationError,
        429: groq.RateLimitError,
        404: groq.NotFoundError,
    }.get(status, groq.APIStatusError)

    def create(**kwargs):
        raise error_cls("boom", response=response, body=None)

    with pytest.raises(LLMError, match=message):
        list(groq_chat(create).stream([{"role": "user", "content": "x"}]))


def test_groq_connection_problems_become_readable_messages():
    import groq
    import httpx

    from regushield.llm import LLMError

    request = httpx.Request("POST", "https://api.groq.com")
    for error, message in (
        (groq.APITimeoutError(request=request), "timeout"),
        (groq.APIConnectionError(request=request), "terhubung"),
    ):

        def create(error=error, **kwargs):
            raise error

        with pytest.raises(LLMError, match=message):
            list(groq_chat(create).stream([{"role": "user", "content": "x"}]))
