import pytest

from eval.evaluate import mean, parse_score, render_markdown


@pytest.mark.parametrize(
    ("text", "score"),
    [
        ('{"score": 0.8, "reason": "ok"}', 0.8),
        ('Hasil:\n```json\n{"score": 1, "reason": "lengkap"}\n```', 1.0),
        ('{"score": 7}', 1.0),  # clamped
        ('{"score": -2}', 0.0),
    ],
)
def test_judge_scores_are_parsed_and_clamped(text, score):
    assert parse_score(text)[0] == score


@pytest.mark.parametrize("text", ["tidak ada json", '{"reason": "x"}', '{"score": "tinggi"}'])
def test_unparseable_judge_output_is_missing_not_zero(text):
    assert parse_score(text)[0] is None


def test_mean_ignores_missing_scores():
    assert mean([1.0, None, 0.5]) == 0.75
    assert mean([None, None]) is None


def test_markdown_report_lists_summary_and_cases():
    report = {
        "created_at": "20260101-000000",
        "config": {
            "llm_model": "m",
            "judge_model": "j",
            "embedding_model": "e",
            "reranker_model": "r",
            "top_k": 4,
        },
        "summary": {"cases": 1, "faithfulness": 1.0},
        "cases": [
            {
                "status": "answered",
                "faithfulness": 1.0,
                "context_relevance": 0.5,
                "correctness": None,
                "pages": [3],
                "question": "Apa sanksinya?",
            }
        ],
    }
    md = render_markdown(report)
    assert "| faithfulness | 1.0 |" in md
    assert "| 1 | answered | 1.0 | 0.5 | None | [3] | Apa sanksinya? |" in md
