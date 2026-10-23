"""Command line: ``python -m regushield {ingest,docs,ask,metrics}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from regushield.config import load_settings
from regushield.engine import IngestError, RAGEngine
from regushield.monitoring import summarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="regushield", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="index one or more PDF files")
    ingest.add_argument("files", nargs="+", type=Path)
    sub.add_parser("docs", help="list indexed documents")
    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--document", default=None, help="limit the search to one PDF")
    sub.add_parser("metrics", help="print the monitoring summary")
    args = parser.parse_args(argv)

    load_dotenv()
    engine = RAGEngine.from_settings(load_settings())

    if args.command == "ingest":
        failed = 0
        for path in args.files:
            try:
                report = engine.ingest(path.name, path.read_bytes())
                print(
                    f"{report.status:9s} {report.source}: {report.pages} pages, "
                    f"{report.chunks} chunks ({report.tables} tables)"
                )
            except (IngestError, OSError) as exc:
                failed += 1
                print(f"failed    {path.name}: {exc}", file=sys.stderr)
        return 1 if failed else 0
    if args.command == "docs":
        for doc in engine.documents():
            print(f"{doc.source}: {doc.pages} pages, {doc.chunks} chunks, {doc.tables} tables")
        return 0
    if args.command == "ask":
        stream = engine.ask(args.question, args.document)
        for delta in stream:
            print(delta, end="", flush=True)
        print()
        for s in stream.result.sources:
            mark = "*" if s.cited else " "
            score = f"{s.relevance:.3f}" if s.relevance is not None else "-"
            print(f"{mark}[{s.number}] {s.source} p.{s.page} (relevance {score})")
        return 0 if stream.result.status != "error" else 1
    print(json.dumps(summarize(engine.log.read()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
