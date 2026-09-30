import sys
import os
import argparse
import json
from uncertain.parser import parse, ParseError
from uncertain.lexer import LexerError
from uncertain.ast_nodes import Span
from uncertain.typechecker import TypeContext, check_program
from uncertain.diagnostics import Diagnostic, KINDS, format_diagnostic, diagnostic_message
from uncertain.output import format_value, to_json

ERROR_CATALOG = {kind: info.explanation for kind, info in KINDS.items()}

def get_version() -> str:
    try:
        from importlib.metadata import version
        return version("uncertain-lang")
    except Exception:
        return "unknown"

def _diag_json(diag: Diagnostic, source_lines: list[str]) -> dict:
    return {
        "kind": diag.kind,
        "severity": diag.severity,
        "line": diag.span.line,
        "col": diag.span.col,
        "message": diagnostic_message(diag),
        "rendered": format_diagnostic(diag, source_lines),
    }

def run(args) -> int:
    """Check (and print) one program. Returns the process exit code."""
    json_out = args.output == "json"

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError) as e:
        msg = f"Error reading {args.file}: {e}"
        print(json.dumps({"status": "failed", "diagnostics": [{"kind": "io-error", "severity": "error", "message": msg}]}) if json_out else msg)
        return 1
    source_lines = source.splitlines()

    try:
        stmts, expr = parse(source)
        ctx = TypeContext(max_unroll=args.max_unroll, base_dir=os.path.dirname(os.path.abspath(args.file)))
        all_diags = check_program(stmts, ctx, expr)
    except (ParseError, LexerError) as e:
        stmts, expr, ctx = [], None, None
        all_diags = [Diagnostic("syntax-error", Span(e.line, e.col, 1), extra={"msg": str(e)})]

    errors = [d for d in all_diags if d.severity == "error"]
    warnings = [d for d in all_diags if d.severity == "warning"]
    status = "failed" if errors else "passed"

    if json_out:
        out = {"status": status, "diagnostics": [_diag_json(d, source_lines) for d in all_diags]}
        if not args.check_only and not errors:
            out["values"] = {name: to_json(typ) for name, typ in ctx.bindings.items()}
            if ctx.result is not None:
                out["result"] = to_json(ctx.result)
        print(json.dumps(out, indent=2))
        return 1 if errors else 0

    for diag in all_diags:
        print(format_diagnostic(diag, source_lines))
        print("")
    if args.check_only:
        if all_diags:
            print(f"Typecheck finished: {len(warnings)} warnings, {len(errors)} errors.")
        else:
            print("Typecheck passed.")
        return 1 if errors else 0
    if errors:
        return 1

    for name, typ in ctx.bindings.items():
        print(f"{name} = {format_value(typ)}")
    if ctx.result is not None:
        print(f"=> {format_value(ctx.result)}")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Uncertain Lang CLI")
    parser.add_argument("file", nargs="?", help="Path to .calc file")
    parser.add_argument("--check-only", action="store_true", help="Only run the type checker")
    parser.add_argument("--output", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--explain", type=str, help="Explain an error code")
    parser.add_argument("--max-unroll", type=int, default=1000, help="Maximum number of loop iterations to unroll (default 1000)")
    parser.add_argument("--lsp", action="store_true", help="Start the Language Server Protocol server")
    parser.add_argument("--version", action="version", version=f"uncertain {get_version()}")

    args = parser.parse_args()

    if args.lsp:
        from uncertain.server import start_server
        start_server()
        sys.exit(0)

    if args.explain:
        code = args.explain
        if code in ERROR_CATALOG:
            print(f"Error Code: {code}")
            print("-" * (12 + len(code)))
            print(ERROR_CATALOG[code])
        else:
            print(f"Unknown error code: {code}. Available codes: {', '.join(ERROR_CATALOG.keys())}")
        sys.exit(0)

    if not args.file:
        parser.error("the following arguments are required: file")
    # Diagnostics echo source lines, which may contain characters the console can't encode (e.g. cp1252 on Windows)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    if args.max_unroll < 0:
        parser.error("--max-unroll must be >= 0")

    try:
        sys.exit(run(args))
    except Exception as e:  # last-resort guard: never show a raw traceback
        print(f"internal error: {type(e).__name__}: {e}\nThis is a bug in Uncertain; please report it.", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
