import sys
import argparse
import json
from uncertain.parser import parse, ParseError
from uncertain.lexer import LexerError
from uncertain.typechecker import TypeContext, check_stmt
from uncertain.diagnostics import format_diagnostic

ERROR_CATALOG = {
    "uncertain-reuse": "This error occurs when the same uncertain variable is used in both operands of an operation (+, -, *, /) without explicitly accounting for their correlation. Because the type checker assumes independence by default, reusing variables leads to incorrect variance calculations.",
    "type-mismatch": "This error occurs when an explicitly provided type annotation does not match the distribution computed by the type checker.",
    "undefined-var": "This error occurs when a variable is referenced before it has been assigned with a `let` statement.",
    "math-domain-error": "This error occurs when an operation is mathematically undefined, such as dividing by a distribution with a zero mean, or taking the log of a negative number.",
    "approximation-warning": "This warning occurs when non-Normal distributions are combined using operations like addition or multiplication. The system uses a moment-matching approximation to compute the resulting mean and variance, which may not perfectly capture the true distribution shape.",
    "uncertain-branch": "This error occurs when a conditional statement (if, while, for) depends on an uncertain value. Control flow must be deterministic at compile-time."
}

def main():
    parser = argparse.ArgumentParser(description="Uncertain Lang CLI")
    parser.add_argument("file", nargs="?", help="Path to .calc file")
    parser.add_argument("--check-only", action="store_true", help="Only run the type checker")
    parser.add_argument("--output", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--explain", type=str, help="Explain an error code")
    parser.add_argument("--lsp", action="store_true", help="Start the Language Server Protocol server")
    
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
        
    try:
        with open(args.file, "r") as f:
            source = f.read()
    except Exception as e:
        if args.output == "json":
            print(json.dumps({"errors": [{"message": f"Error reading {args.file}: {e}"}]}))
        else:
            print(f"Error reading {args.file}: {e}")
        sys.exit(1)
        
    try:
        stmts, expr = parse(source)
    except (ParseError, LexerError) as e:
        if args.output == "json":
            print(json.dumps({"errors": [{"message": str(e)}]}) )
        else:
            print(e)
        sys.exit(1)
        
    ctx = TypeContext()
    all_diags = []
    
    for stmt in stmts:
        diags = check_stmt(stmt, ctx)
        all_diags.extend(diags)
        
    if all_diags:
        source_lines = source.splitlines()
        errors = [d for d in all_diags if d.severity == "error"]
        warnings = [d for d in all_diags if d.severity == "warning"]
        
        if args.output == "json":
            diag_out = []
            for diag in all_diags:
                diag_out.append({
                    "kind": diag.kind,
                    "severity": diag.severity,
                    "line": diag.span.line,
                    "col": diag.span.col,
                    "message": format_diagnostic(diag, source_lines)
                })
            print(json.dumps({"diagnostics": diag_out}))
            if errors:
                sys.exit(1)
        else:
            for diag in all_diags:
                print(format_diagnostic(diag, source_lines))
                print("")
            if args.check_only:
                print(f"Typecheck finished: {len(warnings)} warnings, {len(errors)} errors.")
            if errors:
                sys.exit(1)
        
    if args.check_only:
        if args.output == "text":
            print("Typecheck passed." if not all_diags else "")
        else:
            if not all_diags:
                print(json.dumps({"status": "passed"}))
        sys.exit(0)
        
    if args.output == "json":
        out = {}
        for name, typ in ctx.bindings.items():
            out[name] = {"mean": typ.dist.mean, "stddev": typ.dist.stddev, "family": typ.dist.family}
        print(json.dumps(out, indent=2))
    else:
        for name, typ in ctx.bindings.items():
            print(f"{name} = (mean={typ.dist.mean:.4f}, stddev={typ.dist.stddev:.4f})")
            
if __name__ == "__main__":
    main()
