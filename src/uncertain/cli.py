import sys
import argparse
from uncertain.parser import parse, ParseError
from uncertain.lexer import LexerError
from uncertain.typechecker import TypeContext, check_stmt
from uncertain.evaluator import evaluate
from uncertain.diagnostics import format_diagnostic

def main():
    parser = argparse.ArgumentParser(description="Uncertain Lang CLI")
    parser.add_argument("file", help="Path to .calc file")
    parser.add_argument("--check-only", action="store_true", help="Only run the type checker")
    
    args = parser.parse_args()
    
    try:
        with open(args.file, "r") as f:
            source = f.read()
    except Exception as e:
        print(f"Error reading {args.file}: {e}")
        sys.exit(1)
        
    try:
        stmts, expr = parse(source)
    except (ParseError, LexerError) as e:
        print(e)
        sys.exit(1)
        
    ctx = TypeContext()
    all_diags = []
    
    for stmt in stmts:
        diags = check_stmt(stmt, ctx)
        all_diags.extend(diags)
        
    if all_diags:
        source_lines = source.splitlines()
        for diag in all_diags:
            print(format_diagnostic(diag, source_lines))
            print("")
        if any(diag.severity == "error" for diag in all_diags):
            sys.exit(1)
        
    if args.check_only:
        print("Typecheck passed.")
        sys.exit(0)
        
    env = evaluate(stmts, ctx)
    for name, dist in env.items():
        print(f"{name} = (mean={dist.mean:.4f}, stddev={dist.stddev:.4f})")
        
if __name__ == "__main__":
    main()
