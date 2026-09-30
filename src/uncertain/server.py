import os
from typing import Optional

from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer
from pygls.uris import to_fs_path

from uncertain.parser import parse, ParseError
from uncertain.lexer import LexerError, tokenize
from uncertain.typechecker import TypeContext, MeasuredType, check_program
from uncertain.diagnostics import diagnostic_message
from uncertain.output import format_value
from uncertain.cli import get_version

server = LanguageServer("uncertain-language-server", get_version())

# uri -> TypeContext of the last successful check, used to answer hover requests
_contexts: dict[str, TypeContext] = {}

def _base_dir(uri: str) -> str:
    path = to_fs_path(uri) if uri.startswith("file:") else None
    return os.path.dirname(path) if path else os.getcwd()

def _range(line: int, col: int, length: int, source_lines: list[str]) -> lsp.Range:
    """Convert a 1-indexed span to a 0-indexed LSP range. length <= 0 means 'to end of line'."""
    line0 = max(0, line - 1)
    start = max(0, col - 1)
    line_len = len(source_lines[line0]) if line0 < len(source_lines) else start + 1
    end = start + length if length > 0 else max(start + 1, line_len)
    return lsp.Range(start=lsp.Position(line=line0, character=start), end=lsp.Position(line=line0, character=end))

def compute_diagnostics(uri: str, source: str) -> list[lsp.Diagnostic]:
    source_lines = source.splitlines()
    try:
        stmts, expr = parse(source)
    except (ParseError, LexerError) as e:
        return [lsp.Diagnostic(range=_range(e.line, e.col, 1, source_lines), message=str(e),
                               severity=lsp.DiagnosticSeverity.Error, source="uncertain", code="syntax-error")]

    ctx = TypeContext(base_dir=_base_dir(uri))
    diags = check_program(stmts, ctx, expr)
    _contexts[uri] = ctx
    return [
        lsp.Diagnostic(
            range=_range(d.span.line, d.span.col, d.span.length, source_lines),
            message=diagnostic_message(d),
            severity=lsp.DiagnosticSeverity.Error if d.severity == "error" else lsp.DiagnosticSeverity.Warning,
            source="uncertain",
            code=d.kind,
        )
        for d in diags
    ]

def check_document(ls: LanguageServer, uri: str, source: str) -> list[lsp.Diagnostic]:
    diagnostics = compute_diagnostics(uri, source)
    ls.text_document_publish_diagnostics(lsp.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics))
    return diagnostics

def hover_text(ctx: TypeContext, source: str, line: int, character: int) -> Optional[str]:
    """Markdown describing the variable (or `a.b.c` field path) under a 0-indexed position."""
    try:
        tokens = list(tokenize(source))
    except LexerError:
        return None

    idx = next((i for i, t in enumerate(tokens)
                if t.type == "IDENT" and t.line == line + 1 and t.col - 1 <= character < t.col - 1 + len(t.value)), None)
    if idx is None:
        return None

    # Walk back over `a.b.` so that hovering `c` in `a.b.c` resolves the full field path
    path = [tokens[idx].value]
    i = idx
    while i >= 2 and tokens[i - 1].type == "DOT" and tokens[i - 2].type == "IDENT":
        path.insert(0, tokens[i - 2].value)
        i -= 2

    typ: Optional[MeasuredType] = ctx.lookup(path[0])
    for field in path[1:]:
        if typ is None or not isinstance(typ.dist, dict):
            typ = None
            break
        typ = typ.dist.get(field)

    name = ".".join(path)
    if typ is not None:
        family = f" ({typ.dist.family})" if hasattr(typ.dist, "family") else ""
        return f"**{name}**{family}: `{format_value(typ)}`"
    fn = ctx.lookup_fn(path[0]) if len(path) == 1 else None
    if fn is not None:
        return f"**fn {fn.name}**({', '.join(a.name for a in fn.args)})"
    return None

@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params: lsp.DidOpenTextDocumentParams):
    doc = ls.workspace.get_text_document(params.text_document.uri)
    check_document(ls, doc.uri, doc.source)

@server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: LanguageServer, params: lsp.DidChangeTextDocumentParams):
    doc = ls.workspace.get_text_document(params.text_document.uri)
    check_document(ls, doc.uri, doc.source)

@server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
def did_close(ls: LanguageServer, params: lsp.DidCloseTextDocumentParams):
    _contexts.pop(params.text_document.uri, None)
    ls.text_document_publish_diagnostics(lsp.PublishDiagnosticsParams(uri=params.text_document.uri, diagnostics=[]))

@server.feature(lsp.TEXT_DOCUMENT_HOVER)
def hover(ls: LanguageServer, params: lsp.HoverParams) -> Optional[lsp.Hover]:
    ctx = _contexts.get(params.text_document.uri)
    if ctx is None:
        return None
    doc = ls.workspace.get_text_document(params.text_document.uri)
    text = hover_text(ctx, doc.source, params.position.line, params.position.character)
    if text is None:
        return None
    return lsp.Hover(contents=lsp.MarkupContent(kind=lsp.MarkupKind.Markdown, value=text))

def start_server():
    server.start_io()

if __name__ == "__main__":
    start_server()
