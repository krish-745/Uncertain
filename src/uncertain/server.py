import logging
from pygls.server import LanguageServer
from lsprotocol.types import (
    TEXT_DOCUMENT_DID_CHANGE,
    TEXT_DOCUMENT_DID_OPEN,
    Diagnostic,
    DiagnosticSeverity,
    DidChangeTextDocumentParams,
    DidOpenTextDocumentParams,
    Position,
    Range,
)

from uncertain.parser import parse, ParseError
from uncertain.lexer import LexerError
from uncertain.typechecker import TypeContext, check_stmt

server = LanguageServer("uncertain-language-server", "v0.5.0")

def check_document(ls: LanguageServer, uri: str, source: str):
    diagnostics = []
    
    try:
        stmts, expr = parse(source)
        ctx = TypeContext()
        all_diags = []
        for stmt in stmts:
            diags = check_stmt(stmt, ctx)
            all_diags.extend(diags)
            
        for diag in all_diags:
            severity = DiagnosticSeverity.Error if diag.severity == "error" else DiagnosticSeverity.Warning
            
            # Convert 1-indexed (our AST) to 0-indexed (LSP)
            start = Position(line=diag.span.line - 1, character=diag.span.col - 1)
            end = Position(line=diag.span.line - 1, character=diag.span.col - 1 + diag.span.length)
            
            diagnostics.append(
                Diagnostic(
                    range=Range(start=start, end=end),
                    message=diag.kind + (" (" + diag.extra["msg"] + ")" if diag.extra and "msg" in diag.extra else ""),
                    severity=severity,
                    source="uncertain"
                )
            )
            
    except (ParseError, LexerError) as e:
        # e.line and e.col are 1-indexed
        start = Position(line=max(0, e.line - 1), character=max(0, e.col - 1))
        end = Position(line=max(0, e.line - 1), character=max(0, e.col))
        
        diagnostics.append(
            Diagnostic(
                range=Range(start=start, end=end),
                message=str(e),
                severity=DiagnosticSeverity.Error,
                source="uncertain"
            )
        )
    except Exception as e:
        # Fallback for unexpected errors
        start = Position(line=0, character=0)
        end = Position(line=0, character=1)
        diagnostics.append(
            Diagnostic(
                range=Range(start=start, end=end),
                message=f"Internal Error: {str(e)}",
                severity=DiagnosticSeverity.Error,
                source="uncertain"
            )
        )
        
    ls.publish_diagnostics(uri, diagnostics)

@server.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open(ls: LanguageServer, params: DidOpenTextDocumentParams):
    doc = ls.workspace.get_text_document(params.text_document.uri)
    check_document(ls, params.text_document.uri, doc.source)

@server.feature(TEXT_DOCUMENT_DID_CHANGE)
async def did_change(ls: LanguageServer, params: DidChangeTextDocumentParams):
    doc = ls.workspace.get_text_document(params.text_document.uri)
    check_document(ls, params.text_document.uri, doc.source)

def start_server():
    server.start_io()

if __name__ == "__main__":
    start_server()
