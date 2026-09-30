from dataclasses import fields
from typing import List, Iterator, Optional
from uncertain.lexer import Token, tokenize, LexerError
from uncertain.ast_nodes import *

class ParseError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"{message} at {line}:{col}")
        self.line = line
        self.col = col

# Token type of a distribution keyword -> AST literal class. The number of parameters is the
# number of dataclass fields of that class.
DIST_LIT_TOKENS = {
    "NORMAL": NormalLit,
    "UNIFORM": UniformLit,
    "EXACT": ExactLit,
    "EMPIRICAL": EmpiricalLit,
    "LOGNORMAL": LogNormalLit,
    "POISSON": PoissonLit,
    "BINOMIAL": BinomialLit,
    "GAMMA": GammaLit,
    "BERNOULLI": BernoulliLit,
    "NEGATIVE_BINOMIAL": NegativeBinomialLit,
    "GEOMETRIC": GeometricLit,
    "EXPONENTIAL": ExponentialLit,
}

COMPARISON_TOKENS = ("LANGLE", "RANGLE", "LE", "GE", "EQEQ", "NEQ")

def _tok_span(tok: Token) -> Span:
    return Span(tok.line, tok.col, len(tok.value))

def _join(start: Span, end: Span) -> Span:
    """Span from the start of `start` to the end of `end`."""
    if start.line == end.line and end.length > 0:
        return Span(start.line, start.col, (end.col + end.length) - start.col)
    return Span(start.line, start.col, -1)

class Parser:
    def __init__(self, tokens: Iterator[Token]):
        self.tokens = list(tokens)
        self.pos = 0

    def current(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self) -> Optional[Token]:
        tok = self.current()
        self.pos += 1
        return tok

    def match(self, token_type: str) -> Optional[Token]:
        tok = self.current()
        if tok and tok.type == token_type:
            return self.advance()
        return None

    def error(self, msg: str) -> ParseError:
        curr = self.current()
        if curr:
            return ParseError(msg, curr.line, curr.col)
        if self.tokens:
            last = self.tokens[-1]
            return ParseError(msg + " (unexpected end of file)", last.line, last.col + len(last.value))
        return ParseError(msg + " (unexpected end of file)", 1, 1)

    def expect(self, token_type: str, msg: str) -> Token:
        tok = self.match(token_type)
        if not tok:
            raise self.error(msg)
        return tok

    def parse_program(self) -> tuple[List[Stmt], Optional[Expr]]:
        stmts = []
        while self.current():
            if self.current().type in ("LET", "VAR", "WHILE", "IF", "FOR", "FN", "RETURN"):
                stmts.append(self.parse_statement())
            elif self.current().type == "IDENT":
                if self.pos + 1 < len(self.tokens) and self.tokens[self.pos+1].type == "EQUALS":
                    stmts.append(self.parse_statement())
                elif self.current().value == "import":
                    stmts.append(self.parse_statement())
                else:
                    break
            else:
                break

        expr = None
        if self.current():
            expr = self.parse_expression()

        if self.current():
            raise ParseError("Unexpected tokens at end of program", self.current().line, self.current().col)

        return stmts, expr

    def parse_statement(self, require_semi: bool = True) -> Stmt:
        tok = self.current()
        if not tok:
            raise self.error("Expected statement")
        if tok.type == "LET":
            return self.parse_let_stmt(require_semi)
        elif tok.type == "VAR":
            return self.parse_var_stmt(require_semi)
        elif tok.type == "WHILE":
            return self.parse_while_stmt()
        elif tok.type == "IF":
            return self.parse_if_stmt()
        elif tok.type == "FOR":
            return self.parse_for_stmt()
        elif tok.type == "FN":
            return self.parse_fn_def()
        elif tok.type == "RETURN":
            return self.parse_return_stmt(require_semi)
        elif tok.type == "IDENT":
            if tok.value == "import":
                return self.parse_import_stmt(require_semi)
            return self.parse_assign_stmt(require_semi)
        raise ParseError("Expected statement", tok.line, tok.col)

    def parse_import_stmt(self, require_semi: bool = True) -> ImportStmt:
        start_tok = self.expect("IDENT", "Expected 'import'")
        path = [self.expect("IDENT", "Expected module name").value]
        while self.match("DOT"):
            path.append(self.expect("IDENT", "Expected sub-module name").value)

        as_tok = self.expect("IDENT", "Expected 'as'")
        if as_tok.value != "as":
            raise ParseError(f"Expected 'as', got '{as_tok.value}'", as_tok.line, as_tok.col)

        alias_tok = self.expect("IDENT", "Expected alias name")
        if require_semi:
            self.expect("SEMI", "Expected ';' after import statement")

        return ImportStmt(path, alias_tok.value, _join(_tok_span(start_tok), _tok_span(alias_tok)))

    def _parse_binding(self, keyword: str):
        start_tok = self.expect(keyword.upper(), f"Expected '{keyword}'")
        ident = self.expect("IDENT", f"Expected identifier after '{keyword}'")

        type_ann = None
        if self.match("COLON"):
            type_ann = self.parse_type_ann()

        self.expect("EQUALS", f"Expected '=' in {keyword} statement")
        value = self.parse_expression()
        return start_tok, ident, type_ann, value

    def parse_let_stmt(self, require_semi: bool = True) -> LetStmt:
        start_tok, ident, type_ann, value = self._parse_binding("let")
        if require_semi:
            self.expect("SEMI", "Expected ';' after let statement")
        return LetStmt(ident.value, type_ann, value, _join(_tok_span(start_tok), value.span))

    def parse_var_stmt(self, require_semi: bool = True) -> VarStmt:
        start_tok, ident, type_ann, value = self._parse_binding("var")
        if require_semi:
            self.expect("SEMI", "Expected ';' after var statement")
        return VarStmt(ident.value, type_ann, value, _join(_tok_span(start_tok), value.span))

    def parse_assign_stmt(self, require_semi: bool = True) -> AssignStmt:
        ident_tok = self.expect("IDENT", "Expected identifier")
        self.expect("EQUALS", "Expected '=' in assignment")
        value = self.parse_expression()
        if require_semi:
            self.expect("SEMI", "Expected ';' after assignment")
        return AssignStmt(ident_tok.value, value, _join(_tok_span(ident_tok), value.span))

    def _span_from(self, start_tok: Token) -> Span:
        """Span from `start_tok` to the most recently consumed token."""
        return _join(_tok_span(start_tok), _tok_span(self.tokens[self.pos - 1]))

    def parse_while_stmt(self) -> WhileStmt:
        start_tok = self.expect("WHILE", "Expected 'while'")
        self.expect("LPAREN", "Expected '(' after while")
        condition = self.parse_expression()
        self.expect("RPAREN", "Expected ')' after condition")
        body = self.parse_block()
        return WhileStmt(condition, body, self._span_from(start_tok))

    def parse_if_stmt(self) -> IfStmt:
        start_tok = self.expect("IF", "Expected 'if'")
        self.expect("LPAREN", "Expected '(' after if")
        condition = self.parse_expression()
        self.expect("RPAREN", "Expected ')' after condition")

        true_body = self.parse_block()

        false_body = None
        if self.match("ELSE"):
            false_body = self.parse_block()

        return IfStmt(condition, true_body, false_body, self._span_from(start_tok))

    def parse_for_stmt(self) -> ForStmt:
        start_tok = self.expect("FOR", "Expected 'for'")
        self.expect("LPAREN", "Expected '(' after for")

        init = self.parse_statement(require_semi=True)
        condition = self.parse_expression()
        self.expect("SEMI", "Expected ';' after condition")
        increment = self.parse_statement(require_semi=False)

        self.expect("RPAREN", "Expected ')' after increment")

        body = self.parse_block()
        return ForStmt(init, condition, increment, body, self._span_from(start_tok))

    def parse_fn_def(self) -> FnDefStmt:
        start_tok = self.expect("FN", "Expected 'fn'")
        name_tok = self.expect("IDENT", "Expected function name")

        self.expect("LPAREN", "Expected '(' after function name")
        args = []
        if not self.match("RPAREN"):
            while True:
                arg_name_tok = self.expect("IDENT", "Expected argument name")
                arg_type = None
                if self.match("COLON"):
                    arg_type = self.parse_type_ann()
                args.append(ArgDef(arg_name_tok.value, arg_type, self._span_from(arg_name_tok)))

                if not self.match("COMMA"):
                    break
            self.expect("RPAREN", "Expected ')' after arguments")

        return_type = None
        if self.match("ARROW"):
            return_type = self.parse_type_ann()

        body = self.parse_block()
        return FnDefStmt(name_tok.value, args, return_type, body, self._span_from(start_tok))

    def parse_return_stmt(self, require_semi: bool = True) -> ReturnStmt:
        start_tok = self.expect("RETURN", "Expected 'return'")
        value = self.parse_expression()
        span = _join(_tok_span(start_tok), value.span)
        if require_semi:
            self.expect("SEMI", "Expected ';' after return value")
        return ReturnStmt(value, span)

    def parse_block(self) -> Block:
        start_tok = self.expect("LBRACE", "Expected '{'")
        stmts = []
        while self.current() and self.current().type != "RBRACE":
            stmts.append(self.parse_statement())
        self.expect("RBRACE", "Expected '}'")
        return Block(stmts, self._span_from(start_tok))

    def parse_type_ann(self) -> DistLit:
        self.expect("MEASURED", "Expected 'Measured'")
        self.expect("LANGLE", "Expected '<'")

        tok = self.current()
        if not tok or tok.type not in DIST_LIT_TOKENS:
            raise self.error("Expected a distribution type (e.g. Normal, Uniform, Exact)")
        self.advance()
        cls = DIST_LIT_TOKENS[tok.type]

        self.expect("LPAREN", f"Expected '(' after '{tok.value}'")
        params = []
        for i in range(len(fields(cls))):
            if i > 0:
                self.expect("COMMA", f"Expected ',' ({tok.value} takes {len(fields(cls))} parameters)")
            params.append(self.parse_expression())
        self.expect("RPAREN", f"Expected ')' ({tok.value} takes {len(fields(cls))} parameter(s))")

        # `Measured<...>= value` lexes the closing '>' and the '=' together as '>='; split it.
        closing = self.current()
        if closing and closing.type == "GE":
            self.tokens[self.pos] = Token("EQUALS", "=", closing.line, closing.col + 1)
        else:
            self.expect("RANGLE", "Expected '>'")
        return cls(*params)

    def parse_expression(self) -> Expr:
        return self.parse_comparison()

    def parse_comparison(self) -> Expr:
        left = self.parse_term()
        tok = self.current()
        if tok and tok.type in COMPARISON_TOKENS:
            self.advance()
            right = self.parse_term()
            left = BinOp(tok.value, left, right, _join(left.span, right.span))
        return left

    def parse_term(self) -> Expr:
        left = self.parse_factor()

        while True:
            op_tok = self.match("PLUS") or self.match("MINUS")
            if not op_tok:
                break
            right = self.parse_factor()
            left = BinOp(op_tok.value, left, right, _join(left.span, right.span))

        return left

    def parse_factor(self) -> Expr:
        left = self.parse_unary()

        while True:
            op_tok = self.match("STAR") or self.match("SLASH")
            if not op_tok:
                break
            right = self.parse_unary()
            left = BinOp(op_tok.value, left, right, _join(left.span, right.span))

        return left

    def parse_unary(self) -> Expr:
        tok = self.match("MINUS")
        if tok:
            expr = self.parse_unary()
            zero = NumberLit(0.0, Span(tok.line, tok.col, 1))
            return BinOp("-", zero, expr, _join(_tok_span(tok), expr.span))

        return self.parse_primary()

    def parse_primary(self) -> Expr:
        tok = self.current()
        if not tok:
            raise self.error("Expected an expression")

        node = None
        if tok.type == "NUMBER":
            self.advance()
            node = NumberLit(float(tok.value), _tok_span(tok))

        elif tok.type == "IDENT":
            self.advance()
            if self.match("LPAREN"):
                args, kwargs = self.parse_call_args()
                node = Call(tok.value, args, kwargs, self._span_from(tok))
            else:
                node = VarRef(tok.value, _tok_span(tok))

        elif tok.type == "LPAREN":
            self.advance()
            node = self.parse_expression()
            self.expect("RPAREN", "Expected ')'")

        elif tok.type == "LBRACKET":
            self.advance()
            elements = []
            if not self.match("RBRACKET"):
                while True:
                    elements.append(self.parse_expression())
                    if not self.match("COMMA"):
                        break
                self.expect("RBRACKET", "Expected ']'")
            node = ArrayLit(elements, self._span_from(tok))

        elif tok.type == "LBRACE":
            self.advance()
            struct_fields = {}
            if not self.match("RBRACE"):
                while True:
                    if not self.current() or self.current().type != "IDENT":
                        raise self.error("Expected field name in struct literal")
                    name_tok = self.advance()
                    self.expect("COLON", "Expected ':' after field name")
                    struct_fields[name_tok.value] = self.parse_expression()
                    if not self.match("COMMA"):
                        break
                self.expect("RBRACE", "Expected '}'")
            node = StructLit(struct_fields, self._span_from(tok))

        else:
            raise ParseError(f"Unexpected token {tok.value!r}", tok.line, tok.col)

        while self.current() and self.current().type in ("LBRACKET", "DOT"):
            if self.current().type == "LBRACKET":
                self.advance()
                index = self.parse_expression()
                end_tok = self.expect("RBRACKET", "Expected ']'")
                node = ArrayAccess(node, index, _join(node.span, _tok_span(end_tok)))
            else:
                self.advance()
                if not self.current() or self.current().type != "IDENT":
                    raise self.error("Expected field name after '.'")
                name_tok = self.advance()
                if self.match("LPAREN"):
                    # `module.function(...)`
                    args, kwargs = self.parse_call_args()
                    node = Call(name_tok.value, args, kwargs, _join(node.span, _tok_span(self.tokens[self.pos - 1])), target=node)
                else:
                    node = FieldAccess(node, name_tok.value, _join(node.span, _tok_span(name_tok)))

        return node

    def parse_call_args(self) -> tuple[list[Expr], dict[str, Expr]]:
        """Parse call arguments after the opening '(' up to and including the closing ')'."""
        args = []
        kwargs = {}
        if self.match("RPAREN"):
            return args, kwargs
        while True:
            tok = self.current()
            if tok and tok.type == "IDENT" and self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == "EQUALS":
                self.advance()
                self.advance()
                kwargs[tok.value] = self.parse_expression()
            else:
                args.append(self.parse_expression())
            if not self.match("COMMA"):
                break
        self.expect("RPAREN", "Expected ')'")
        return args, kwargs

def parse(source: str) -> tuple[List[Stmt], Optional[Expr]]:
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse_program()
