from typing import List, Iterator, Optional
from uncertain.lexer import Token, tokenize, LexerError
from uncertain.ast_nodes import *

class ParseError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"{message} at {line}:{col}")
        self.line = line
        self.col = col

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

    def expect(self, token_type: str, msg: str) -> Token:
        tok = self.match(token_type)
        if not tok:
            curr = self.current()
            if curr:
                raise ParseError(msg, curr.line, curr.col)
            else:
                raise ParseError(msg + " (unexpected EOF)", -1, -1)
        return tok

    def parse_program(self) -> tuple[List[Stmt], Optional[Expr]]:
        stmts = []
        while self.current():
            if self.current().type in ("LET", "VAR", "WHILE"):
                stmts.append(self.parse_statement())
            elif self.current().type == "IDENT" and self.pos + 1 < len(self.tokens) and self.tokens[self.pos+1].type == "EQUALS":
                stmts.append(self.parse_statement())
            else:
                break
        
        expr = None
        if self.current():
            expr = self.parse_expression()
            
        if self.current():
            raise ParseError("Unexpected tokens at end of program", self.current().line, self.current().col)
            
        return stmts, expr

    def parse_statement(self) -> Stmt:
        tok = self.current()
        if tok.type == "LET":
            return self.parse_let_stmt()
        elif tok.type == "VAR":
            return self.parse_var_stmt()
        elif tok.type == "WHILE":
            return self.parse_while_stmt()
        elif tok.type == "IDENT":
            return self.parse_assign_stmt()
        raise ParseError("Expected statement", tok.line, tok.col)

    def parse_let_stmt(self) -> LetStmt:
        start_tok = self.expect("LET", "Expected 'let'")
        ident = self.expect("IDENT", "Expected identifier after 'let'")
        
        type_ann = None
        if self.match("COLON"):
            type_ann = self.parse_type_ann()
            
        self.expect("EQUALS", "Expected '=' in let statement")
        
        value = self.parse_expression()
        self.expect("SEMI", "Expected ';' after let statement")
        
        length = (value.span.col + value.span.length) - start_tok.col
        span = Span(start_tok.line, start_tok.col, length)
        return LetStmt(ident.value, type_ann, value, span)

    def parse_var_stmt(self) -> VarStmt:
        start_tok = self.expect("VAR", "Expected 'var'")
        ident = self.expect("IDENT", "Expected identifier after 'var'")
        
        type_ann = None
        if self.match("COLON"):
            type_ann = self.parse_type_ann()
            
        self.expect("EQUALS", "Expected '=' in var statement")
        
        value = self.parse_expression()
        self.expect("SEMI", "Expected ';' after var statement")
        
        length = (value.span.col + value.span.length) - start_tok.col
        span = Span(start_tok.line, start_tok.col, length)
        return VarStmt(ident.value, type_ann, value, span)

    def parse_assign_stmt(self) -> AssignStmt:
        ident_tok = self.expect("IDENT", "Expected identifier")
        self.expect("EQUALS", "Expected '=' in assignment")
        value = self.parse_expression()
        self.expect("SEMI", "Expected ';' after assignment")
        
        length = (value.span.col + value.span.length) - ident_tok.col
        span = Span(ident_tok.line, ident_tok.col, length)
        return AssignStmt(ident_tok.value, value, span)

    def parse_while_stmt(self) -> WhileStmt:
        start_tok = self.expect("WHILE", "Expected 'while'")
        self.expect("LPAREN", "Expected '(' after while")
        condition = self.parse_expression()
        self.expect("RPAREN", "Expected ')' after condition")
        
        body = self.parse_block()
        length = (body.span.col + body.span.length) - start_tok.col
        span = Span(start_tok.line, start_tok.col, length)
        return WhileStmt(condition, body, span)

    def parse_block(self) -> Block:
        start_tok = self.expect("LBRACE", "Expected '{'")
        stmts = []
        while self.current() and self.current().type != "RBRACE":
            stmts.append(self.parse_statement())
        end_tok = self.expect("RBRACE", "Expected '}'")
        length = (end_tok.col + len(end_tok.value)) - start_tok.col
        return Block(stmts, Span(start_tok.line, start_tok.col, length))

    def parse_type_ann(self) -> DistLit:
        self.expect("MEASURED", "Expected 'Measured'")
        self.expect("LANGLE", "Expected '<'")
        
        tok = self.current()
        if not tok:
            raise ParseError("Expected 'Normal' or 'Exact'", -1, -1)
            
        if tok.type == "NORMAL":
            self.advance()
            self.expect("LPAREN", "Expected '('")
            mean = self.parse_expression()
            self.expect("COMMA", "Expected ','")
            stddev = self.parse_expression()
            self.expect("RPAREN", "Expected ')'")
            dist = NormalLit(mean, stddev)
        elif tok.type == "UNIFORM":
            self.advance()
            self.expect("LPAREN", "Expected '('")
            min_val = self.parse_expression()
            self.expect("COMMA", "Expected ','")
            max_val = self.parse_expression()
            self.expect("RPAREN", "Expected ')'")
            dist = UniformLit(min_val, max_val)
        elif tok.type == "EXACT":
            self.advance()
            self.expect("LPAREN", "Expected '('")
            val = self.parse_expression()
            self.expect("RPAREN", "Expected ')'")
            dist = ExactLit(val)
        elif tok.type == "EMPIRICAL":
            self.advance()
            self.expect("LPAREN", "Expected '('")
            val = self.parse_expression()
            self.expect("RPAREN", "Expected ')'")
            dist = EmpiricalLit(val)
        elif tok.type == "LOGNORMAL":
            self.advance()
            self.expect("LPAREN", "Expected '('")
            mean = self.parse_expression()
            self.expect("COMMA", "Expected ','")
            stddev = self.parse_expression()
            self.expect("RPAREN", "Expected ')'")
            dist = LogNormalLit(mean, stddev)
        elif tok.type == "POISSON":
            self.advance()
            self.expect("LPAREN", "Expected '('")
            lam = self.parse_expression()
            self.expect("RPAREN", "Expected ')'")
            dist = PoissonLit(lam)
        elif tok.type == "BINOMIAL":
            self.advance()
            self.expect("LPAREN", "Expected '('")
            n = self.parse_expression()
            self.expect("COMMA", "Expected ','")
            p = self.parse_expression()
            self.expect("RPAREN", "Expected ')'")
            dist = BinomialLit(n, p)
        elif tok.type == "GAMMA":
            self.advance()
            self.expect("LPAREN", "Expected '('")
            k = self.parse_expression()
            self.expect("COMMA", "Expected ','")
            theta = self.parse_expression()
            self.expect("RPAREN", "Expected ')'")
            dist = GammaLit(k, theta)
        elif tok.type == "BERNOULLI":
            self.advance()
            self.expect("LPAREN", "Expected '('")
            p = self.parse_expression()
            self.expect("RPAREN", "Expected ')'")
            dist = BernoulliLit(p)
        elif tok.type == "NEGATIVE_BINOMIAL":
            self.advance()
            self.expect("LPAREN", "Expected '('")
            r = self.parse_expression()
            self.expect("COMMA", "Expected ','")
            p = self.parse_expression()
            self.expect("RPAREN", "Expected ')'")
            dist = NegativeBinomialLit(r, p)
        elif tok.type == "GEOMETRIC":
            self.advance()
            self.expect("LPAREN", "Expected '('")
            p = self.parse_expression()
            self.expect("RPAREN", "Expected ')'")
            dist = GeometricLit(p)
        elif tok.type == "EXPONENTIAL":
            self.advance()
            self.expect("LPAREN", "Expected '('")
            lam = self.parse_expression()
            self.expect("RPAREN", "Expected ')'")
            dist = ExponentialLit(lam)
        else:
            raise ParseError("Expected a distribution type", tok.line, tok.col)
            
        self.expect("RANGLE", "Expected '>'")
        return dist

    def parse_expression(self) -> Expr:
        return self.parse_comparison()

    def parse_comparison(self) -> Expr:
        left = self.parse_term()
        op_tok = self.match("LANGLE") or self.match("RANGLE")
        if op_tok:
            right = self.parse_term()
            length = (right.span.col + right.span.length) - left.span.col
            span = Span(left.span.line, left.span.col, length)
            left = BinOp(op_tok.value, left, right, span)
        return left

    def parse_term(self) -> Expr:
        left = self.parse_factor()
        
        while True:
            op_tok = self.match("PLUS") or self.match("MINUS")
            if not op_tok:
                break
            right = self.parse_factor()
            length = (right.span.col + right.span.length) - left.span.col
            span = Span(left.span.line, left.span.col, length)
            left = BinOp(op_tok.value, left, right, span)
            
        return left

    def parse_factor(self) -> Expr:
        left = self.parse_unary()
        
        while True:
            op_tok = self.match("STAR") or self.match("SLASH")
            if not op_tok:
                break
            right = self.parse_unary()
            length = (right.span.col + right.span.length) - left.span.col
            span = Span(left.span.line, left.span.col, length)
            left = BinOp(op_tok.value, left, right, span)
            
        return left

    def parse_unary(self) -> Expr:
        tok = self.match("MINUS")
        if tok:
            expr = self.parse_unary()
            length = (expr.span.col + expr.span.length) - tok.col
            span = Span(tok.line, tok.col, length)
            zero = NumberLit(0.0, Span(tok.line, tok.col, 1))
            return BinOp("-", zero, expr, span)
            
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        tok = self.current()
        if not tok:
            raise ParseError("Unexpected EOF", -1, -1)
            
        if tok.type == "NUMBER":
            self.advance()
            return NumberLit(float(tok.value), Span(tok.line, tok.col, len(tok.value)))
            
        if tok.type == "IDENT":
            self.advance()
            if self.match("LPAREN"):
                args = []
                kwargs = {}
                if not self.match("RPAREN"):
                    while True:
                        is_kwarg = False
                        if self.current() and self.current().type == "IDENT":
                            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == "EQUALS":
                                kwarg_name = self.advance().value
                                self.advance()
                                kwarg_val = self.parse_expression()
                                kwargs[kwarg_name] = kwarg_val
                                is_kwarg = True
                        
                        if not is_kwarg:
                            args.append(self.parse_expression())
                            
                        if not self.match("COMMA"):
                            break
                    end_tok = self.expect("RPAREN", "Expected ')'")
                else:
                    end_tok = self.tokens[self.pos-1]
                    
                length = (end_tok.col + len(end_tok.value)) - tok.col
                return Call(tok.value, args, kwargs, Span(tok.line, tok.col, length))
            else:
                return VarRef(tok.value, Span(tok.line, tok.col, len(tok.value)))
                
        if tok.type == "LPAREN":
            self.advance()
            expr = self.parse_expression()
            self.expect("RPAREN", "Expected ')'")
            return expr
            
        if tok.type == "LBRACKET":
            self.advance()
            elements = []
            if not self.match("RBRACKET"):
                while True:
                    elements.append(self.parse_expression())
                    if not self.match("COMMA"):
                        break
                end_tok = self.expect("RBRACKET", "Expected ']'")
            else:
                end_tok = self.tokens[self.pos-1]
            length = (end_tok.col + len(end_tok.value)) - tok.col
            return ArrayLit(elements, Span(tok.line, tok.col, length))
            
        raise ParseError(f"Unexpected token {tok.value!r}", tok.line, tok.col)

def parse(source: str) -> tuple[List[Stmt], Optional[Expr]]:
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse_program()
