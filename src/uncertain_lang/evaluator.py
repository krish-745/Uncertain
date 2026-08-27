from uncertain_lang.ast_nodes import LetStmt
from uncertain_lang.distributions import Dist
from uncertain_lang.typechecker import TypeContext

def evaluate(typed_program: list[LetStmt], ctx: TypeContext) -> dict[str, Dist]:
    env: dict[str, Dist] = {}
    for stmt in typed_program:
        typ = ctx.lookup(stmt.name)
        if typ:
            env[stmt.name] = typ.dist
    return env
