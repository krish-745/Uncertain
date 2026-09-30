from uncertain import server

class StubServer:
    def __init__(self):
        self.published = []

    def text_document_publish_diagnostics(self, params):
        self.published.append(params)

def test_check_document_publishes_diagnostics():
    ls = StubServer()
    source = "let a = sensor_read();\nlet b = sqrt(a - 15);\n"
    diags = server.check_document(ls, "file:///tmp/doc.calc", source)

    assert len(ls.published) == 1
    assert ls.published[0].uri == "file:///tmp/doc.calc"
    assert [d.code for d in diags] == ["math-domain-error"]
    rng = diags[0].range
    assert (rng.start.line, rng.start.character) == (1, 8)
    assert rng.end.character > rng.start.character

def test_syntax_error_is_published():
    ls = StubServer()
    diags = server.check_document(ls, "file:///tmp/doc.calc", "let a = ;")
    assert [d.code for d in diags] == ["syntax-error"]

def test_multiline_span_range_is_valid():
    ls = StubServer()
    source = "let a = nope +\n   1;"
    diags = server.check_document(ls, "file:///tmp/doc.calc", source)
    for d in diags:
        assert d.range.end.character >= d.range.start.character

def test_hover_shows_distribution():
    ls = StubServer()
    uri = "file:///tmp/hover.calc"
    source = "let a = sensor_read();\nlet s = {x: a};\nlet b = s.x;"
    server.check_document(ls, uri, source)
    ctx = server._contexts[uri]

    text = server.hover_text(ctx, source, 0, 4)   # `a` on line 1
    assert "**a**" in text and "mean=10.0000" in text
    text = server.hover_text(ctx, source, 2, 10)  # `x` in `s.x`
    assert "**s.x**" in text
    assert server.hover_text(ctx, source, 0, 1) is None  # `let` keyword

def test_hover_on_module_function(tmp_path):
    (tmp_path / "geo.calc").write_text("fn area(r) { return 3.14 * square(r); }")
    source = "import geo as g;\nlet a = g.area(2);"
    uri = (tmp_path / "main.calc").as_uri()
    server.check_document(StubServer(), uri, source)
    ctx = server._contexts[uri]
    assert server.hover_text(ctx, source, 1, 11) == "**fn g.area**(r)"
