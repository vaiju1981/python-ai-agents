"""Textual DSL parser (PR-D2).

A small lexer + recursive-descent parser for the analytics DSL:

    SELECT <metric>[, <metric> ...]
           [BY <dim>[, <dim> ...]]
           [WHERE <filter> [AND <filter> ...]]
           [SINCE <n> DAYS|WEEKS|MONTHS]
           [BETWEEN <date> AND <date>]
           [ORDER BY <metric|dim> [ASC|DESC]]
           [LIMIT <n>]

A ``<metric>`` is a catalog name, a ``table.column`` base ref, or an inline
``expr AS alias`` (mirroring ``query_planner.derivedMetrics``). Filters support
``= != <> < <= > >= IN (...) LIKE``. The parser produces a :class:`DslQuery`
(see ``ast.py``), which compiles to a ``QuerySpec`` via ``to_spec``.

This is a *real* grammar (no regex hack) so malformed input yields a scoped
:class:`DslParseError` naming the offending clause.
"""

from __future__ import annotations

from collections import namedtuple

from demos.analytics.src.analytics.dsl.ast import DslFilter, DslQuery

Tok = namedtuple("Tok", ["kind", "text"])

RESERVED = {
    "select", "by", "where", "and", "since", "between", "order",
    "asc", "desc", "limit", "as", "in", "like", "days", "weeks", "months",
}
COMPARISONS = {"=", "!=", "<>", "<", "<=", ">", ">="}


class DslParseError(ValueError):
    """A scoped failure parsing the DSL, naming the offending clause/token."""


def parse(text: str) -> DslQuery:
    """Parse a DSL string into a :class:`DslQuery` (or raise ``DslParseError``)."""
    toks = _tokenize(text)
    if not toks:
        raise DslParseError("empty query")
    return _Parser(toks).parse_query()


def _tokenize(text: str) -> list[Tok]:
    tokens: list[Tok] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            j, buf = i + 1, []
            while j < n:
                if text[j] == "\\" and j + 1 < n:
                    buf.append(text[j + 1])
                    j += 2
                    continue
                if text[j] == quote:
                    break
                buf.append(text[j])
                j += 1
            if j >= n:
                raise DslParseError("unterminated string literal")
            # Double-quoted strings are identifiers (multi-word metric/dim names);
            # single-quoted strings are scalar values.
            tokens.append(Tok("ID" if quote == '"' else "STR", "".join(buf)))
            i = j + 1
            continue
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            while j < n and (text[j].isdigit() or text[j] == "."):
                j += 1
            tokens.append(Tok("NUM", text[i:j]))
            i = j
            continue
        if c in "()":
            tokens.append(Tok("LP" if c == "(" else "RP", c))
            i += 1
            continue
        if c == ",":
            tokens.append(Tok("COMMA", c))
            i += 1
            continue
        if c in "=<>!+-*/":
            two = text[i : i + 2]
            if two in ("<=", ">=", "!=", "<>"):
                tokens.append(Tok("OP", two))
                i += 2
                continue
            tokens.append(Tok("OP", c))
            i += 1
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] in "._"):
                j += 1
            tokens.append(Tok("ID", text[i:j]))
            i = j
            continue
        raise DslParseError(f"unexpected character '{c}' at position {i}")
    return tokens


class _Parser:
    def __init__(self, tokens: list[Tok]) -> None:
        self.toks = tokens
        self.i = 0

    # -- token helpers -----------------------------------------------------
    def peek(self) -> Tok | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def peek_kw(self, word: str) -> bool:
        t = self.peek()
        return t is not None and t.kind == "ID" and t.text.lower() == word

    def next(self) -> Tok:
        if self.i >= len(self.toks):
            raise DslParseError("unexpected end of query")
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, kind: str, text: str | None = None) -> Tok:
        t = self.peek()
        if t is None or t.kind != kind:
            want = text or kind
            raise DslParseError(f"expected {want}")
        if text is not None and t.text != text:
            raise DslParseError(f"expected '{text}' but got '{t.text}'")
        return self.next()

    def expect_id(self, word: str | None = None) -> Tok:
        t = self.peek()
        if t is None or t.kind != "ID":
            raise DslParseError("expected identifier" + (f" '{word}'" if word else ""))
        if word is not None and t.text.lower() != word:
            raise DslParseError(f"expected '{word}' but got '{t.text}'")
        return self.next()

    def expect_kw(self, word: str) -> None:
        self.expect_id(word)

    # -- top level ---------------------------------------------------------
    def parse_query(self) -> DslQuery:
        self.expect_kw("select")
        metrics = self.parse_metric_list()
        dims: list[str] = []
        if self.peek_kw("by"):
            self.next()
            dims = self.parse_dim_list()
        filters: list[DslFilter] = []
        if self.peek_kw("where"):
            self.next()
            filters = self.parse_filter_list()

        last_days = None
        between_start = between_end = None
        if self.peek_kw("since"):
            self.next()
            last_days = self.parse_duration()
        if self.peek_kw("between"):
            self.next()
            between_start, between_end = self.parse_between()

        order_by = None
        descending = True
        if self.peek_kw("order"):
            self.next()
            self.expect_kw("by")
            order_by = self.parse_operand()
            if self.peek_kw("asc"):
                self.next()
                descending = False
            elif self.peek_kw("desc"):
                self.next()
                descending = True

        limit = None
        if self.peek_kw("limit"):
            self.next()
            limit = int(self.expect("NUM").text)

        if self.peek() is not None:
            raise DslParseError(f"unexpected token '{self.peek().text}'")
        return DslQuery(
            metrics=tuple(metrics),
            dimensions=tuple(dims),
            filters=tuple(filters),
            last_days=last_days,
            between_start=between_start,
            between_end=between_end,
            order_by=order_by,
            descending=descending,
            limit=limit,
        )

    # -- metrics / dimensions ---------------------------------------------
    def parse_metric_list(self) -> list[str]:
        out = [self.parse_metric()]
        while self.peek() is not None and self.peek().kind == "COMMA":
            self.next()
            out.append(self.parse_metric())
        return out

    def parse_metric(self) -> str:
        text, compound = self.parse_expr()
        alias = None
        if self.peek_kw("as"):
            self.next()
            alias = self.expect("ID").text
        if compound and alias is None:
            raise DslParseError("inline expression requires AS <alias>")
        return f"{text} AS {alias}" if alias else text

    def parse_dim_list(self) -> list[str]:
        out = [self.parse_operand()]
        while self.peek() is not None and self.peek().kind == "COMMA":
            self.next()
            out.append(self.parse_operand())
        return out

    def parse_operand(self) -> str:
        t = self.expect("ID")
        if t.text.lower() in RESERVED:
            raise DslParseError(f"unexpected keyword '{t.text}'")
        return t.text

    # -- expression (for inline metrics) ----------------------------------
    def parse_expr(self) -> tuple[str, bool]:
        start = self.i
        compound = self.parse_additive()
        text = " ".join(t.text for t in self.toks[start : self.i])
        return text, compound

    def parse_additive(self) -> bool:
        compound = self.parse_term()
        while True:
            t = self.peek()
            if t is not None and t.kind == "OP" and t.text in ("+", "-"):
                self.next()
                self.parse_term()
                compound = True
            else:
                break
        return compound

    def parse_term(self) -> bool:
        compound = self.parse_factor()
        while True:
            t = self.peek()
            if t is not None and t.kind == "OP" and t.text in ("*", "/"):
                self.next()
                self.parse_factor()
                compound = True
            else:
                break
        return compound

    def parse_factor(self) -> bool:
        t = self.peek()
        if t is None:
            raise DslParseError("expected operand")
        if t.kind == "OP" and t.text == "-":
            self.next()
            self.parse_factor()
            return True
        if t.kind == "OP" and t.text == "+":
            self.next()
            self.parse_factor()
            return False
        if t.kind == "LP":
            self.next()
            self.parse_additive()
            self.expect("RP", ")")
            return False
        if t.kind == "NUM":
            self.next()
            return False
        if t.kind == "ID":
            if t.text.lower() in RESERVED:
                raise DslParseError(f"unexpected keyword '{t.text}' in expression")
            self.next()
            return False
        raise DslParseError(f"expected operand but got '{t.text}'")

    # -- filters ----------------------------------------------------------
    def parse_filter_list(self) -> list[DslFilter]:
        out = [self.parse_filter()]
        while self.peek_kw("and"):
            self.next()
            out.append(self.parse_filter())
        return out

    def parse_filter(self) -> DslFilter:
        col = self.parse_operand()
        t = self.peek()
        if t is not None and t.kind == "OP" and t.text in COMPARISONS:
            op = self.next().text
        elif t is not None and t.kind == "ID" and t.text.lower() in ("in", "like"):
            op = self.next().text.upper()
        else:
            raise DslParseError(f"expected comparison operator after '{col}'")
        value = self.parse_filter_value(op)
        return DslFilter(col, op, value)

    def parse_filter_value(self, op: str) -> str | list[str]:
        if op.upper() == "IN":
            self.expect("LP", "(")
            vals: list[str] = [self.parse_scalar()]
            while self.peek() is not None and self.peek().kind == "COMMA":
                self.next()
                vals.append(self.parse_scalar())
            self.expect("RP", ")")
            return vals
        return self.parse_scalar()

    def parse_scalar(self) -> str:
        t = self.peek()
        if t is None:
            raise DslParseError("expected a value")
        if t.kind in ("NUM", "STR"):
            self.next()
            return t.text
        if t.kind == "ID":
            if t.text.lower() in RESERVED:
                raise DslParseError(f"unexpected keyword '{t.text}' as a value")
            self.next()
            return t.text
        raise DslParseError(f"expected a value but got '{t.text}'")

    # -- time -------------------------------------------------------------
    def parse_duration(self) -> int:
        n = self.expect("NUM")
        unit = self.expect("ID").text.lower()
        if unit == "days":
            return int(n.text)
        if unit == "weeks":
            return int(n.text) * 7
        if unit == "months":
            return int(n.text) * 30
        raise DslParseError(f"unknown time unit '{unit}' (expected DAYS/WEEKS/MONTHS)")

    def parse_between(self) -> tuple[str, str]:
        start = self.parse_date()
        self.expect_kw("and")
        end = self.parse_date()
        return start, end

    def parse_date(self) -> str:
        t = self.peek()
        if t is not None and t.kind in ("STR", "NUM"):
            self.next()
            return t.text
        raise DslParseError("expected a date literal in BETWEEN")
