"""
GAL Lexer  [lex]
Tokenises a GAL source file into a flat list of Token objects.

Handles (per specification):
  § 4.2  line comments (//) and block comments (/* */) — not nested
  § 4.4  keywords
  § 4.5  identifiers
  § 4.6  integer / float / char / string / raw-string ('\"\"\"') literals
         true / false are treated as BOOL_LIT (not in keyword list)
  § 4.7  operators and punctuators
  § 4.8  visibility modifier  *  (same token as MUL; context decides)
  § 22.1 backtick-enclosed operator names  `+`  `for`  etc.
  § 28   pragma delimiters  [.  .]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


# ---------------------------------------------------------------------------
# Token kinds
# ---------------------------------------------------------------------------

class TK(Enum):
    # --- keywords (§ 4.4) ---------------------------------------------------
    KW_ADDR       = auto()
    KW_ALIAS      = auto()
    KW_ASM        = auto()
    KW_AUTO       = auto()
    KW_BOOL       = auto()   # the *type* keyword
    KW_BREAK      = auto()
    KW_CASE       = auto()
    KW_CAST       = auto()
    KW_CHAR       = auto()   # the *type* keyword
    KW_CONST      = auto()
    KW_CONTINUE   = auto()
    KW_ELIF       = auto()
    KW_ELSE       = auto()
    KW_ENUM       = auto()
    KW_FN         = auto()
    KW_FOR        = auto()
    KW_GOTO       = auto()
    KW_IF         = auto()
    KW_IMUT       = auto()
    KW_IMPORT     = auto()
    KW_INCLUDE    = auto()
    KW_LABEL      = auto()
    KW_LET        = auto()
    KW_MIXIN      = auto()
    KW_MUT        = auto()
    KW_OBJECT     = auto()
    KW_PTR        = auto()
    KW_REF        = auto()
    KW_RETURN     = auto()
    KW_TEMPLATE   = auto()
    KW_THIS       = auto()   # lowercase
    KW_THIS_TYPE  = auto()   # uppercase This
    KW_TYPE       = auto()
    KW_VAL        = auto()
    KW_VAR        = auto()
    KW_VOID       = auto()
    KW_WHILE      = auto()

    # --- built-in type names (fundamental types, §5) ------------------------
    # Stored as identifiers but we give them their own kind so the parser
    # doesn't need a separate lookup table.
    KW_INTB  = auto(); KW_INTW  = auto(); KW_INTD  = auto()
    KW_INTQ  = auto(); KW_INTO  = auto()
    KW_UINTB = auto(); KW_UINTW = auto(); KW_UINTD = auto()
    KW_UINTQ = auto(); KW_UINTO = auto()
    KW_USIZE = auto(); KW_SSIZE = auto()
    KW_FLOATW = auto(); KW_FLOATD = auto()
    KW_FLOATQ = auto(); KW_FLOATO = auto()

    # --- literals -----------------------------------------------------------
    INT_LIT        = auto()   # value: int,  base: 10|16|2
    FLOAT_LIT      = auto()   # value: float (raw string kept for C emit)
    CHAR_LIT       = auto()   # value: int (code point)
    STRING_LIT     = auto()   # value: bytes (decoded escape sequences)
    RAW_STRING_LIT = auto()   # value: str   (verbatim content)
    BOOL_LIT       = auto()   # value: bool  (true/false)

    # --- backtick operator name  `sym`  (§22.1) ----------------------------
    BACKTICK_NAME  = auto()   # value: str, the name between backticks

    # --- operators / punctuators (§ 4.7, §28) ------------------------------
    LBRACE      = auto()  # {
    RBRACE      = auto()  # }
    LBRACKET    = auto()  # [
    RBRACKET    = auto()  # ]
    LPAREN      = auto()  # (
    RPAREN      = auto()  # )
    COLONCOLON  = auto()  # ::
    DOT         = auto()  # .
    ARROW       = auto()  # ->
    STAR        = auto()  # *   (multiply / deref-by-subscript / vis-modifier)
    PLUS        = auto()  # +
    MINUS       = auto()  # -
    SLASH       = auto()  # /
    PERCENT     = auto()  # %
    EQ          = auto()  # =
    PLUSEQ      = auto()  # +=
    MINUSEQ     = auto()  # -=
    SLASHEQ     = auto()  # /=
    PERCENTEQ   = auto()  # %=
    EQEQ        = auto()  # ==
    BANGEQ      = auto()  # !=
    STAREQ      = auto()  # *=   (not in spec operator list but needed)
    LT          = auto()  # <
    GT          = auto()  # >
    LTEQ        = auto()  # <=
    GTEQ        = auto()  # >=
    AMPAMP      = auto()  # &&
    PIPEPIPE    = auto()  # ||
    BANG        = auto()  # !
    PIPE        = auto()  # |
    DOTDOT      = auto()  # ..
    COMMA       = auto()  # ,
    COLON       = auto()  # :
    SEMI        = auto()  # ;
    UNDERSCORE  = auto()  # _

    # pragma brackets  [.  .]
    PRAGMA_OPEN  = auto()  # [.
    PRAGMA_CLOSE = auto()  # .]

    # --- structural ---------------------------------------------------------
    IDENT = auto()
    EOF   = auto()


# ---------------------------------------------------------------------------
# Keyword tables
# ---------------------------------------------------------------------------

_KEYWORDS: dict[str, TK] = {
    "addr":     TK.KW_ADDR,
    "alias":    TK.KW_ALIAS,
    "asm":      TK.KW_ASM,
    "auto":     TK.KW_AUTO,
    "bool":     TK.KW_BOOL,
    "break":    TK.KW_BREAK,
    "case":     TK.KW_CASE,
    "cast":     TK.KW_CAST,
    "char":     TK.KW_CHAR,
    "const":    TK.KW_CONST,
    "continue": TK.KW_CONTINUE,
    "elif":     TK.KW_ELIF,
    "else":     TK.KW_ELSE,
    "enum":     TK.KW_ENUM,
    "fn":       TK.KW_FN,
    "for":      TK.KW_FOR,
    "goto":     TK.KW_GOTO,
    "if":       TK.KW_IF,
    "imut":     TK.KW_IMUT,
    "import":   TK.KW_IMPORT,
    "include":  TK.KW_INCLUDE,
    "label":    TK.KW_LABEL,
    "let":      TK.KW_LET,
    "mixin":    TK.KW_MIXIN,
    "mut":      TK.KW_MUT,
    "object":   TK.KW_OBJECT,
    "ptr":      TK.KW_PTR,
    "ref":      TK.KW_REF,
    "return":   TK.KW_RETURN,
    "template": TK.KW_TEMPLATE,
    "this":     TK.KW_THIS,
    "This":     TK.KW_THIS_TYPE,
    "type":     TK.KW_TYPE,
    "val":      TK.KW_VAL,
    "var":      TK.KW_VAR,
    "void":     TK.KW_VOID,
    "while":    TK.KW_WHILE,
    # fundamental types (§5) — not in spec keyword list but reserved
    "intb":   TK.KW_INTB,  "intw":   TK.KW_INTW,  "intd":   TK.KW_INTD,
    "intq":   TK.KW_INTQ,  "into":   TK.KW_INTO,
    "uintb":  TK.KW_UINTB, "uintw":  TK.KW_UINTW, "uintd":  TK.KW_UINTD,
    "uintq":  TK.KW_UINTQ, "uinto":  TK.KW_UINTO,
    "usize":  TK.KW_USIZE, "ssize":  TK.KW_SSIZE,
    "floatw": TK.KW_FLOATW,"floatd": TK.KW_FLOATD,
    "floatq": TK.KW_FLOATQ,"floato": TK.KW_FLOATO,
    # boolean literals (§5.4 — values of type bool)
    "true":  None,  # handled specially below → BOOL_LIT
    "false": None,
}


# ---------------------------------------------------------------------------
# Token dataclass
# ---------------------------------------------------------------------------

@dataclass
class Token:
    kind:   TK
    text:   str           # raw source text
    value:  object        # parsed value (int/float/str/bool/None)
    line:   int
    col:    int

    def __repr__(self) -> str:
        return f"Token({self.kind.name}, {self.text!r}, line={self.line}, col={self.col})"


# ---------------------------------------------------------------------------
# LexError
# ---------------------------------------------------------------------------

class LexError(Exception):
    def __init__(self, msg: str, line: int, col: int) -> None:
        super().__init__(f"Lex error at line {line}, col {col}: {msg}")
        self.line = line
        self.col  = col


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

class Lexer:
    """
    Single-pass hand-written lexer.  Call `tokenize()` to get the token list.
    The final token is always EOF.
    """

    def __init__(self, source: str, filename: str = "<input>") -> None:
        self._src      = source
        self._filename = filename
        self._pos      = 0
        self._line     = 1
        self._col      = 1
        self._tokens:  List[Token] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(self) -> List[Token]:
        while self._pos < len(self._src):
            self._skip_whitespace_and_comments()
            if self._pos >= len(self._src):
                break
            self._next_token()
        self._tokens.append(Token(TK.EOF, "", None, self._line, self._col))
        return self._tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _ch(self) -> str:
        """Current character, or '' at end."""
        return self._src[self._pos] if self._pos < len(self._src) else ""

    def _peek(self, offset: int = 1) -> str:
        p = self._pos + offset
        return self._src[p] if p < len(self._src) else ""

    def _advance(self) -> str:
        ch = self._src[self._pos]
        self._pos += 1
        if ch == "\n":
            self._line += 1
            self._col   = 1
        else:
            self._col  += 1
        return ch

    def _starts_with(self, s: str) -> bool:
        return self._src.startswith(s, self._pos)

    def _error(self, msg: str) -> LexError:
        return LexError(msg, self._line, self._col)

    def _emit(self, kind: TK, text: str, value: object, line: int, col: int) -> None:
        self._tokens.append(Token(kind, text, value, line, col))

    # ------------------------------------------------------------------
    # Whitespace / comment skipping  [lex.comment]
    # ------------------------------------------------------------------

    def _skip_whitespace_and_comments(self) -> None:
        while self._pos < len(self._src):
            # whitespace
            if self._ch in " \t\r\n":
                self._advance()
                continue
            # line comment  //
            if self._starts_with("//"):
                while self._pos < len(self._src) and self._ch != "\n":
                    self._advance()
                continue
            # block comment  /* ... */   (shall not be nested §4.2)
            if self._starts_with("/*"):
                start_line, start_col = self._line, self._col
                self._advance(); self._advance()  # consume /*
                while self._pos < len(self._src):
                    if self._starts_with("*/"):
                        self._advance(); self._advance()
                        break
                    self._advance()
                else:
                    raise LexError("unterminated block comment",
                                   start_line, start_col)
                continue
            break

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _next_token(self) -> None:
        line, col = self._line, self._col
        ch = self._ch

        # ---- raw string literal  """  (§4.6 [8]) — check before " -------
        if self._starts_with('"""'):
            self._lex_raw_string(line, col)
            return

        # ---- string literal  " ----------------------------------------
        if ch == '"':
            self._lex_string(line, col)
            return

        # ---- char literal  ' ------------------------------------------
        if ch == "'":
            self._lex_char(line, col)
            return

        # ---- backtick operator name  `sym`  (§22.1) --------------------
        if ch == "`":
            self._lex_backtick(line, col)
            return

        # ---- pragma open  [.  or  regular [  --------------------------
        if ch == "[":
            if self._peek() == ".":
                self._advance(); self._advance()
                self._emit(TK.PRAGMA_OPEN, "[.", None, line, col)
            else:
                self._advance()
                self._emit(TK.LBRACKET, "[", None, line, col)
            return

        # ---- pragma close  .]  or  regular ]  -------------------------
        if ch == "]":
            self._advance()
            self._emit(TK.RBRACKET, "]", None, line, col)
            return

        if ch == "." and self._peek() == "]":
            self._advance(); self._advance()
            self._emit(TK.PRAGMA_CLOSE, ".]", None, line, col)
            return

        # ---- numeric literal -------------------------------------------
        if ch.isdigit() or (ch == "." and self._peek().isdigit()):
            self._lex_number(line, col)
            return

        # ---- identifier / keyword  -------------------------------------
        if ch.isalpha() or ch == "_":
            self._lex_ident_or_keyword(line, col)
            return

        # ---- multi-char operators (longest match) ----------------------
        self._lex_operator(line, col)

    # ------------------------------------------------------------------
    # Raw string  """..."""  (§4.6 [8])
    # ------------------------------------------------------------------

    def _lex_raw_string(self, line: int, col: int) -> None:
        self._advance(); self._advance(); self._advance()  # consume """
        start = self._pos
        while self._pos < len(self._src):
            if self._starts_with('"""'):
                content = self._src[start:self._pos]
                self._advance(); self._advance(); self._advance()  # consume """
                self._emit(TK.RAW_STRING_LIT, '"""' + content + '"""',
                           content, line, col)
                return
            self._advance()
        raise self._error('unterminated raw string literal (missing """)')

    # ------------------------------------------------------------------
    # Regular string  "..."  (§4.6 [2,6,7])
    # ------------------------------------------------------------------

    _ESCAPES = {"\\": "\\", '"': '"', "'": "'", "0": "\0"}

    def _lex_string(self, line: int, col: int) -> None:
        self._advance()  # consume opening "
        buf: list[str] = []
        raw_start = self._pos - 1
        while self._pos < len(self._src):
            ch = self._ch
            if ch == '"':
                self._advance()
                raw = self._src[raw_start: self._pos]
                self._emit(TK.STRING_LIT, raw, "".join(buf), line, col)
                return
            if ch == "\\":
                self._advance()
                esc = self._ch
                if esc not in self._ESCAPES:
                    raise self._error(f"unknown escape sequence \\{esc}")
                buf.append(self._ESCAPES[esc])
                self._advance()
            elif ch == "\n":
                raise self._error("unterminated string literal")
            else:
                buf.append(ch)
                self._advance()
        raise self._error("unterminated string literal")

    # ------------------------------------------------------------------
    # Char literal  '.'  (§4.6 [3,6])
    # ------------------------------------------------------------------

    def _lex_char(self, line: int, col: int) -> None:
        self._advance()  # consume '
        ch = self._ch
        if ch == "\\":
            self._advance()
            esc = self._ch
            if esc not in self._ESCAPES:
                raise self._error(f"unknown escape sequence \\{esc}")
            val_ch = self._ESCAPES[esc]
            self._advance()
        elif ch == "'" or ch == "\n":
            raise self._error("empty or unterminated char literal")
        else:
            val_ch = ch
            self._advance()
        if self._ch != "'":
            raise self._error("char literal must contain exactly one character")
        self._advance()  # consume closing '
        self._emit(TK.CHAR_LIT, repr(val_ch), ord(val_ch), line, col)

    # ------------------------------------------------------------------
    # Backtick operator name  `sym`  (§22.1)
    # ------------------------------------------------------------------

    def _lex_backtick(self, line: int, col: int) -> None:
        self._advance()  # consume `
        start = self._pos
        while self._pos < len(self._src) and self._ch != "`":
            if self._ch == "\n":
                raise self._error("unterminated backtick name")
            self._advance()
        if self._pos >= len(self._src):
            raise self._error("unterminated backtick name")
        name = self._src[start:self._pos]
        self._advance()  # consume closing `
        self._emit(TK.BACKTICK_NAME, f"`{name}`", name, line, col)

    # ------------------------------------------------------------------
    # Numeric literal  (§4.6 [4,5])
    # ------------------------------------------------------------------

    def _lex_number(self, line: int, col: int) -> None:
        start = self._pos
        is_float = False
        base = 10

        if self._ch == "0" and self._peek() in ("x", "X"):
            # hexadecimal
            self._advance(); self._advance()
            base = 16
            if not (self._ch.isdigit() or self._ch in "abcdefABCDEF"):
                raise self._error("expected hex digits after 0x")
            while self._ch.isdigit() or self._ch in "abcdefABCDEF_":
                self._advance()
        elif self._ch == "0" and self._peek() in ("b", "B"):
            # binary
            self._advance(); self._advance()
            base = 2
            if self._ch not in "01":
                raise self._error("expected binary digits after 0b")
            while self._ch in "01_":
                self._advance()
        else:
            # decimal integer or float
            while self._ch.isdigit() or self._ch == "_":
                self._advance()
            if self._ch == "." and self._peek().isdigit():
                is_float = True
                self._advance()  # consume .
                while self._ch.isdigit() or self._ch == "_":
                    self._advance()
            if self._ch in ("e", "E"):
                is_float = True
                self._advance()
                if self._ch in ("+", "-"):
                    self._advance()
                if not self._ch.isdigit():
                    raise self._error("expected digits in float exponent")
                while self._ch.isdigit() or self._ch == "_":
                    self._advance()

        raw = self._src[start:self._pos]
        clean = raw.replace("_", "")  # _ as digit separator (common extension)

        if is_float:
            self._emit(TK.FLOAT_LIT, raw, float(clean), line, col)
        else:
            self._emit(TK.INT_LIT, raw, int(clean, base), line, col)

    # ------------------------------------------------------------------
    # Identifier / keyword  (§4.4, §4.5)
    # ------------------------------------------------------------------

    def _lex_ident_or_keyword(self, line: int, col: int) -> None:
        start = self._pos
        # identifiers may start with _ or letter; body: letter, digit, _
        while self._ch.isalnum() or self._ch == "_":
            self._advance()
        word = self._src[start:self._pos]

        # special: _ alone is the wildcard punctuator (§4.7, §8.1)
        # but _foo, _bar etc. are regular identifiers
        if word == "_":
            self._emit(TK.UNDERSCORE, "_", None, line, col)
            return

        # true / false → BOOL_LIT  (§5.4)
        if word == "true":
            self._emit(TK.BOOL_LIT, word, True, line, col)
            return
        if word == "false":
            self._emit(TK.BOOL_LIT, word, False, line, col)
            return

        # keyword lookup
        kw = _KEYWORDS.get(word)
        if kw is not None:
            self._emit(kw, word, None, line, col)
        else:
            self._emit(TK.IDENT, word, word, line, col)

    # ------------------------------------------------------------------
    # Operators and punctuators  (§4.7, §28)
    # ------------------------------------------------------------------

    # Ordered longest-first so multi-char ops beat single-char prefixes.
    _OPS: list[tuple[str, TK]] = [
        # 3-char — none in GAL
        # 2-char
        ("::",  TK.COLONCOLON),
        ("->",  TK.ARROW),
        ("+=",  TK.PLUSEQ),
        ("-=",  TK.MINUSEQ),
        ("/=",  TK.SLASHEQ),
        ("%=",  TK.PERCENTEQ),
        ("*=",  TK.STAREQ),
        ("==",  TK.EQEQ),
        ("!=",  TK.BANGEQ),
        ("<=",  TK.LTEQ),
        (">=",  TK.GTEQ),
        ("&&",  TK.AMPAMP),
        ("||",  TK.PIPEPIPE),
        ("..",  TK.DOTDOT),
        # pragma close — handled before ] but keep here as fallback
        (".]",  TK.PRAGMA_CLOSE),
        # 1-char
        ("{",   TK.LBRACE),
        ("}",   TK.RBRACE),
        ("(",   TK.LPAREN),
        (")",   TK.RPAREN),
        (".",   TK.DOT),
        ("*",   TK.STAR),
        ("+",   TK.PLUS),
        ("-",   TK.MINUS),
        ("/",   TK.SLASH),
        ("%",   TK.PERCENT),
        ("=",   TK.EQ),
        ("<",   TK.LT),
        (">",   TK.GT),
        ("!",   TK.BANG),
        ("|",   TK.PIPE),
        (",",   TK.COMMA),
        (":",   TK.COLON),
        (";",   TK.SEMI),
    ]

    def _lex_operator(self, line: int, col: int) -> None:
        for text, kind in self._OPS:
            if self._starts_with(text):
                for _ in text:
                    self._advance()
                self._emit(kind, text, None, line, col)
                return
        raise self._error(f"unexpected character {self._ch!r}")


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def lex(source: str, filename: str = "<input>") -> List[Token]:
    return Lexer(source, filename).tokenize()
