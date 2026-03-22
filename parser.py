"""
GAL Parser  — recursive-descent, covers the full language specification.
Consumes a token list from the lexer and produces a TranslationUnit AST.
"""

from __future__ import annotations
from typing import List, Optional, Union
from lexer import Token, TK
from ast_nodes import *


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ParseError(Exception):
    def __init__(self, msg: str, line: int, col: int) -> None:
        super().__init__(f"Parse error at {line}:{col}: {msg}")
        self.line = line
        self.col  = col


# ---------------------------------------------------------------------------
# Token-set constants
# ---------------------------------------------------------------------------

_FUNDAMENTAL_KINDS = frozenset({
    TK.KW_INTB, TK.KW_INTW, TK.KW_INTD, TK.KW_INTQ, TK.KW_INTO,
    TK.KW_UINTB, TK.KW_UINTW, TK.KW_UINTD, TK.KW_UINTQ, TK.KW_UINTO,
    TK.KW_USIZE, TK.KW_SSIZE,
    TK.KW_FLOATW, TK.KW_FLOATD, TK.KW_FLOATQ, TK.KW_FLOATO,
    TK.KW_BOOL, TK.KW_CHAR,
})

_TYPE_START = frozenset({
    *_FUNDAMENTAL_KINDS,
    TK.KW_PTR, TK.KW_REF, TK.KW_VAL,
    TK.KW_OBJECT, TK.KW_ENUM, TK.KW_AUTO, TK.KW_VOID,
    TK.UNDERSCORE, TK.LPAREN, TK.IDENT,
})

# Tokens that can begin a no-paren-call argument after a callable expression
_ARG_STARTERS = frozenset({
    TK.INT_LIT, TK.FLOAT_LIT, TK.STRING_LIT, TK.CHAR_LIT,
    TK.BOOL_LIT, TK.RAW_STRING_LIT, TK.IDENT, TK.LBRACKET, TK.KW_FN,
})

_DECL_KW = frozenset({TK.KW_LET, TK.KW_VAR, TK.KW_CONST})

_ASSIGN_OPS = frozenset({
    TK.EQ, TK.PLUSEQ, TK.MINUSEQ, TK.STAREQ, TK.SLASHEQ, TK.PERCENTEQ,
})


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class Parser:

    def __init__(self, tokens: List[Token]) -> None:
        self._toks = tokens
        self._pos  = 0

    # ------------------------------------------------------------------ utils

    def _peek(self, offset: int = 0) -> Token:
        i = self._pos + offset
        return self._toks[min(i, len(self._toks) - 1)]

    def _advance(self) -> Token:
        t = self._toks[self._pos]
        if t.kind != TK.EOF:
            self._pos += 1
        return t

    def _at(self, *kinds: TK) -> bool:
        return self._peek().kind in kinds

    def _match(self, *kinds: TK) -> bool:
        if self._at(*kinds):
            self._advance()
            return True
        return False

    def _expect(self, kind: TK, msg: str = "") -> Token:
        t = self._peek()
        if t.kind != kind:
            raise ParseError(msg or f"expected {kind.name}, got {t.kind.name} ({t.text!r})",
                             t.line, t.col)
        return self._advance()

    def _lc(self) -> tuple[int, int]:
        t = self._peek()
        return t.line, t.col

    def _err(self, msg: str) -> ParseError:
        t = self._peek()
        return ParseError(msg, t.line, t.col)

    # --------------------------------------------------- lookahead helpers

    def _has_eq_before_comma(self) -> bool:
        """Scan forward for = (not ==) before the next top-level , or {."""
        depth = 0
        i = self._pos
        while i < len(self._toks):
            k = self._toks[i].kind
            if k in (TK.LPAREN, TK.LBRACKET, TK.LBRACE):
                depth += 1
            elif k in (TK.RPAREN, TK.RBRACKET, TK.RBRACE):
                depth -= 1
            elif depth == 0:
                if k == TK.EQ:
                    return True
                if k in (TK.COMMA, TK.LBRACE, TK.SEMI, TK.EOF):
                    return False
            i += 1
        return False

    def _is_three_part_while(self) -> bool:
        """True if the while body starts with a for-init (ident [: type] = expr ,)."""
        # Keyword-prefixed init → definitely three-part
        if self._at(*_DECL_KW):
            return True
        # ident optionally followed by : (type annotation) then = → three-part
        if self._at(TK.IDENT):
            pk1 = self._peek(1).kind
            if pk1 == TK.EQ:      # while x = 1, …
                return True
            if pk1 == TK.COLON:   # while x: T = 1, …
                return True
        return False

    def _try_parse_generic_args(self) -> Optional[List[TypeNode]]:
        """Tentatively parse <Type, ...>. Backtracks on failure."""
        saved = self._pos
        try:
            self._expect(TK.LT)
            args: List[TypeNode] = []
            if not self._at(TK.GT):
                args.append(self._parse_type())
                while self._match(TK.COMMA):
                    args.append(self._parse_type())
            self._expect(TK.GT)
            return args
        except ParseError:
            self._pos = saved
            return None

    # ------------------------------------------------------------------ entry

    def parse(self) -> TranslationUnit:
        line, col = self._lc()
        stmts: List[StmtNode] = []
        while not self._at(TK.EOF):
            stmts.append(self._parse_stmt())
        return TranslationUnit(line, col, stmts)

    # ====================================================================
    # Statements
    # ====================================================================

    def _parse_stmt(self) -> StmtNode:
        t = self._peek()
        line, col = t.line, t.col

        # ---- eval-prefix: const or template before fn/if/for/while ----------
        if t.kind == TK.KW_CONST:
            nk = self._peek(1).kind
            if nk == TK.KW_FN:
                self._advance(); return self._parse_fn('const')
            if nk == TK.KW_IF:
                self._advance(); return self._parse_if('const')
            if nk == TK.KW_FOR:
                self._advance(); return self._parse_for('const')
            if nk == TK.KW_WHILE:
                self._advance(); return self._parse_while('const')
            return self._parse_var_or_tuple(self._advance().text)

        if t.kind == TK.KW_TEMPLATE:
            nk = self._peek(1).kind
            if nk == TK.KW_FN:
                self._advance(); return self._parse_fn('template')
            if nk == TK.KW_IF:
                self._advance(); return self._parse_if('template')
            if nk == TK.KW_FOR:
                self._advance(); return self._parse_for('template')
            if nk == TK.KW_WHILE:
                self._advance(); return self._parse_while('template')
            raise self._err("'template' must be followed by fn, if, for, or while")

        if t.kind in (TK.KW_LET, TK.KW_VAR):
            return self._parse_var_or_tuple(self._advance().text)

        if t.kind == TK.KW_TYPE:
            return self._parse_type_decl()

        if t.kind == TK.KW_FN:
            return self._parse_fn(None)

        if t.kind == TK.KW_ALIAS:
            return self._parse_alias()

        if t.kind == TK.KW_IMPORT:
            return self._parse_import()

        if t.kind == TK.KW_INCLUDE:
            return self._parse_include()

        if t.kind == TK.KW_IF:
            # bare `if const { }` — tests whether we're in compile-time context
            if self._peek(1).kind == TK.KW_CONST:
                self._advance(); self._advance()
                body = self._parse_compound()
                return IfStmt(line, col, None, True, None, body, [], None)
            return self._parse_if(None)

        if t.kind == TK.KW_WHILE:
            return self._parse_while(None)

        if t.kind == TK.KW_FOR:
            return self._parse_for(None)

        if t.kind == TK.KW_CASE:
            return self._parse_case()

        if t.kind == TK.KW_LABEL:
            return self._parse_label()

        if t.kind == TK.KW_GOTO:
            return self._parse_goto()

        if t.kind == TK.KW_MIXIN:
            return self._parse_mixin()

        if t.kind == TK.KW_ASM:
            return self._parse_asm()

        if t.kind == TK.KW_RETURN:
            return self._parse_return()

        if t.kind == TK.KW_BREAK:
            self._advance(); self._expect(TK.SEMI)
            return BreakStmt(line, col)

        if t.kind == TK.KW_CONTINUE:
            self._advance(); self._expect(TK.SEMI)
            return ContinueStmt(line, col)

        if t.kind == TK.LBRACE:
            return self._parse_compound()

        # ---- expression statement (handles no-paren calls) ------------------
        expr = self._parse_expr()
        self._expect(TK.SEMI)
        return ExprStmt(line, col, expr)

    # ====================================================================
    # Declarations
    # ====================================================================

    # ---- variable / tuple destructuring ------------------------------------

    def _parse_var_or_tuple(self, keyword: str) -> VarDecl | TupleDestructure:
        line, col = self._lc()

        # Tuple destructuring: let (a, _, c) ...
        if self._at(TK.LPAREN):
            self._advance()
            bindings: List[Optional[str]] = []
            while not self._at(TK.RPAREN, TK.EOF):
                if self._match(TK.UNDERSCORE):
                    bindings.append(None)
                else:
                    bindings.append(self._expect(TK.IDENT).text)
                if not self._match(TK.COMMA):
                    break
            self._expect(TK.RPAREN)
            type_ann = self._parse_type_ann() if self._at(TK.COLON) else None
            self._expect(TK.EQ)
            value = self._parse_expr()
            self._expect(TK.SEMI)
            return TupleDestructure(line, col, keyword, bindings, type_ann, value)

        if self._at(TK.UNDERSCORE):
            name = '_'; self._advance()
        else:
            name = self._expect(TK.IDENT).text
        exported  = self._match(TK.STAR)
        generics  = self._parse_generic_param_list() if self._at(TK.LT) else []
        pragmas   = self._parse_pragmas()
        type_ann  = self._parse_type_ann() if self._at(TK.COLON) else None
        init      = None
        if self._match(TK.EQ):
            init = self._parse_expr()
        self._expect(TK.SEMI)
        return VarDecl(line, col, keyword, name, exported, generics, pragmas,
                       type_ann, init)

    # ---- type declaration --------------------------------------------------

    def _parse_type_decl(self) -> TypeDecl:
        line, col = self._lc()
        self._expect(TK.KW_TYPE)
        name      = self._expect(TK.IDENT).text
        exported  = self._match(TK.STAR)
        generics  = self._parse_generic_param_list() if self._at(TK.LT) else []
        pragmas   = self._parse_pragmas()
        self._expect(TK.EQ)
        type_node = self._parse_type()
        self._expect(TK.SEMI)
        return TypeDecl(line, col, name, exported, generics, pragmas, type_node)

    # ---- function declaration ----------------------------------------------

    def _parse_fn(self, eval_prefix: Optional[str]) -> FunctionDecl | OperatorOverload:
        line, col = self._lc()
        self._expect(TK.KW_FN)

        # Operator overload: fn `op` ...
        if self._at(TK.BACKTICK_NAME):
            op       = self._advance().value
            generics = self._parse_generic_param_list() if self._at(TK.LT) else []
            self._expect(TK.LPAREN)
            params   = self._parse_param_list()
            self._expect(TK.RPAREN)
            pragmas  = self._parse_pragmas()
            ret      = self._parse_type_ann() if self._at(TK.COLON) else None
            body     = self._parse_compound()
            return OperatorOverload(line, col, op, generics, params, ret, body)

        name      = self._expect(TK.IDENT).text
        exported  = self._match(TK.STAR)
        generics  = self._parse_generic_param_list() if self._at(TK.LT) else []
        self._expect(TK.LPAREN)
        params    = self._parse_param_list()
        self._expect(TK.RPAREN)
        pragmas   = self._parse_pragmas()
        ret       = self._parse_type_ann() if self._at(TK.COLON) else None
        # Bodyless declaration (extern / import): fn foo(...) : T ;
        if self._match(TK.SEMI):
            body = CompoundStmt(line, col, [])
            return FunctionDecl(line, col, eval_prefix, name, exported, generics,
                                params, pragmas, ret, body)
        body      = self._parse_compound()
        return FunctionDecl(line, col, eval_prefix, name, exported, generics,
                            params, pragmas, ret, body)

    # ---- alias declaration -------------------------------------------------

    def _parse_alias(self) -> AliasDecl:
        line, col = self._lc()
        self._expect(TK.KW_ALIAS)
        if self._at(TK.UNDERSCORE):
            name = '_'; self._advance()
        else:
            name = self._expect(TK.IDENT).text
        exported = self._match(TK.STAR)
        self._expect(TK.EQ)
        if self._at(TK.LBRACE):
            target = self._parse_compound()
        else:
            target = self._parse_expr()
        self._expect(TK.SEMI)
        return AliasDecl(line, col, name, exported, target)

    # ---- import / include --------------------------------------------------

    def _parse_import(self) -> ImportDirective:
        line, col = self._lc()
        self._expect(TK.KW_IMPORT)
        path     = self._expect(TK.STRING_LIT).value
        reexport = self._match(TK.STAR)
        self._expect(TK.SEMI)
        return ImportDirective(line, col, path, reexport)

    def _parse_include(self) -> IncludeDirective:
        line, col = self._lc()
        self._expect(TK.KW_INCLUDE)
        path = self._expect(TK.STRING_LIT).value
        self._expect(TK.SEMI)
        return IncludeDirective(line, col, path)

    # ====================================================================
    # Control-flow statements
    # ====================================================================

    def _parse_compound(self) -> CompoundStmt:
        line, col = self._lc()
        self._expect(TK.LBRACE)
        body: List[StmtNode] = []
        while not self._at(TK.RBRACE, TK.EOF):
            body.append(self._parse_stmt())
        self._expect(TK.RBRACE)
        return CompoundStmt(line, col, body)

    # ---- if ----------------------------------------------------------------

    def _parse_if(self, prefix: Optional[str]) -> IfStmt:
        line, col = self._lc()
        self._expect(TK.KW_IF)
        self._expect(TK.LPAREN)
        cond  = self._parse_expr()
        self._expect(TK.RPAREN)
        then_ = self._parse_compound()
        elifs: List[ElifClause] = []
        else_: Optional[CompoundStmt] = None
        while self._at(TK.KW_ELIF):
            el, ec = self._lc()
            self._advance()
            self._expect(TK.LPAREN)
            ec2 = self._parse_expr()
            self._expect(TK.RPAREN)
            eb  = self._parse_compound()
            elifs.append(ElifClause(el, ec, ec2, eb))
        if self._match(TK.KW_ELSE):
            else_ = self._parse_compound()
        return IfStmt(line, col, prefix, False, cond, then_, elifs, else_)

    # ---- while -------------------------------------------------------------

    def _parse_while(self, prefix: Optional[str]) -> WhileStmt:
        line, col = self._lc()
        self._expect(TK.KW_WHILE)
        init:   Optional[ForInit] = None
        update: Optional[ExprNode] = None

        if self._is_three_part_while():
            # Parse for-init: keyword? ident type_ann? = expr
            iline, icol = self._lc()
            ikw = self._advance().text if self._at(*_DECL_KW) else None
            iname = self._expect(TK.IDENT).text
            itype = self._parse_type_ann() if self._at(TK.COLON) else None
            self._expect(TK.EQ)
            ival  = self._parse_expr()
            init  = ForInit(iline, icol, ikw, iname, itype, ival)
            self._expect(TK.COMMA)
            cond  = self._parse_expr()
            self._expect(TK.COMMA)
            update = self._parse_expr()
        else:
            cond = self._parse_expr()
            if self._match(TK.COMMA):
                # Three-part without declaration keyword: init was the expression
                iline, icol = cond.line, cond.col
                init = ForInit(iline, icol, None, None, None, cond)
                cond = self._parse_expr()
                self._expect(TK.COMMA)
                update = self._parse_expr()

        body = self._parse_compound()
        return WhileStmt(line, col, prefix, init, cond, update, body)

    # ---- for ---------------------------------------------------------------

    def _parse_for(self, prefix: Optional[str]) -> ForStmt:
        line, col = self._lc()
        self._expect(TK.KW_FOR)

        index: Optional[ForIndex] = None

        # Check for for-index: _ , or keyword? ident type_ann? = expr ,
        is_underscore_index = (self._at(TK.UNDERSCORE) and
                               self._peek(1).kind == TK.COMMA)
        has_eq = (not self._at(TK.UNDERSCORE) and self._has_eq_before_comma())

        if is_underscore_index:
            iline, icol = self._lc()
            self._advance()   # _
            self._advance()   # ,
            index = ForIndex(iline, icol, None, None, None, WildcardExpr(iline, icol))

        elif has_eq or self._at(*_DECL_KW):
            iline, icol = self._lc()
            ikw   = self._advance().text if self._at(*_DECL_KW) else None
            iname = self._expect(TK.IDENT).text
            itype = self._parse_type_ann() if self._at(TK.COLON) else None
            self._expect(TK.EQ)
            ival  = self._parse_expr()
            self._expect(TK.COMMA)
            index = ForIndex(iline, icol, ikw, iname, itype, ival)

        # Element binding
        ekw: Optional[str] = None
        ename: Optional[str] = None
        etype: Optional[TypeNode] = None

        if self._at(TK.UNDERSCORE):
            self._advance()
        else:
            if self._at(*_DECL_KW):
                ekw = self._advance().text
            ename = self._expect(TK.IDENT).text
            if self._at(TK.COLON):
                etype = self._parse_type_ann()

        self._expect(TK.COMMA)
        iterable = self._parse_expr()
        body     = self._parse_compound()
        return ForStmt(line, col, prefix, index, ekw, ename, etype, iterable, body)

    # ---- case --------------------------------------------------------------

    def _parse_case(self) -> CaseStmt:
        line, col = self._lc()
        self._expect(TK.KW_CASE)
        self._expect(TK.LPAREN)
        disc = self._parse_expr()
        self._expect(TK.RPAREN)
        self._expect(TK.LBRACE)
        arms: List[CaseArm] = []
        while not self._at(TK.RBRACE, TK.EOF):
            al, ac = self._lc()
            self._expect(TK.KW_LABEL)
            if self._at(TK.UNDERSCORE):
                self._advance()
                pattern = None
            else:
                lo = self._parse_expr()
                if self._match(TK.DOTDOT):
                    hi = self._parse_expr()
                    pattern = RangeExpr(al, ac, lo, hi)
                else:
                    pattern = lo
            self._expect(TK.COLON)
            body: List[StmtNode] = []
            while not self._at(TK.KW_LABEL, TK.RBRACE, TK.EOF):
                body.append(self._parse_stmt())
            arms.append(CaseArm(al, ac, pattern, body))
        self._expect(TK.RBRACE)
        return CaseStmt(line, col, disc, arms)

    # ---- label / goto ------------------------------------------------------

    def _parse_label(self) -> LabelDecl:
        line, col = self._lc()
        self._expect(TK.KW_LABEL)
        name = self._expect(TK.IDENT).text
        self._expect(TK.SEMI)
        return LabelDecl(line, col, name)

    def _parse_goto(self) -> GotoStmt:
        line, col = self._lc()
        self._expect(TK.KW_GOTO)
        t = self._peek()
        if t.kind == TK.INT_LIT:
            self._advance()
            self._expect(TK.SEMI)
            return GotoStmt(line, col, t.value)
        name = self._expect(TK.IDENT).text
        self._expect(TK.SEMI)
        return GotoStmt(line, col, name)

    # ---- mixin -------------------------------------------------------------

    def _parse_mixin(self) -> MixinStmt:
        line, col = self._lc()
        self._expect(TK.KW_MIXIN)
        self._expect(TK.LPAREN)
        expr = self._parse_expr()
        self._expect(TK.RPAREN)
        self._expect(TK.SEMI)
        return MixinStmt(line, col, expr)

    # ---- asm ---------------------------------------------------------------

    def _parse_asm(self) -> AsmStmt:
        line, col = self._lc()
        self._expect(TK.KW_ASM)
        self._expect(TK.LPAREN)
        tmpl     = self._parse_expr()
        self._expect(TK.COMMA)
        outputs  = self._parse_asm_operand_list()
        self._expect(TK.COMMA)
        inputs   = self._parse_asm_operand_list()
        self._expect(TK.COMMA)
        clobbers = self._parse_asm_operand_list()
        self._expect(TK.RPAREN)
        self._expect(TK.SEMI)
        return AsmStmt(line, col, tmpl, outputs, inputs, clobbers)

    def _parse_asm_operand_list(self) -> List[Optional[ExprNode]]:
        if self._at(TK.UNDERSCORE):
            self._advance()
            return []
        ops = [self._parse_expr()]
        # Only consume more commas if the next token isn't ) 
        # (we can't peek two commas deep cleanly; single-operand per list for POC)
        return ops

    # ---- return ------------------------------------------------------------

    def _parse_return(self) -> ReturnStmt:
        line, col = self._lc()
        self._expect(TK.KW_RETURN)
        value: Optional[ExprNode] = None
        if not self._at(TK.SEMI):
            value = self._parse_expr()
        self._expect(TK.SEMI)
        return ReturnStmt(line, col, value)

    # ====================================================================
    # Types
    # ====================================================================

    def _parse_type_ann(self) -> TypeNode:
        """Consume `: qualifier? type` and return the type node."""
        self._expect(TK.COLON)
        # Optional top-level qualifier (mut / imut) — noted but not wrapped
        self._match(TK.KW_MUT, TK.KW_IMUT)
        return self._parse_type()

    def _parse_type(self) -> TypeNode:
        """Parse a full type, including union `T | U | ...`."""
        line, col = self._lc()
        left = self._parse_type_atom()
        if self._at(TK.PIPE):
            alts = [left]
            while self._match(TK.PIPE):
                alts.append(self._parse_type_atom())
            return UnionType(line, col, alts)
        return left

    def _parse_type_atom(self) -> TypeNode:
        """Parse a single type (no union)."""
        line, col = self._lc()
        t = self._peek()

        if t.kind == TK.KW_PTR:
            self._advance()
            qual = self._advance().text if self._at(TK.KW_MUT, TK.KW_IMUT) else None
            return PointerType(line, col, qual, self._parse_type_atom())

        if t.kind == TK.KW_REF:
            self._advance()
            qual = self._advance().text if self._at(TK.KW_MUT, TK.KW_IMUT) else None
            return ReferenceType(line, col, qual, self._parse_type_atom())

        if t.kind == TK.KW_VAL:
            self._advance()
            qual = self._advance().text if self._at(TK.KW_MUT, TK.KW_IMUT) else None
            return ValueType(line, col, qual, self._parse_type_atom())

        if t.kind == TK.KW_OBJECT:
            return self._parse_object_type()

        if t.kind == TK.KW_ENUM:
            return self._parse_enum_type()

        if t.kind == TK.LPAREN:
            self._advance()
            types = [self._parse_type()]
            while self._match(TK.COMMA):
                types.append(self._parse_type())
            self._expect(TK.RPAREN)
            return TupleType(line, col, types)

        if t.kind == TK.KW_AUTO:
            self._advance()
            return self._parse_array_suffix(AutoType(line, col))

        if t.kind == TK.KW_VOID:
            self._advance()
            return self._parse_array_suffix(VoidType(line, col))

        if t.kind == TK.UNDERSCORE:
            self._advance()
            return WildcardType(line, col)

        if t.kind in _FUNDAMENTAL_KINDS:
            self._advance()
            return self._parse_array_suffix(FundamentalType(line, col, t.text))

        if t.kind == TK.IDENT:
            self._advance()
            return self._parse_array_suffix(IdentType(line, col, t.text))

        raise self._err(f"expected type, got {t.kind.name} ({t.text!r})")

    def _parse_array_suffix(self, base: TypeNode) -> TypeNode:
        """Consume zero or more `[size]` suffixes."""
        while self._at(TK.LBRACKET):
            line, col = self._lc()
            self._advance()
            if self._at(TK.UNDERSCORE):
                self._advance(); size = None
            else:
                size = self._parse_expr()
            self._expect(TK.RBRACKET)
            base = ArrayType(line, col, base, size)
        return base

    # ---- object type -------------------------------------------------------

    def _parse_object_type(self) -> ObjectType:
        line, col = self._lc()
        self._expect(TK.KW_OBJECT)
        parents: List[str] = []
        while self._match(TK.COMMA):
            parents.append(self._expect(TK.IDENT).text)
        self._expect(TK.LBRACE)
        fields: List[FieldDecl] = []
        while not self._at(TK.RBRACE, TK.EOF):
            fields.append(self._parse_field_decl())
        self._expect(TK.RBRACE)
        return ObjectType(line, col, parents, fields)

    def _parse_field_decl(self) -> FieldDecl:
        line, col = self._lc()
        kw  = self._advance().text if self._at(*_DECL_KW) else None
        name = self._expect(TK.IDENT).text
        exported = self._match(TK.STAR)
        pragmas  = self._parse_pragmas()
        type_ann = self._parse_type_ann() if self._at(TK.COLON) else None
        init     = None
        if self._match(TK.EQ):
            init = self._parse_expr()
        self._expect(TK.SEMI)
        return FieldDecl(line, col, kw, name, exported, pragmas, type_ann, init)

    # ---- enum type ---------------------------------------------------------

    def _parse_enum_type(self) -> EnumType:
        line, col = self._lc()
        self._expect(TK.KW_ENUM)
        self._expect(TK.LBRACE)
        variants: List[EnumVariant] = []
        while not self._at(TK.RBRACE, TK.EOF):
            vl, vc = self._lc()
            name = self._expect(TK.IDENT).text
            self._expect(TK.EQ)
            value = self._parse_expr()
            self._expect(TK.SEMI)
            variants.append(EnumVariant(vl, vc, name, value))
        self._expect(TK.RBRACE)
        return EnumType(line, col, variants)

    # ====================================================================
    # Expressions  (precedence: 11=lowest … 1=highest)
    # ====================================================================

    def _parse_expr(self) -> ExprNode:
        return self._parse_assign()

    # level 11 — assignment  (right-associative)
    def _parse_assign(self) -> ExprNode:
        line, col = self._lc()
        left = self._parse_or()
        if self._at(*_ASSIGN_OPS):
            op    = self._advance().text
            right = self._parse_assign()          # right-assoc
            return AssignExpr(line, col, op, left, right)
        return left

    # level 9 — ||
    def _parse_or(self) -> ExprNode:
        line, col = self._lc()
        left = self._parse_and()
        while self._at(TK.PIPEPIPE):
            op    = self._advance().text
            right = self._parse_and()
            left  = BinaryExpr(line, col, op, left, right)
        return left

    # level 8 — &&
    def _parse_and(self) -> ExprNode:
        line, col = self._lc()
        left = self._parse_eq()
        while self._at(TK.AMPAMP):
            op    = self._advance().text
            right = self._parse_eq()
            left  = BinaryExpr(line, col, op, left, right)
        return left

    # level 7 — == !=
    def _parse_eq(self) -> ExprNode:
        line, col = self._lc()
        left = self._parse_cmp()
        while self._at(TK.EQEQ, TK.BANGEQ):
            op    = self._advance().text
            right = self._parse_cmp()
            left  = BinaryExpr(line, col, op, left, right)
        return left

    # level 6 — < > <= >=
    def _parse_cmp(self) -> ExprNode:
        line, col = self._lc()
        left = self._parse_range()
        while self._at(TK.LT, TK.GT, TK.LTEQ, TK.GTEQ):
            op    = self._advance().text
            right = self._parse_range()
            left  = BinaryExpr(line, col, op, left, right)
        return left

    # level 5 — ..
    def _parse_range(self) -> ExprNode:
        line, col = self._lc()
        left = self._parse_add()
        if self._match(TK.DOTDOT):
            right = self._parse_add()
            return RangeExpr(line, col, left, right)
        return left

    # level 4 — + -
    def _parse_add(self) -> ExprNode:
        line, col = self._lc()
        left = self._parse_mul()
        while self._at(TK.PLUS, TK.MINUS):
            op    = self._advance().text
            right = self._parse_mul()
            left  = BinaryExpr(line, col, op, left, right)
        return left

    # level 3 — * / %
    def _parse_mul(self) -> ExprNode:
        line, col = self._lc()
        left = self._parse_unary()
        while self._at(TK.STAR, TK.SLASH, TK.PERCENT):
            op    = self._advance().text
            right = self._parse_unary()
            left  = BinaryExpr(line, col, op, left, right)
        return left

    # level 2 — unary prefix  (right-associative)
    def _parse_unary(self) -> ExprNode:
        line, col = self._lc()

        if self._at(TK.BANG, TK.MINUS, TK.PLUS):
            op = self._advance().text
            return UnaryExpr(line, col, op, self._parse_unary())

        if self._at(TK.KW_ADDR):
            self._advance()
            self._expect(TK.LPAREN)
            operand = self._parse_expr()
            self._expect(TK.RPAREN)
            return AddrExpr(line, col, operand)

        if self._at(TK.KW_CAST):
            self._advance()
            self._expect(TK.LT)
            target = self._parse_type()
            self._expect(TK.GT)
            self._expect(TK.LPAREN)
            operand = self._parse_expr()
            self._expect(TK.RPAREN)
            return CastExpr(line, col, target, operand)

        return self._parse_postfix()

    # level 1 — postfix: () [] . -> :: and no-paren calls
    def _parse_postfix(self) -> ExprNode:
        line, col = self._lc()
        node = self._parse_primary()

        while True:
            # paren call: f(args)
            if self._at(TK.LPAREN):
                self._advance()
                args = self._parse_arg_list()
                self._expect(TK.RPAREN)
                node = CallExpr(line, col, node, args, [])

            # subscript: base[idx]
            elif self._at(TK.LBRACKET):
                self._advance()
                idx = self._parse_expr()
                self._expect(TK.RBRACKET)
                node = SubscriptExpr(line, col, node, idx)

            # member: obj.field  (also entry for UFCS call with generic)
            elif self._at(TK.DOT):
                self._advance()
                member = self._expect(TK.IDENT).text
                node   = MemberExpr(line, col, node, member)
                # Optional explicit generic args: obj.method<T>(args)
                if self._at(TK.LT):
                    generics = self._try_parse_generic_args()
                    if generics is not None:
                        self._expect(TK.LPAREN)
                        args = self._parse_arg_list()
                        self._expect(TK.RPAREN)
                        node = CallExpr(line, col, node, args, generics)

            # arrow: ptr->field
            elif self._at(TK.ARROW):
                self._advance()
                member = self._expect(TK.IDENT).text
                node   = ArrowExpr(line, col, node, member)

            # scope resolution: S::N  or  _::N
            elif self._at(TK.COLONCOLON):
                self._advance()
                name = self._expect(TK.IDENT).text
                node = ScopeExpr(line, col, node, name)

            else:
                break

        # No-paren call: `f arg, arg` or `obj.method arg, arg`
        if (isinstance(node, (IdentExpr, MemberExpr)) and
                self._peek().kind in _ARG_STARTERS):
            args = [self._parse_expr()]
            while self._match(TK.COMMA):
                args.append(self._parse_expr())
            node = CallExpr(line, col, node, args, [])

        return node

    # ---- primary -----------------------------------------------------------

    def _parse_primary(self) -> ExprNode:
        line, col = self._lc()
        t = self._peek()

        # Literals
        if t.kind == TK.INT_LIT:
            self._advance(); return IntLiteral(line, col, t.value, t.text)
        if t.kind == TK.FLOAT_LIT:
            self._advance(); return FloatLiteral(line, col, t.value, t.text)
        if t.kind == TK.BOOL_LIT:
            self._advance(); return BoolLiteral(line, col, t.value)
        if t.kind == TK.CHAR_LIT:
            self._advance(); return CharLiteral(line, col, t.value)
        if t.kind == TK.STRING_LIT:
            self._advance(); return StringLiteral(line, col, t.value)
        if t.kind == TK.RAW_STRING_LIT:
            self._advance(); return RawStringLiteral(line, col, t.value)

        # Wildcard _
        if t.kind == TK.UNDERSCORE:
            self._advance(); return WildcardExpr(line, col)

        # Lambda
        if t.kind == TK.KW_FN:
            return self._parse_lambda()

        # Parenthesised expression or tuple
        if t.kind == TK.LPAREN:
            return self._parse_paren_or_tuple()

        # Array literal or range
        if t.kind == TK.LBRACKET:
            return self._parse_array_lit()

        # Fundamental type used as constructor: intd(x), floatd(x), bool(x) …
        if t.kind in _FUNDAMENTAL_KINDS:
            self._advance()
            type_node = FundamentalType(line, col, t.text)
            self._expect(TK.LPAREN)
            operand = self._parse_expr()
            self._expect(TK.RPAREN)
            return TypeConstructExpr(line, col, type_node, operand)

        # this / This
        if t.kind in (TK.KW_THIS, TK.KW_THIS_TYPE):
            self._advance(); return IdentExpr(line, col, t.text)

        # Identifier (and built-in reflection symbols)
        if t.kind == TK.IDENT:
            name = t.value
            self._advance()

            if name == 'typeof':
                self._expect(TK.LPAREN)
                op = self._parse_expr(); self._expect(TK.RPAREN)
                return TypeofExpr(line, col, op)

            if name in ('sizeof', 'alignof'):
                self._expect(TK.LT)
                tn = self._parse_type(); self._expect(TK.GT)
                self._expect(TK.LPAREN); self._expect(TK.RPAREN)
                return (SizeofExpr if name == 'sizeof' else AlignofExpr)(line, col, tn)

            if name == 'offsetof':
                self._expect(TK.LT)
                tn = self._parse_type(); self._expect(TK.GT)
                self._expect(TK.LPAREN)
                field = self._parse_expr(); self._expect(TK.RPAREN)
                return OffsetofExpr(line, col, tn, field)

            if name == 'stringof':
                self._expect(TK.LPAREN)
                op = self._parse_expr(); self._expect(TK.RPAREN)
                return StringofExpr(line, col, op)

            if name == 'identof':
                self._expect(TK.LPAREN)
                op = self._parse_expr(); self._expect(TK.RPAREN)
                return IdentofExpr(line, col, op)

            if name == 'fieldof':
                self._expect(TK.LT)
                tn = self._parse_type(); self._expect(TK.GT)
                self._expect(TK.LPAREN)
                idx = self._parse_expr(); self._expect(TK.RPAREN)
                return FieldofExpr(line, col, tn, idx)

            if name == 'fieldSizeof':
                self._expect(TK.LT)
                tn = self._parse_type(); self._expect(TK.GT)
                self._expect(TK.LPAREN); self._expect(TK.RPAREN)
                return FieldSizeofExpr(line, col, tn)

            if name == 'returnof':
                self._expect(TK.LPAREN)
                fn = self._parse_expr(); self._expect(TK.RPAREN)
                return ReturnofExpr(line, col, fn)

            return IdentExpr(line, col, name)

        raise self._err(f"unexpected token {t.kind.name} ({t.text!r}) in expression")

    # ---- lambda ------------------------------------------------------------

    def _parse_lambda(self) -> LambdaExpr:
        line, col = self._lc()
        self._expect(TK.KW_FN)
        generics = self._parse_generic_param_list() if self._at(TK.LT) else []
        self._expect(TK.LPAREN)
        params = self._parse_param_list()
        self._expect(TK.RPAREN)
        ret  = self._parse_type_ann() if self._at(TK.COLON) else None
        body = self._parse_compound()
        return LambdaExpr(line, col, generics, params, ret, body)

    # ---- parenthesised / tuple ---------------------------------------------

    def _parse_paren_or_tuple(self) -> ExprNode:
        line, col = self._lc()
        self._expect(TK.LPAREN)
        first = self._parse_expr()
        if self._match(TK.COMMA):
            elems = [first]
            while not self._at(TK.RPAREN, TK.EOF):
                elems.append(self._parse_expr())
                if not self._match(TK.COMMA):
                    break
            self._expect(TK.RPAREN)
            return TupleExpr(line, col, elems)
        self._expect(TK.RPAREN)
        return first   # just parenthesised

    # ---- array literal / range ---------------------------------------------

    def _parse_array_lit(self) -> ArrayLiteralExpr:
        line, col = self._lc()
        self._expect(TK.LBRACKET)
        if self._at(TK.RBRACKET):
            self._advance()
            return ArrayLiteralExpr(line, col, [], None, None)
        first = self._parse_expr()
        if isinstance(first, RangeExpr):
            # [lo .. hi]  — range already parsed inside parse_expr via _parse_range
            self._expect(TK.RBRACKET)
            return ArrayLiteralExpr(line, col, [], first.lo, first.hi)
        elems = [first]
        while self._match(TK.COMMA):
            if self._at(TK.RBRACKET):
                break
            elems.append(self._parse_expr())
        self._expect(TK.RBRACKET)
        return ArrayLiteralExpr(line, col, elems, None, None)

    # ====================================================================
    # Helpers shared by declarations and expressions
    # ====================================================================

    def _parse_generic_param_list(self) -> List[GenericParam]:
        """Parse `< T, U = int, V.. >` generic parameter declarations."""
        self._expect(TK.LT)
        params: List[GenericParam] = []
        while not self._at(TK.GT, TK.EOF):
            line, col = self._lc()
            name = self._expect(TK.IDENT).text
            variadic = False
            default: Optional[TypeNode] = None
            # T.. treated as variadic (spec writes T...)
            if self._at(TK.DOTDOT):
                self._advance()
                # consume trailing . if present (T... = DOTDOT DOT)
                self._match(TK.DOT)
                variadic = True
            elif self._match(TK.EQ):
                default = self._parse_type()
            params.append(GenericParam(line, col, name, default, variadic))
            if not self._match(TK.COMMA):
                break
        self._expect(TK.GT)
        return params

    def _parse_param_list(self) -> List[Param]:
        """Parse comma-separated function parameters."""
        params: List[Param] = []
        while not self._at(TK.RPAREN, TK.EOF):
            line, col = self._lc()
            kw   = self._advance().text if self._at(*_DECL_KW) else None
            name = self._expect(TK.IDENT).text
            type_ann = self._parse_type_ann() if self._at(TK.COLON) else None
            default  = None
            if self._match(TK.EQ):
                default = self._parse_expr()
            params.append(Param(line, col, kw, name, type_ann, default))
            if not self._match(TK.COMMA):
                break
        return params

    def _parse_arg_list(self) -> List[ExprNode]:
        """Parse comma-separated call arguments, supporting name: expr form."""
        args: List[ExprNode] = []
        while not self._at(TK.RPAREN, TK.EOF):
            line, col = self._lc()
            # named arg: ident : expr
            if self._at(TK.IDENT) and self._peek(1).kind == TK.COLON:
                name = self._advance().text
                self._advance()  # consume :
                val  = self._parse_expr()
                args.append(NamedArgExpr(line, col, name, val))
            else:
                args.append(self._parse_expr())
            if not self._match(TK.COMMA):
                break
        return args

    def _parse_pragmas(self) -> List[Pragma]:
        """Parse zero or more `[. pragma-list .]` blocks."""
        pragmas: List[Pragma] = []
        while self._at(TK.PRAGMA_OPEN):
            self._advance()
            pragmas.append(self._parse_pragma_item())
            while self._match(TK.COMMA):
                pragmas.append(self._parse_pragma_item())
            self._expect(TK.PRAGMA_CLOSE)
        return pragmas

    def _parse_pragma_item(self) -> Pragma:
        line, col = self._lc()
        # Pragma names may coincide with keywords (e.g. import, inline, static)
        t = self._peek()
        if not (t.kind == TK.IDENT or t.kind.name.startswith("KW_")):
            raise self._err(f"expected pragma name, got {t.kind.name}")
        name = t.text; self._advance()
        specifier: Optional[str] = None
        argument: Optional[ExprNode] = None
        if self._match(TK.COLON):
            st = self._peek()
            if not (st.kind == TK.IDENT or st.kind.name.startswith("KW_")):
                raise self._err(f"expected pragma specifier, got {st.kind.name}")
            specifier = st.text; self._advance()
            if self._at(TK.LPAREN):
                self._advance()
                argument = self._parse_expr()
                self._expect(TK.RPAREN)
        elif self._at(TK.LPAREN):
            self._advance()
            argument = self._parse_expr()
            self._expect(TK.RPAREN)
        return Pragma(line, col, name, specifier, argument)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def parse(tokens, filename: str = "<input>") -> TranslationUnit:
    return Parser(tokens).parse()
