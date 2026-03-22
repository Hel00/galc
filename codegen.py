"""
GAL → C++23 (freestanding, clang) code generator.

Mapping summary:
  intb/w/d/q/o        → __INT8_TYPE__ / __INT16_TYPE__ / __INT32_TYPE__ /
                         __INT64_TYPE__ / __int128
  uintb/w/d/q/o       → unsigned variants
  usize/ssize         → __SIZE_TYPE__ / __PTRDIFF_TYPE__
  floatw/d/q/o        → _Float16 / float / double / long double
  bool                → bool
  char                → char
  ptr T               → T*
  ref T               → T&
  val T               → T&&
  T[N]                → T[N]   (with .len synthesised as constexpr)
  (T1,T2)             → struct _gal_tuple_N { T1 _0; T2 _1; };
  T1|T2               → union active type (compile-time fixed, emit as T)
  object              → struct
  enum                → struct with constexpr members
  fn                  → function / lambda
  const fn            → consteval function
  template fn         → textual expansion via macro (not yet supported, stub)
  generics            → C++ templates
  operator overload   → operator overload
  case                → switch
  label/goto          → C++ label/goto
  addr(x)             → &x
  cast<T>(x)          → __builtin_bit_cast(T, x)
  T(x)                → static_cast<T>(x)
  asm(...)            → __asm__ (GAS extended)
  pragma inline:always → [[clang::always_inline]]
  pragma inline:never  → [[clang::noinline]]
  pragma noreturn      → [[noreturn]]
  pragma packed        → __attribute__((packed))
  pragma volatile      → volatile
  pragma deprecated    → [[deprecated]]
  pragma align(N)      → alignas(N)
  pragma section(s)    → __attribute__((section(s)))
  pragma export(s)     → __attribute__((visibility("default"))) + asm alias
  pragma import(s)     → extern declaration
  pragma convention(s) → __attribute__((calling_convention))
  pragma static        → static
  pragma cold          → [[clang::cold]]
  pragma hot           → [[clang::hot]]
  pragma bit:set(N)    → : N  (bitfield)
  pragma bit:get(T)    → sizeof(T)*8  constexpr initializer
  mixin                → compile-time string eval (stubbed)
  include              → #include equivalent (re-parse inline)
  import               → forward-declare exported symbols
  typeof               → decltype
  sizeof<T>()          → sizeof(T)
  alignof<T>()         → alignof(T)
  offsetof<T>(f)       → __builtin_offsetof(T,f)
  stringof(x)          → #x via macro
  identof(x)           → #x via macro
  fieldof<T>(i)        → not emittable at runtime; stub
  returnof(f)          → decltype(f())
  UFCS x.f(args)       → f(x, args)  (already resolved by parser into MemberExpr+CallExpr)
"""

from __future__ import annotations
import os
import re
import sys
from typing import List, Optional, Union
from ast_nodes import *
from lexer import lex
from parser import parse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _indent(s: str, n: int = 1) -> str:
    pad = "    " * n
    return "\n".join(pad + l if l else l for l in s.split("\n"))


# ---------------------------------------------------------------------------
# Type name mapping
# ---------------------------------------------------------------------------

_FUNDAMENTAL_MAP: dict[str, str] = {
    "intb":   "__INT8_TYPE__",
    "intw":   "__INT16_TYPE__",
    "intd":   "__INT32_TYPE__",
    "intq":   "__INT64_TYPE__",
    "into":   "__int128",
    "uintb":  "__UINT8_TYPE__",
    "uintw":  "__UINT16_TYPE__",
    "uintd":  "__UINT32_TYPE__",
    "uintq":  "__UINT64_TYPE__",
    "uinto":  "unsigned __int128",
    "usize":  "__SIZE_TYPE__",
    "ssize":  "__PTRDIFF_TYPE__",
    "floatw": "_Float16",
    "floatd": "float",
    "floatq": "double",
    "floato": "long double",
    "bool":   "bool",
    "char":   "char",
    "void":   "void",
}

_OPERATOR_MAP: dict[str, str] = {
    "+":  "+",  "-":  "-",  "*":  "*",  "/":  "/",  "%":  "%",
    "==": "==", "!=": "!=", "<":  "<",  ">":  ">",  "<=": "<=", ">=": ">=",
    "&&": "&&", "||": "||", "!":  "!",
    "=":  "=",  "+=": "+=", "-=": "-=", "*=": "*=", "/=": "/=", "%=": "%=",
    "|":  "|",
}

# calling-convention names GAL → clang attribute strings
_CONV_MAP: dict[str, str] = {
    "cdecl":    "cdecl",
    "stdcall":  "stdcall",
    "fastcall": "fastcall",
    "ms_abi":   "ms_abi",
    "sysv_abi": "sysv_abi",
}


# ---------------------------------------------------------------------------
# Tuple registry  (we synthesise a struct for each unique shape)
# ---------------------------------------------------------------------------

class TupleRegistry:
    def __init__(self) -> None:
        self._map:   dict[tuple, str] = {}   # type-sig tuple → struct name
        self._decls: list[str]        = []

    def get(self, type_sigs: list[str], gen: "Codegen") -> str:
        key = tuple(type_sigs)
        if key in self._map:
            return self._map[key]
        name = f"_gal_tuple_{len(self._map)}"
        fields = "\n".join(f"    {t} _{i};" for i, t in enumerate(type_sigs))
        decl = f"struct {name} {{\n{fields}\n}};"
        self._decls.append(decl)
        self._map[key] = name
        return name

    def preamble(self) -> str:
        return "\n".join(self._decls)


# ---------------------------------------------------------------------------
# Main code generator
# ---------------------------------------------------------------------------

class Codegen:

    def __init__(self, filename: str = "<input>",
                 include_dirs: Optional[List[str]] = None) -> None:
        self._filename    = filename
        self._include_dirs = include_dirs or []
        self._tuples      = TupleRegistry()
        self._lambda_ctr  = 0
        self._out_parts:  list[str] = []
        # track anonymous lambdas that need to be hoisted to file scope
        self._hoisted:    list[str] = []
        # registries for transpile-time features
        self._template_fns:  dict[str, FunctionDecl] = {}   # name → node
        self._type_decls:    dict[str, TypeDecl]     = {}   # name → node
        self._const_strings: dict[str, str]          = {}   # name → string value

    # ------------------------------------------------------------------ entry

    def generate(self, tu: TranslationUnit) -> str:
        header = (
            "// Generated by GAL compiler  (C++23, freestanding)\n"
            "#pragma clang diagnostic push\n"
            "#pragma clang diagnostic ignored \"-Weverything\"\n"
        )

        body_parts: list[str] = []
        # First pass: populate registries so forward references work
        for stmt in tu.stmts:
            self._register(stmt)
        for stmt in tu.stmts:
            body_parts.append(self._stmt(stmt))

        tuple_preamble = self._tuples.preamble()
        hoisted        = "\n".join(self._hoisted)

        sections = [header]
        if tuple_preamble:
            sections.append(tuple_preamble)
        if hoisted:
            sections.append(hoisted)
        sections.extend(body_parts)
        sections.append("#pragma clang diagnostic pop\n")
        return "\n".join(s for s in sections if s)

    # ====================================================================
    # Statements
    # ====================================================================

    def _stmt(self, node: StmtNode) -> str:
        t = type(node)

        if t is CompoundStmt:    return self._compound(node)
        if t is ExprStmt:        return self._expr(node.expr) + ";"
        if t is VarDecl:         return self._var_decl(node)
        if t is TupleDestructure:return self._tuple_destructure(node)
        if t is TypeDecl:        return self._type_decl(node)
        if t is FunctionDecl:    return self._fn_decl(node)
        if t is OperatorOverload:return self._op_overload(node)
        if t is AliasDecl:       return self._alias_decl(node)
        if t is ImportDirective: return self._import(node)
        if t is IncludeDirective:return self._include(node)
        if t is ReturnStmt:      return self._return(node)
        if t is BreakStmt:       return "break;"
        if t is ContinueStmt:    return "continue;"
        if t is IfStmt:          return self._if_stmt(node)
        if t is WhileStmt:       return self._while_stmt(node)
        if t is ForStmt:         return self._for_stmt(node)
        if t is CaseStmt:        return self._case_stmt(node)
        if t is LabelDecl:       return f"{node.name}:;"
        if t is GotoStmt:        return self._goto(node)
        if t is MixinStmt:       return self._mixin(node)
        if t is AsmStmt:         return self._asm_stmt(node)
        raise NotImplementedError(f"_stmt: unhandled {t.__name__}")

    def _compound(self, node: CompoundStmt) -> str:
        inner = "\n".join(self._stmt(s) for s in node.body)
        return "{\n" + _indent(inner) + "\n}"

    def _return(self, node: ReturnStmt) -> str:
        if node.value is None:
            return "return;"
        return f"return {self._expr(node.value)};"

    def _goto(self, node: GotoStmt) -> str:
        if isinstance(node.target, int):
            # raw address goto — use computed goto extension
            return f"goto *((void*){node.target:#x});"
        return f"goto {node.target};"

    def _mixin(self, node: MixinStmt) -> str:
        # At compile time we could evaluate the string; for the POC emit a
        # static_assert so the programmer knows it needs the full pipeline.
        expr = self._expr(node.expr)
        return f"/* mixin({expr}) — runtime mixin not supported in transpiler */"

    # ---- variable declaration ---------------------------------------------

    def _var_decl(self, node: VarDecl) -> str:
        # Register const strings for mixin evaluation
        if node.keyword == "const" and node.initializer is not None:
            s = self._try_eval_string(node.initializer)
            if s is not None:
                self._const_strings[node.name] = s
        pragmas   = self._pragma_prefix(node.pragmas)
        storage   = ""

        # const → constexpr
        if node.keyword == "const":
            storage = "constexpr "
        elif self._pragma_has(node.pragmas, "static"):
            storage = "static "

        volatile_ = "volatile " if self._pragma_has(node.pragmas, "volatile") else ""
        align     = self._pragma_align(node.pragmas)
        deprecated= self._pragma_deprecated(node.pragmas)
        export_   = self._pragma_export_attr(node.pragmas)

        # bit:get → sizeof(T)*8  initializer override
        bit_get   = self._pragma_bit_get(node.pragmas)

        if node.type_ann is not None:
            decl = f"{storage}{volatile_}{align}{deprecated}{export_}{self._decl(node.type_ann, node.name)}"
        else:
            if node.keyword == "const":
                decl = f"constexpr auto {node.name}"
            else:
                decl = f"{storage}{volatile_}auto {node.name}"

        if bit_get is not None:
            return f"{decl} = {bit_get};"
        if node.initializer is not None:
            return f"{decl} = {self._expr(node.initializer)};"
        return f"{decl};"

    def _tuple_destructure(self, node: TupleDestructure) -> str:
        lines = []
        tmp   = f"_gal_td_{id(node) & 0xFFFF}"
        rhs   = self._expr(node.value)
        if node.keyword == "const":
            lines.append(f"constexpr auto {tmp} = {rhs};")
        else:
            lines.append(f"auto {tmp} = {rhs};")
        for i, name in enumerate(node.bindings):
            if name is None:
                continue
            if node.keyword == "const":
                lines.append(f"constexpr auto {name} = {tmp}._{i};")
            else:
                lines.append(f"auto {name} = {tmp}._{i};")
        return "\n".join(lines)

    # ---- type declaration -------------------------------------------------

    def _type_decl(self, node: TypeDecl) -> str:
        pragmas    = self._pragma_prefix(node.pragmas)
        deprecated = self._pragma_deprecated(node.pragmas)

        if node.generics:
            tparams = self._template_params(node.generics)
            prefix  = f"template<{tparams}>\n"
        else:
            prefix = ""

        inner = node.type_node
        t     = type(inner)

        # object → struct definition
        if t is ObjectType:
            return prefix + deprecated + self._object_def(node.name, inner,
                                                           node.pragmas)
        # enum → struct with constexpr statics
        if t is EnumType:
            return prefix + deprecated + self._enum_def(node.name, inner)

        # simple typedef
        ctype = self._type(inner)
        return f"{prefix}{deprecated}using {node.name} = {ctype};"

    def _object_def(self, name: str, node: ObjectType,
                    pragmas: List[Pragma]) -> str:
        packed  = "__attribute__((packed)) " if self._pragma_has(pragmas, "packed") else ""
        parents = ""
        if node.parents:
            parents = " : " + ", ".join(f"public {p}" for p in node.parents)

        field_lines: list[str] = []
        for f in node.fields:
            field_lines.append(self._field_decl(f))

        body = "\n".join(field_lines)
        return (f"struct {packed}{name}{parents} {{\n"
                + _indent(body) + "\n};")

    def _field_decl(self, node: FieldDecl) -> str:
        volatile_  = "volatile " if self._pragma_has(node.pragmas, "volatile") else ""
        static_    = "static "   if self._pragma_has(node.pragmas, "static")   else ""
        deprecated = self._pragma_deprecated(node.pragmas)
        bit_set    = self._pragma_bit_set(node.pragmas)
        align      = self._pragma_align(node.pragmas)

        if node.type_ann is not None:
            base = f"{deprecated}{static_}{volatile_}{align}"
            decl = self._decl(node.type_ann, node.name)
        else:
            base = f"{deprecated}{static_}{volatile_}{align}"
            decl = f"auto {node.name}"

        suffix = f" : {bit_set}" if bit_set is not None else ""
        init   = f" = {self._expr(node.initializer)}" if node.initializer is not None else ""
        return f"{base}{decl}{suffix}{init};"

    def _enum_def(self, name: str, node: EnumType) -> str:
        # GAL enums are tagged unions; each variant holds any value.
        # We emit a namespace-like struct with constexpr static members.
        lines = [f"struct {name} {{"]
        for v in node.variants:
            val = self._expr(v.value)
            lines.append(f"    static constexpr auto {v.name} = {val};")
        lines.append("};")
        return "\n".join(lines)

    # ---- function declaration --------------------------------------------

    def _fn_decl(self, node: FunctionDecl) -> str:
        attrs = self._pragma_fn_attrs(node.pragmas)
        deprecated = self._pragma_deprecated(node.pragmas)

        if node.generics:
            tparams = self._template_params(node.generics)
            prefix  = f"template<{tparams}>\n"
        else:
            prefix = ""

        if node.eval_prefix == "const":
            storage = "consteval "
        elif node.eval_prefix == "template":
            # Template fns are inlined at every call site by the transpiler.
            # No C++ definition is emitted — the body lives in _template_fns registry.
            return f"// template fn '{node.name}' inlined at call sites"
        else:
            storage = ""

        ret  = self._type(node.ret_type) if node.ret_type else "void"
        vg   = {g.name for g in node.generics if g.variadic}
        params = self._param_list(node.params, vg)
        body   = self._compound(node.body)
        export_link = self._pragma_export_linkage(node.pragmas)
        import_link = self._pragma_import_linkage(node.pragmas)

        if import_link:
            return f"{prefix}extern {ret} {node.name}({params}) {import_link};"

        return (f"{prefix}{deprecated}{export_link}{storage}"
                f"{attrs}{ret} {node.name}({params}) {body}")

    def _op_overload(self, node: OperatorOverload) -> str:
        if node.generics:
            prefix = f"template<{self._template_params(node.generics)}>\n"
        else:
            prefix = ""
        ret    = self._type(node.ret_type) if node.ret_type else "void"
        vg     = {g.name for g in node.generics if g.variadic}
        params = self._param_list(node.params, vg)
        body   = self._compound(node.body)

        # `for` overload — special: not a C++ operator
        if node.op == "for":
            # emit as a free function named _gal_for_<type>
            first_type = self._type(node.params[0].type_ann) if node.params else "auto"
            return (f"{prefix}{ret} _gal_for_{first_type.replace(' ','_').replace('*','p')}"
                    f"({params}) {body}")

        return f"{prefix}{ret} operator{node.op}({params}) {body}"

    # ---- alias -----------------------------------------------------------

    def _alias_decl(self, node: AliasDecl) -> str:
        if isinstance(node.target, CompoundStmt):
            # alias myScope = { ... } — emit an anonymous namespace-struct
            lines = [f"struct _gal_scope_{node.name} {{"]
            for s in node.target.body:
                lines.append(_indent(self._stmt(s)))
            lines.append("};")
            if node.name == "_":
                # unnamed alias — inject all exported names via using
                lines.append(f"// unnamed alias — members accessed via _gal_scope_{node.name}::")
            else:
                lines.append(f"static _gal_scope_{node.name} {node.name};")
            return "\n".join(lines)
        # expression alias
        target = self._expr(node.target)
        return f"auto& {node.name} = {target};"

    # ---- import / include -----------------------------------------------

    def _import(self, node: ImportDirective) -> str:
        # For the POC: resolve the file, parse it, emit only exported decls
        # as forward declarations. A full impl would do proper module tracking.
        path = self._resolve_path(node.path)
        if path is None:
            return f"// import \"{node.path}\" — file not found"
        try:
            src = open(path).read()
            tu2 = parse(lex(src))
            lines = [f"// import \"{node.path}\""]
            for s in tu2.stmts:
                fwd = self._forward_decl(s)
                if fwd:
                    lines.append(fwd)
            return "\n".join(lines)
        except Exception as e:
            return f"// import \"{node.path}\" failed: {e}"

    def _include(self, node: IncludeDirective) -> str:
        path = self._resolve_path(node.path)
        if path is None:
            return f"// include \"{node.path}\" — file not found"
        try:
            src  = open(path).read()
            tu2  = parse(lex(src))
            gen2 = Codegen(path, self._include_dirs)
            # re-use tuple registry so shapes stay consistent
            gen2._tuples = self._tuples
            return gen2.generate(tu2)
        except Exception as e:
            return f"// include \"{node.path}\" failed: {e}"

    def _resolve_path(self, path: str) -> Optional[str]:
        base = os.path.dirname(self._filename)
        for d in [base] + self._include_dirs:
            full = os.path.join(d, path)
            if os.path.isfile(full):
                return full
        return None

    def _forward_decl(self, node: StmtNode) -> str:
        """Emit a minimal forward declaration for exported top-level items."""
        if isinstance(node, FunctionDecl) and node.exported:
            ret    = self._type(node.ret_type) if node.ret_type else "void"
            params = self._param_list(node.params)
            prefix = f"template<{self._template_params(node.generics)}>\n" if node.generics else ""
            return f"{prefix}{ret} {node.name}({params});"
        if isinstance(node, VarDecl) and node.exported:
            if node.type_ann:
                return f"extern {self._type(node.type_ann)} {node.name};"
            return f"// extern auto {node.name}; // type not known without definition"
        if isinstance(node, TypeDecl) and node.exported:
            return self._type_decl(node)
        return ""

    def _register(self, node: StmtNode) -> None:
        """Populate transpile-time registries from top-level declarations."""
        if isinstance(node, FunctionDecl) and node.eval_prefix == "template":
            self._template_fns[node.name] = node
        if isinstance(node, TypeDecl):
            self._type_decls[node.name] = node
        if isinstance(node, VarDecl) and node.keyword == "const" and node.initializer is not None:
            s = self._try_eval_string(node.initializer)
            if s is not None:
                self._const_strings[node.name] = s

    # ----------------------------------------------------------------
    # Transpile-time string evaluation (for mixin)
    # ----------------------------------------------------------------

    def _try_eval_string(self, node: ExprNode) -> Optional[str]:
        """Try to evaluate an expression to a Python string at transpile time.
        Returns None if the expression cannot be statically evaluated."""
        if isinstance(node, StringLiteral):
            return node.value
        if isinstance(node, RawStringLiteral):
            return node.value
        if isinstance(node, IdentExpr):
            return self._const_strings.get(node.name)
        if isinstance(node, BinaryExpr) and node.op == "+":
            l = self._try_eval_string(node.left)
            r = self._try_eval_string(node.right)
            if l is not None and r is not None:
                return l + r
        return None

    # ----------------------------------------------------------------
    # Mixin — transpile-time expansion
    # ----------------------------------------------------------------

    def _mixin(self, node: MixinStmt) -> str:
        s = self._try_eval_string(node.expr)
        if s is None:
            expr = self._expr(node.expr)
            return f"/* mixin({expr}) — could not evaluate string at transpile time */"
        # Register any new declarations the mixin introduces
        try:
            inner_tu = parse(lex(s))
        except Exception as e:
            return f"/* mixin parse error: {e} */"
        for stmt in inner_tu.stmts:
            self._register(stmt)
        parts = [self._stmt(stmt) for stmt in inner_tu.stmts]
        return "\n".join(parts)

    # ----------------------------------------------------------------
    # Template fn — transpile-time body inlining at call sites
    # ----------------------------------------------------------------

    def _inline_template_call(self, fn: FunctionDecl,
                              args: List[ExprNode]) -> str:
        """Substitute parameters with argument expressions and emit the body."""
        # Build substitution map: param name → emitted arg string
        subst: dict[str, str] = {}
        for i, param in enumerate(fn.params):
            if i < len(args):
                arg = args[i]
                # char[_] param may receive a raw identifier (stringified, §17.2 [2])
                if isinstance(arg, IdentExpr) and param.type_ann is not None:
                    if (isinstance(param.type_ann, ArrayType) and
                            isinstance(param.type_ann.element, FundamentalType) and
                            param.type_ann.element.name == "char"):
                        subst[param.name] = f'"{arg.name}"'
                        continue
                subst[param.name] = self._expr(arg)
            elif param.default is not None:
                subst[param.name] = self._expr(param.default)

        # For char[_] params, register their string values so mixin inside
        # the body can evaluate string expressions using those names.
        injected: list[str] = []
        for i, param in enumerate(fn.params):
            if i < len(args):
                arg = args[i]
                val_str: Optional[str] = None
                if isinstance(param.type_ann, ArrayType) and \
                        isinstance(param.type_ann.element, FundamentalType) and \
                        param.type_ann.element.name == "char":
                    if isinstance(arg, IdentExpr):
                        val_str = arg.name          # identifier → its text
                    else:
                        val_str = self._try_eval_string(arg)
                else:
                    val_str = self._try_eval_string(arg)
                if val_str is not None:
                    self._const_strings[param.name] = val_str
                    injected.append(param.name)

        # Emit the body with substitution applied via whole-word string replacement
        raw = self._compound(fn.body)
        for pname, pval in subst.items():
            raw = re.sub(rf'\b{re.escape(pname)}\b', pval, raw)

        # Clean up injected param names so they don't pollute outer scope
        for name in injected:
            self._const_strings.pop(name, None)

        # Keep braces for safety (avoids name collisions)
        return raw



    def _if_stmt(self, node: IfStmt) -> str:
        if node.is_ifconst:
            # `if const { }` — tests compile-time context
            # emit as: if consteval { ... }  (C++23)
            return f"if consteval {self._compound(node.then_)}"

        keyword = "if"
        if node.prefix == "const":
            keyword = "if constexpr"
        elif node.prefix == "template":
            keyword = "if constexpr"  # best approximation

        cond  = self._expr(node.condition)
        then_ = self._compound(node.then_)
        parts = [f"{keyword} ({cond}) {then_}"]

        for elif_ in node.elifs:
            ec   = self._expr(elif_.condition)
            eb   = self._compound(elif_.body)
            parts.append(f"else {keyword} ({ec}) {eb}")

        if node.else_:
            parts.append(f"else {self._compound(node.else_)}")

        return " ".join(parts)

    def _while_stmt(self, node: WhileStmt) -> str:
        prefix = ""
        if node.prefix in ("const", "template"):
            prefix = "/* compile-time unroll */ "

        if node.init is not None:
            # Three-part while → C++ for
            iname = node.init.name or "_gal_wi"
            itype = self._type(node.init.type_ann) if node.init.type_ann else "auto"
            if node.init.keyword == "const":
                itype = "constexpr " + itype
            ival  = self._expr(node.init.value)
            cond  = self._expr(node.cond)
            upd   = self._expr(node.update) if node.update else ""
            body  = self._compound(node.body)
            return f"{prefix}for ({itype} {iname} = {ival}; {cond}; {upd}) {body}"

        cond = self._expr(node.cond)
        body = self._compound(node.body)
        return f"{prefix}while ({cond}) {body}"

    def _for_stmt(self, node: ForStmt) -> str:
        prefix = ""
        if node.prefix in ("const", "template"):
            prefix = "/* compile-time unroll */ "

        iterable = self._expr(node.iterable)
        elem     = node.elem_name or "_gal_e"
        body_src = self._compound(node.body)

        # Index binding
        idx_decl = ""
        if node.index is not None and node.index.name:
            idx_decl = f"__SIZE_TYPE__ {node.index.name} = 0; "

        # Use range-based for; if index needed, track separately
        elem_type = self._type(node.elem_type) if node.elem_type else "auto"
        if node.elem_keyword == "const":
            elem_type = "const " + elem_type

        if idx_decl:
            # Wrap in a block to scope the index
            inner = (f"{idx_decl}\n"
                     f"for ({elem_type}& {elem} : {iterable}) {{\n"
                     f"    {body_src[1:-1].strip()}\n"
                     f"    ++{node.index.name};\n"
                     f"}}")
            return f"{prefix}{{\n{_indent(inner)}\n}}"

        return f"{prefix}for ({elem_type}& {elem} : {iterable}) {body_src}"

    def _case_stmt(self, node: CaseStmt) -> str:
        disc  = self._expr(node.discriminant)
        lines = [f"switch ({disc}) {{"]
        for arm in node.arms:
            if arm.pattern is None:
                lines.append("    default: {")
            elif isinstance(arm.pattern, RangeExpr):
                # GCC/Clang range case extension
                lo = self._expr(arm.pattern.lo)
                hi = self._expr(arm.pattern.hi)
                lines.append(f"    case {lo} ... {hi}: {{")
            else:
                val = self._expr(arm.pattern)
                lines.append(f"    case {val}: {{")
            for s in arm.body:
                lines.append(_indent(self._stmt(s), 2))
            lines.append("    }")
        lines.append("}")
        return "\n".join(lines)

    # ====================================================================
    # Expressions
    # ====================================================================

    def _expr(self, node: ExprNode) -> str:
        t = type(node)

        if t is IntLiteral:    return str(node.value)
        if t is FloatLiteral:  return node.raw
        if t is BoolLiteral:   return "true" if node.value else "false"
        if t is CharLiteral:   return f"'\\x{node.value:02x}'"
        if t is StringLiteral: return self._string_lit(node.value)
        if t is RawStringLiteral: return self._raw_string_lit(node.value)
        if t is WildcardExpr:  return "_gal_discard"
        if t is IdentExpr:
            if node.name == "this": return "this"
            if node.name == "This": return "decltype(*this)"
            return node.name

        if t is ScopeExpr:
            if isinstance(node.scope, WildcardExpr):
                # _::name — enclosing outer scope, maps to C++ global :: prefix
                return f"::{node.name}"
            return f"{self._expr(node.scope)}::{node.name}"

        if t is MemberExpr:
            if node.member == "len":
                obj = self._expr(node.obj)
                return f"(sizeof({obj})/sizeof(({obj})[0]))"
            return f"{self._expr(node.obj)}.{node.member}"
        if t is ArrowExpr:     return f"{self._expr(node.ptr)}->{node.member}"

        if t is SubscriptExpr:
            return f"{self._expr(node.base)}[{self._expr(node.index)}]"

        if t is UnaryExpr:
            return f"({node.op}{self._expr(node.operand)})"

        if t is BinaryExpr:
            op = _OPERATOR_MAP.get(node.op, node.op)
            return f"({self._expr(node.left)} {op} {self._expr(node.right)})"

        if t is AssignExpr:
            op = _OPERATOR_MAP.get(node.op, node.op)
            return f"({self._expr(node.left)} {op} {self._expr(node.right)})"

        if t is AddrExpr:
            return f"(&{self._expr(node.operand)})"

        if t is CastExpr:
            ctype = self._type(node.target)
            return f"__builtin_bit_cast({ctype}, {self._expr(node.operand)})"

        if t is TypeConstructExpr:
            ctype = self._type(node.type_node)
            return f"static_cast<{ctype}>({self._expr(node.operand)})"

        if t is CallExpr:
            return self._call_expr(node)

        if t is TupleExpr:
            return self._tuple_expr(node)

        if t is ArrayLiteralExpr:
            return self._array_lit_expr(node)

        if t is RangeExpr:
            # bare range not inside [] — shouldn't appear; emit as comment
            lo = self._expr(node.lo); hi = self._expr(node.hi)
            return f"/* range {lo}..{hi} */"

        if t is LambdaExpr:
            return self._lambda_expr(node)

        # Reflection
        if t is TypeofExpr:
            return f"decltype({self._expr(node.operand)})"
        if t is SizeofExpr:
            return f"sizeof({self._type(node.type_node)})"
        if t is AlignofExpr:
            return f"alignof({self._type(node.type_node)})"
        if t is OffsetofExpr:
            return f"__builtin_offsetof({self._type(node.type_node)}, {self._expr(node.field)})"
        if t is StringofExpr:
            # Best we can do at transpile time — wrap in a macro-like string
            inner = self._expr(node.operand)
            return f"([] {{ return \"{inner}\"; }}())"
        if t is IdentofExpr:
            inner = self._expr(node.operand)
            return f"([] {{ return \"{inner}\"; }}())"
        if t is FieldofExpr:
            # Resolve at transpile time: look up the type and emit the field name
            tname = self._type(node.type_node)
            decl  = self._type_decls.get(tname)
            if (decl is not None and isinstance(decl.type_node, ObjectType)
                    and isinstance(node.index, IntLiteral)):
                idx    = node.index.value
                fields = decl.type_node.fields
                if 0 <= idx < len(fields):
                    return fields[idx].name
            return f"/* fieldof<{tname}>({self._expr(node.index)}) — unresolved */"
        if t is FieldSizeofExpr:
            return f"/* fieldSizeof<{self._type(node.type_node)}>() */"
        if t is ReturnofExpr:
            fn = self._expr(node.fn)
            return f"decltype({fn}())"

        if t is NamedArgExpr:
            return f".{node.name} = {self._expr(node.value)}"

        raise NotImplementedError(f"_expr: unhandled {t.__name__}")

    def _call_expr(self, node: CallExpr) -> str:
        # Template fn — inline the body at the call site
        callee_name = None
        if isinstance(node.callee, IdentExpr):
            callee_name = node.callee.name
        elif isinstance(node.callee, MemberExpr):
            callee_name = node.callee.member

        if callee_name and callee_name in self._template_fns:
            fn   = self._template_fns[callee_name]
            args = list(node.args)
            # UFCS: prepend the object as first arg
            if isinstance(node.callee, MemberExpr):
                args = [node.callee.obj] + args
            return self._inline_template_call(fn, args)

        args = ", ".join(self._expr(a) for a in node.args)
        generics = ""
        if node.generics:
            generics = "<" + ", ".join(self._type(g) for g in node.generics) + ">"

        if isinstance(node.callee, MemberExpr):
            obj    = self._expr(node.callee.obj)
            method = node.callee.member
            return f"{obj}.{method}{generics}({args})"

        callee = self._expr(node.callee)
        return f"{callee}{generics}({args})"

    def _tuple_expr(self, node: TupleExpr) -> str:
        type_sigs = ["auto"] * len(node.elements)
        struct_name = self._tuples.get(type_sigs, self)
        vals = ", ".join(self._expr(e) for e in node.elements)
        return f"{struct_name}{{{vals}}}"

    def _array_lit_expr(self, node: ArrayLiteralExpr) -> str:
        if node.range_lo is not None:
            lo = self._expr(node.range_lo)
            hi = self._expr(node.range_hi)
            # emit as a lambda returning an initializer-list-initialised array
            # (C++ doesn't have runtime ranges directly)
            return (f"([&]() {{\n"
                    f"    constexpr auto _lo = {lo}, _hi = {hi};\n"
                    f"    constexpr auto _n  = (_hi >= _lo) ? (_hi - _lo + 1) : (_lo - _hi + 1);\n"
                    f"    __INT32_TYPE__ _arr[_n];\n"
                    f"    for (__SIZE_TYPE__ _i = 0; _i < _n; ++_i)\n"
                    f"        _arr[_i] = (_hi >= _lo) ? _lo + (__INT32_TYPE__)_i : _lo - (__INT32_TYPE__)_i;\n"
                    f"    return _arr;\n"
                    f"}}())")
        elems = ", ".join(self._expr(e) for e in node.elements)
        return "{" + elems + "}"

    def _lambda_expr(self, node: LambdaExpr) -> str:
        params = self._param_list(node.params)
        ret    = (" -> " + self._type(node.ret_type)) if node.ret_type else ""
        body   = self._compound(node.body)
        if node.generics:
            tparams = self._template_params(node.generics)
            return f"[&]<{tparams}>({params}){ret} {body}"
        return f"[&]({params}){ret} {body}"

    def _string_lit(self, value: str) -> str:
        escaped = (value.replace("\\", "\\\\")
                        .replace('"',  '\\"')
                        .replace("\0", "\\0")
                        .replace("\n", "\\n")
                        .replace("\r", "\\r")
                        .replace("\t", "\\t"))
        return f'"{escaped}"'

    def _raw_string_lit(self, value: str) -> str:
        # Use C++ raw string literal; pick a delimiter unlikely to appear
        delim = "GAL"
        while f"){delim}\"" in value:
            delim += "_"
        return f'R"{delim}({value}){delim}"'

    # ====================================================================
    # Types
    # ====================================================================

    def _decl(self, type_node: TypeNode, name: str) -> str:
        """Emit a full C++ declaration 'type name' with correct array bracket placement."""
        if isinstance(type_node, ArrayType):
            suffix = "[]" if type_node.size is None else f"[{self._expr(type_node.size)}]"
            return self._decl(type_node.element, name + suffix)
        return f"{self._type(type_node)} {name}"

    def _type(self, node: TypeNode) -> str:
        t = type(node)

        if t is FundamentalType:
            return _FUNDAMENTAL_MAP.get(node.name, node.name)

        if t is VoidType:      return "void"
        if t is AutoType:      return "auto"
        if t is WildcardType:  return "auto"  # deduce
        if t is IdentType:     return node.name

        if t is PointerType:
            inner = self._type(node.pointee)
            qual  = "const " if node.qualifier == "imut" else ""
            return f"{qual}{inner}*"

        if t is ReferenceType:
            inner = self._type(node.referent)
            qual  = "const " if node.qualifier == "imut" else ""
            return f"{qual}{inner}&"

        if t is ValueType:
            inner = self._type(node.inner)
            return f"{inner}&&"

        if t is ArrayType:
            inner = self._type(node.element)
            if node.size is None:
                # _ size — array of unknown bound; usually used with initializer
                return f"{inner}[]"
            size = self._expr(node.size)
            return f"{inner}[{size}]"

        if t is TupleType:
            type_sigs = [self._type(e) for e in node.elements]
            return self._tuples.get(type_sigs, self)

        if t is UnionType:
            # Active type is compile-time fixed; we just use the first
            # alternative (the user must not mix types at runtime).
            # For a real impl this would need more analysis.
            return self._type(node.alternatives[0])

        if t is ObjectType:
            # Inline anonymous struct
            fields = "; ".join(
                f"{self._type(f.type_ann) if f.type_ann else 'auto'} {f.name}"
                for f in node.fields
            )
            return f"struct {{ {fields}; }}"

        if t is EnumType:
            # Inline anonymous enum-struct — emit as anonymous struct
            members = " ".join(
                f"static constexpr auto {v.name} = {self._expr(v.value)};"
                for v in node.variants
            )
            return f"struct {{ {members} }}"

        raise NotImplementedError(f"_type: unhandled {t.__name__}")

    # ====================================================================
    # Parameters / template params
    # ====================================================================

    def _param_list(self, params: List[Param],
                    variadic_generics: Optional[set] = None) -> str:
        parts = []
        vg = variadic_generics or set()
        for p in params:
            if p.type_ann is not None:
                ann = p.type_ann
                # T[_] where T is a variadic generic → T... name
                if (isinstance(ann, ArrayType) and
                        isinstance(ann.element, IdentType) and
                        ann.element.name in vg):
                    pack = ann.element.name
                    ctype = "const " if p.keyword == "const" else ""
                    parts.append(f"{ctype}{pack}... {p.name}")
                    continue
                # Array params: use _decl for correct bracket placement
                # but decay unknown-size arrays to pointer
                if isinstance(ann, ArrayType):
                    elem = self._type(ann.element)
                    ctype = ("const " if p.keyword == "const" else "") + elem + "*"
                    default = f" = {self._expr(p.default)}" if p.default else ""
                    parts.append(f"{ctype} {p.name}{default}")
                    continue
                ctype = self._type(ann)
            else:
                ctype = "auto"
            if p.keyword == "const":
                ctype = "const " + ctype
            default = f" = {self._expr(p.default)}" if p.default else ""
            parts.append(f"{ctype} {p.name}{default}")
        return ", ".join(parts)

    def _template_params(self, params: List[GenericParam]) -> str:
        parts = []
        for p in params:
            if p.variadic:
                parts.append(f"typename... {p.name}")
            elif p.default:
                parts.append(f"typename {p.name} = {self._type(p.default)}")
            else:
                parts.append(f"typename {p.name}")
        return ", ".join(parts)

    # ====================================================================
    # Pragma helpers
    # ====================================================================

    def _pragma_has(self, pragmas: List[Pragma], name: str) -> bool:
        return any(p.name == name for p in pragmas)

    def _pragma_get(self, pragmas: List[Pragma], name: str) -> Optional[Pragma]:
        for p in pragmas:
            if p.name == name:
                return p
        return None

    def _pragma_prefix(self, pragmas: List[Pragma]) -> str:
        return ""  # attrs placed inline below

    def _pragma_fn_attrs(self, pragmas: List[Pragma]) -> str:
        attrs: list[str] = []
        for p in pragmas:
            if p.name == "inline":
                if p.specifier == "always":
                    attrs.append("[[clang::always_inline]]")
                elif p.specifier == "never":
                    attrs.append("[[clang::noinline]]")
            elif p.name == "noreturn":
                attrs.append("[[noreturn]]")
            elif p.name == "cold":
                attrs.append("[[clang::cold]]")
            elif p.name == "hot":
                attrs.append("[[clang::hot]]")
            elif p.name == "convention":
                if p.argument:
                    conv = self._expr(p.argument).strip('"')
                    cattr = _CONV_MAP.get(conv, conv)
                    attrs.append(f"__attribute__(({cattr}))")
            elif p.name == "asmStackframe":
                if p.specifier == "off":
                    attrs.append('__attribute__((naked))')
        return " ".join(attrs) + (" " if attrs else "")

    def _pragma_deprecated(self, pragmas: List[Pragma]) -> str:
        p = self._pragma_get(pragmas, "deprecated")
        if p is None:
            return ""
        if p.argument:
            msg = self._expr(p.argument)
            return f"[[deprecated({msg})]] "
        return "[[deprecated]] "

    def _pragma_align(self, pragmas: List[Pragma]) -> str:
        p = self._pragma_get(pragmas, "align")
        if p is None:
            return ""
        return f"alignas({self._expr(p.argument)}) "

    def _pragma_export_attr(self, pragmas: List[Pragma]) -> str:
        p = self._pragma_get(pragmas, "export")
        if p is None:
            return ""
        sym = self._expr(p.argument)
        return f'__attribute__((visibility("default"), alias({sym}))) '

    def _pragma_export_linkage(self, pragmas: List[Pragma]) -> str:
        p = self._pragma_get(pragmas, "export")
        if p is None:
            return ""
        return '__attribute__((visibility("default"))) '

    def _pragma_import_linkage(self, pragmas: List[Pragma]) -> Optional[str]:
        p = self._pragma_get(pragmas, "import")
        if p is None:
            return None
        sym = self._expr(p.argument)
        return f'__asm__({sym})'

    def _pragma_bit_get(self, pragmas: List[Pragma]) -> Optional[str]:
        p = self._pragma_get(pragmas, "bit")
        if p is None or p.specifier != "get":
            return None
        ctype = self._type(p.argument) if p.argument else "int"
        return f"sizeof({ctype}) * 8"

    def _pragma_bit_set(self, pragmas: List[Pragma]) -> Optional[str]:
        p = self._pragma_get(pragmas, "bit")
        if p is None or p.specifier != "set":
            return None
        return self._expr(p.argument)

    def _pragma_section(self, pragmas: List[Pragma]) -> str:
        p = self._pragma_get(pragmas, "section")
        if p is None:
            return ""
        sec = self._expr(p.argument)
        return f"__attribute__((section({sec}))) "

    # ====================================================================
    # Inline assembly
    # ====================================================================

    def _asm_stmt(self, node: AsmStmt) -> str:
        tmpl = self._expr(node.template)
        outs = ", ".join(self._expr(o) for o in node.outputs if o is not None)
        ins  = ", ".join(self._expr(i) for i in node.inputs  if i is not None)
        clob = ", ".join(self._expr(c) for c in node.clobbers if c is not None)
        parts = [tmpl]
        if outs or ins or clob:
            parts.append(f": {outs}")
        if ins or clob:
            parts.append(f": {ins}")
        if clob:
            parts.append(f": {clob}")
        return f"__asm__ volatile({' '.join(parts)});"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate(tu: TranslationUnit, filename: str = "<input>",
             include_dirs: Optional[List[str]] = None) -> str:
    return Codegen(filename, include_dirs).generate(tu)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: codegen.py <file.gal>", file=sys.stderr)
        sys.exit(1)
    src  = open(sys.argv[1]).read()
    tu   = parse(lex(src))
    print(generate(tu, sys.argv[1]))
