"""
GAL AST Nodes

One dataclass per grammatical construct.  All nodes carry (line, col) for
diagnostics.  Optional fields use None as the absent sentinel.

Section references are to the GAL specification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class Node:
    line: int
    col:  int


# ---------------------------------------------------------------------------
# § 5  Types
# ---------------------------------------------------------------------------

@dataclass
class FundamentalType(Node):
    """intd, usize, bool, char, void, etc."""
    name: str          # exact keyword text


@dataclass
class IdentType(Node):
    """A user-defined or generic type referenced by name."""
    name: str


@dataclass
class PointerType(Node):
    """ptr mut? T   [basic.compound.ptr]"""
    qualifier: Optional[str]   # 'mut' | 'imut' | None
    pointee:   "TypeNode"


@dataclass
class ReferenceType(Node):
    """ref mut? T   [basic.compound.ref]"""
    qualifier: Optional[str]
    referent:  "TypeNode"


@dataclass
class ValueType(Node):
    """val mut? T   [basic.compound.val]"""
    qualifier: Optional[str]
    inner:     "TypeNode"


@dataclass
class ArrayType(Node):
    """T[N]  or  T[_]   [basic.array]"""
    element: "TypeNode"
    size:    Optional["ExprNode"]   # None means _ (deduce from initializer)


@dataclass
class TupleType(Node):
    """(T1, T2, ...)   [basic.tuple]"""
    elements: List["TypeNode"]


@dataclass
class UnionType(Node):
    """T1 | T2 | ...   [basic.union]"""
    alternatives: List["TypeNode"]


@dataclass
class ObjectType(Node):
    """object , Parent1, Parent2 { fields }   [class]"""
    parents: List[str]            # identifier names of parent types
    fields:  List["FieldDecl"]


@dataclass
class EnumType(Node):
    """enum { Var = expr; ... }   [dcl.enum]"""
    variants: List["EnumVariant"]


@dataclass
class AutoType(Node):
    """auto   [dcl.wild.auto]"""


@dataclass
class WildcardType(Node):
    """_  used as type   [dcl.wild.underscore]"""


@dataclass
class VoidType(Node):
    """void   [basic.void]"""


# Union alias for all type nodes
TypeNode = (
    FundamentalType | IdentType | PointerType | ReferenceType | ValueType |
    ArrayType | TupleType | UnionType | ObjectType | EnumType |
    AutoType | WildcardType | VoidType
)


# ---------------------------------------------------------------------------
# § 14  Object fields
# ---------------------------------------------------------------------------

@dataclass
class FieldDecl(Node):
    """A single field inside an object body.   [class.general]"""
    keyword:    Optional[str]       # 'const' | 'let' | 'var' | None
    name:       str
    exported:   bool                # visibility modifier *
    pragmas:    List["Pragma"]
    type_ann:   Optional[TypeNode]
    initializer: Optional["ExprNode"]


# ---------------------------------------------------------------------------
# § 15  Enum variant
# ---------------------------------------------------------------------------

@dataclass
class EnumVariant(Node):
    name:  str
    value: "ExprNode"


# ---------------------------------------------------------------------------
# § 16.4  Generic parameter
# ---------------------------------------------------------------------------

@dataclass
class GenericParam(Node):
    name:     str
    default:  Optional[TypeNode]   # T = SomeType
    variadic: bool                 # T...


# ---------------------------------------------------------------------------
# § 28  Pragmas
# ---------------------------------------------------------------------------

@dataclass
class Pragma(Node):
    """[. pragma-list .]   [dcl.pragma]"""
    name:      str
    specifier: Optional[str]       # e.g. 'always' in inline: always
    argument:  Optional["ExprNode"]


# ---------------------------------------------------------------------------
# § 33  Expressions
# ---------------------------------------------------------------------------

@dataclass
class IntLiteral(Node):
    value: int
    raw:   str


@dataclass
class FloatLiteral(Node):
    value: float
    raw:   str


@dataclass
class BoolLiteral(Node):
    value: bool


@dataclass
class CharLiteral(Node):
    value: int    # code point


@dataclass
class StringLiteral(Node):
    value: str    # decoded


@dataclass
class RawStringLiteral(Node):
    value: str    # verbatim content


@dataclass
class IdentExpr(Node):
    name: str


@dataclass
class WildcardExpr(Node):
    """Standalone _ as an expression (discard / omit)."""


@dataclass
class ScopeExpr(Node):
    """S::N   [basic.lookup.qual] / [lex.scope]"""
    scope: "ExprNode"    # left side (may be _ for outer scope)
    name:  str


@dataclass
class MemberExpr(Node):
    """obj.field   [lex.scope]"""
    obj:    "ExprNode"
    member: str


@dataclass
class ArrowExpr(Node):
    """ptr->field   [lex.scope]"""
    ptr:    "ExprNode"
    member: str


@dataclass
class SubscriptExpr(Node):
    """ptr[idx]  or  arr[idx]   [basic.compound.ptr]"""
    base:  "ExprNode"
    index: "ExprNode"


@dataclass
class CallExpr(Node):
    """f(args)  or  x.f(args) after UFCS rewrite   [over.call]"""
    callee:   "ExprNode"
    args:     List["ExprNode"]
    generics: List[TypeNode]      # explicit generic args <T, U>


@dataclass
class UnaryExpr(Node):
    op:      str           # '!' | '-' | '+'
    operand: "ExprNode"


@dataclass
class BinaryExpr(Node):
    op:    str             # one of the binary operators
    left:  "ExprNode"
    right: "ExprNode"


@dataclass
class AssignExpr(Node):
    op:    str             # '=' | '+=' | '-=' | '*=' | '/=' | '%='
    left:  "ExprNode"
    right: "ExprNode"


@dataclass
class AddrExpr(Node):
    """addr(expr)   [basic.compound.addr]"""
    operand: "ExprNode"


@dataclass
class CastExpr(Node):
    """cast<T>(expr)   [expr.cast]"""
    target:  TypeNode
    operand: "ExprNode"


@dataclass
class TypeConstructExpr(Node):
    """T(expr)  — non-reinterpret conversion   [dcl.fct.typecast]"""
    type_node: TypeNode
    operand:   "ExprNode"


@dataclass
class TupleExpr(Node):
    """(e1, e2, ...)   [basic.tuple]"""
    elements: List["ExprNode"]


@dataclass
class ArrayLiteralExpr(Node):
    """[e1, e2, ...]  or  [N1 .. N2]"""
    elements: List["ExprNode"]          # empty if range
    range_lo: Optional["ExprNode"]      # N1 in N1..N2
    range_hi: Optional["ExprNode"]      # N2


@dataclass
class RangeExpr(Node):
    """N1 .. N2   [basic.array.range]"""
    lo: "ExprNode"
    hi: "ExprNode"


@dataclass
class LambdaExpr(Node):
    """fn<T...>(params) : ret { body }   [dcl.fct.lambda]"""
    generics: List[GenericParam]
    params:   List["Param"]
    ret_type: Optional[TypeNode]
    body:     "CompoundStmt"


@dataclass
class TypeofExpr(Node):
    """typeof(expr)   [over.reflect]"""
    operand: "ExprNode"


@dataclass
class SizeofExpr(Node):
    """sizeof<T>()   [over.reflect]"""
    type_node: TypeNode


@dataclass
class AlignofExpr(Node):
    """alignof<T>()   [over.reflect]"""
    type_node: TypeNode


@dataclass
class OffsetofExpr(Node):
    """offsetof<T>(field)   [over.reflect]"""
    type_node: TypeNode
    field:     "ExprNode"


@dataclass
class StringofExpr(Node):
    """stringof(expr)   [over.reflect]"""
    operand: "ExprNode"


@dataclass
class IdentofExpr(Node):
    """identof(expr)   [over.reflect]"""
    operand: "ExprNode"


@dataclass
class FieldofExpr(Node):
    """fieldof<T>(index)   [over.reflect]"""
    type_node: TypeNode
    index:     "ExprNode"


@dataclass
class FieldSizeofExpr(Node):
    """fieldSizeof<T>()   [over.reflect]"""
    type_node: TypeNode


@dataclass
class ReturnofExpr(Node):
    """returnof(fn)   [over.reflect]"""
    fn: "ExprNode"


@dataclass
class NamedArgExpr(Node):
    """identifier: expr  — used in object construction  [class.ctor]"""
    name:  str
    value: "ExprNode"


# Union alias for all expression nodes
ExprNode = (
    IntLiteral | FloatLiteral | BoolLiteral | CharLiteral |
    StringLiteral | RawStringLiteral |
    IdentExpr | WildcardExpr | ScopeExpr | MemberExpr | ArrowExpr |
    SubscriptExpr | CallExpr | UnaryExpr | BinaryExpr | AssignExpr |
    AddrExpr | CastExpr | TypeConstructExpr |
    TupleExpr | ArrayLiteralExpr | RangeExpr | LambdaExpr |
    TypeofExpr | SizeofExpr | AlignofExpr | OffsetofExpr |
    StringofExpr | IdentofExpr | FieldofExpr | FieldSizeofExpr | ReturnofExpr |
    NamedArgExpr
)


# ---------------------------------------------------------------------------
# § 16  Function parameter
# ---------------------------------------------------------------------------

@dataclass
class Param(Node):
    keyword:    Optional[str]       # 'const' | 'let' | 'var' | None
    name:       str
    type_ann:   Optional[TypeNode]
    default:    Optional[ExprNode]


# ---------------------------------------------------------------------------
# § 7  Declarations (statement-level)
# ---------------------------------------------------------------------------

@dataclass
class VarDecl(Node):
    """const/let/var id * <G> [pragma] : T = expr ;   [dcl.var]"""
    keyword:    str                 # 'const' | 'let' | 'var'
    name:       str
    exported:   bool
    generics:   List[GenericParam]
    pragmas:    List[Pragma]
    type_ann:   Optional[TypeNode]
    initializer: Optional[ExprNode]


@dataclass
class TypeDecl(Node):
    """type id * <G> [pragma] = T ;   [dcl.type]"""
    name:     str
    exported: bool
    generics: List[GenericParam]
    pragmas:  List[Pragma]
    type_node: TypeNode


@dataclass
class FunctionDecl(Node):
    """(const|template)? fn id * <G> (params) [pragma] : ret { body }   [dcl.fct]"""
    eval_prefix: Optional[str]      # 'const' | 'template' | None
    name:        str
    exported:    bool
    generics:    List[GenericParam]
    params:      List[Param]
    pragmas:     List[Pragma]
    ret_type:    Optional[TypeNode]
    body:        "CompoundStmt"


@dataclass
class OperatorOverload(Node):
    """fn `op` <G> (params) : ret { body }   [over.general]"""
    op:       str                   # the operator symbol text
    generics: List[GenericParam]
    params:   List[Param]
    ret_type: Optional[TypeNode]
    body:     "CompoundStmt"


@dataclass
class AliasDecl(Node):
    """alias id * = target ;   [basic.scope]"""
    name:     str                   # '_' means unnamed / inject
    exported: bool
    target:   "ExprNode | CompoundStmt"


@dataclass
class ImportDirective(Node):
    """import "file" * ;   [dcl.import]"""
    path:     str
    reexport: bool


@dataclass
class IncludeDirective(Node):
    """include "file" ;   [dcl.import]"""
    path: str


# ---------------------------------------------------------------------------
# § 19  Statements
# ---------------------------------------------------------------------------

@dataclass
class CompoundStmt(Node):
    """{ stmt* }   [stmt.general]"""
    body: List["StmtNode"]


@dataclass
class ExprStmt(Node):
    """expr ;"""
    expr: ExprNode


@dataclass
class ReturnStmt(Node):
    """return expr? ;   [dcl.fct]"""
    value: Optional[ExprNode]


@dataclass
class BreakStmt(Node):
    pass


@dataclass
class ContinueStmt(Node):
    pass


@dataclass
class IfStmt(Node):
    """(const|template)? if (cond) { } elif* else?   [stmt.if]
       Also handles bare `if const { }` (compile-time context test)."""
    prefix:    Optional[str]        # 'const' | 'template' | None
    is_ifconst: bool                # True for `if const { }` form
    condition: Optional[ExprNode]   # None when is_ifconst
    then_:     CompoundStmt
    elifs:     List["ElifClause"]
    else_:     Optional[CompoundStmt]


@dataclass
class ElifClause(Node):
    condition: ExprNode
    body:      CompoundStmt


@dataclass
class WhileStmt(Node):
    """(const|template)? while cond { }
       or  while init, cond, update { }   [stmt.iter]"""
    prefix:  Optional[str]
    init:    Optional["ForInit"]    # three-part form
    cond:    ExprNode
    update:  Optional[ExprNode]     # three-part form
    body:    CompoundStmt


@dataclass
class ForInit(Node):
    """The initializer clause of a three-part while."""
    keyword: Optional[str]
    name:    str
    type_ann: Optional[TypeNode]
    value:   ExprNode


@dataclass
class ForIndex(Node):
    """Optional index binding in a for statement."""
    keyword:  Optional[str]
    name:     Optional[str]         # None when _
    type_ann: Optional[TypeNode]
    value:    ExprNode


@dataclass
class ForStmt(Node):
    """(const|template)? for idx? elem, iterable { }   [stmt.iter]"""
    prefix:  Optional[str]
    index:   Optional[ForIndex]
    elem_keyword: Optional[str]
    elem_name:    Optional[str]     # None when _
    elem_type:    Optional[TypeNode]
    iterable:     ExprNode
    body:         CompoundStmt


@dataclass
class LabelDecl(Node):
    """label id ;   [stmt.label]"""
    name: str


@dataclass
class GotoStmt(Node):
    """goto id ;  or  goto integer ;   [stmt.label]"""
    target: "str | int"             # identifier name or raw address


@dataclass
class MixinStmt(Node):
    """mixin(expr) ;   [dcl.mixin]"""
    expr: ExprNode


@dataclass
class CaseStmt(Node):
    """case(expr) { label arm: stmts ... }   [stmt.case]"""
    discriminant: ExprNode
    arms:         List["CaseArm"]


@dataclass
class CaseArm(Node):
    """label expr|range|_ : stmts"""
    pattern: "ExprNode | RangeExpr | None"   # None = wildcard _
    body:    List["StmtNode"]


@dataclass
class AsmStmt(Node):
    """asm(str, outs, ins, clobbers) ;   [stmt.asm]"""
    template:  ExprNode
    outputs:   List[Optional[ExprNode]]   # None for _
    inputs:    List[Optional[ExprNode]]
    clobbers:  List[Optional[ExprNode]]


@dataclass
class TupleDestructure(Node):
    """const/let/var (a, _, c) : T = expr ;   [basic.tuple]"""
    keyword:  str
    bindings: List[Optional[str]]   # None for _
    type_ann: Optional[TypeNode]
    value:    ExprNode


# Union alias for all statement nodes
StmtNode = (
    CompoundStmt | ExprStmt | VarDecl | TypeDecl | FunctionDecl |
    OperatorOverload | AliasDecl | ImportDirective | IncludeDirective |
    ReturnStmt | BreakStmt | ContinueStmt |
    IfStmt | WhileStmt | ForStmt |
    LabelDecl | GotoStmt | MixinStmt | CaseStmt | AsmStmt |
    TupleDestructure
)


# ---------------------------------------------------------------------------
# Top-level translation unit
# ---------------------------------------------------------------------------

@dataclass
class TranslationUnit(Node):
    stmts: List[StmtNode]
