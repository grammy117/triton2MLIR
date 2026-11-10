"""Checked standard-math boundary for pure default libdevice declarations.

Reuse Meta's unary/binary conversions from ConversionPatterns.hpp at
17f0c234ce474bee334af36e351c62eca7852836. That table duplicates the f32
tanh/atanh entries; the configured 0c27923 optimizer also leaves their f64
forms untouched. Only those two missing mappings are expanded here.

The logical profile selects standard MLIR math, not a specific GPU library's
approximations or hidden effects. A symbol alone cannot override library/type
identity, purity or source attributes.
"""

from xdsl.dialects import builtin, math
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError

PROFILE = "logical-libdevice-standard-math-v1"
UNARY = {
    "fabs": "absf", "sin": "sin", "cos": "cos", "tan": "tan",
    "asin": "asin", "acos": "acos", "atan": "atan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "asinh": "asinh", "acosh": "acosh", "atanh": "atanh",
    "log": "log", "log10": "log10", "log1p": "log1p",
    "exp": "exp", "exp2": "exp2", "erf": "erf",
    "sqrt": "sqrt", "rsqrt": "rsqrt", "ceil": "ceil",
    "floor": "floor", "trunc": "trunc",
}
BINARY = {"atan2": "atan2", "pow": "powf"}
SYMBOLS = {f"__nv_{name}{suffix}": (f"math.{target}", dtype, arity)
           for names, arity in ((UNARY, 1), (BINARY, 2))
           for name, target in names.items()
           for suffix, dtype in (("f", builtin.f32), ("", builtin.f64))}


def contract(op, source_id):
    def refuse(message):
        raise ConversionError(message, opcode="tt.extern_elementwise", source_id=source_id)

    def attr(name):
        return op.properties.get(name, op.attributes.get(name))

    pure = attr("pure")
    if not isinstance(pure, builtin.IntegerAttr) or pure.type != builtin.i1 or pure.value.data == 0:
        refuse("external math requires a typed pure declaration; effects need their own implementation")
    symbol, library, path = (attr(name) for name in ("symbol", "libname", "libpath"))
    if not all(isinstance(value, builtin.StringAttr) for value in (symbol, library, path)):
        refuse("external math requires typed symbol and library declarations")
    if path.data or library.data not in ("", "libdevice"):
        refuse("external implementation cannot be replaced by a default-library symbol mapping")
    selected = SYMBOLS.get(symbol.data)
    if selected is None:
        refuse("external symbol needs a qualified executable implementation")
    target, dtype, arity = selected
    if len(op.operands) != arity or len(op.results) != 1:
        refuse("external math arity differs from its declared library function")
    result_type = op.results[0].type
    element = result_type.element_type if isinstance(result_type, builtin.TensorType) else result_type
    if element != dtype or any(value.type != result_type for value in op.operands):
        refuse("external math operand/result types differ from the declared library prototype")
    if isinstance(result_type, builtin.TensorType) and (
        any(n < 0 for n in result_type.get_shape()) or not isinstance(result_type.encoding, builtin.NoneAttr)
    ):
        refuse("external math requires static logical tiles without physical encodings")
    if any(name not in ("op_name", "op_name__", "pure", "symbol", "libname", "libpath")
           and not name.startswith("prov.") for name in (*op.properties, *op.attributes)):
        refuse("external math has an unqualified source property")
    return {"profile": PROFILE, "symbol": symbol.data, "library": library.data,
            "library_path": path.data, "pure": True, "target": target,
            "dtype": str(dtype), "arity": arity,
            "numeric_boundary": "standard-MLIR-math; target-accuracy-is-separate",
            "physical_mapping": "undecided"}


def expand_missing(module, source_ids):
    from .shared_tensor import opcode

    plans = []
    for op in module.walk():
        if opcode(op) != "tt.extern_elementwise":
            continue
        ids = source_ids(op.location)
        record = contract(op, next(iter(ids), "external-math"))
        if record["symbol"] in ("__nv_tanh", "__nv_atanh"):
            plans.append((op, record, ids))
    reports = []
    for original, record, ids in plans:
        replacement = (math.TanhOp if record["symbol"] == "__nv_tanh" else math.AtanhOp)(original.operands[0])
        replacement.location = original.location
        replacement.attributes.update({name: value for name, value in original.attributes.items() if name.startswith("prov.")})
        Rewriter.replace_op(original, replacement)
        reports.append({"kind": "source-external-math-table-completion", "source_ids": ids, **record})
    return reports
