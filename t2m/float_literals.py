"""Preserve IEEE literal words before Python floating conversions lose them.

The typed-token adapter follows xDSL 0.68's AttributeParser. IEEE NaNs are
carried separately from its approximate FloatAttr until source auditing ends;
executable constants then use standard integer constants and arith.bitcast.
No custom attribute, dialect or literal annotation supplies runtime semantics.
"""

# The scalar typed-token adapter below is adapted from xDSL 0.68.0's
# xdsl/parser/attribute_parser.py (Apache-2.0 WITH LLVM-exception).
# Changes retain original IEEE storage words before FloatAttr conversion.
# The upstream license is retained in docs/licenses/xdsl-LICENSE.txt.

import math
from dataclasses import dataclass

from xdsl.dialects import arith, builtin
from xdsl.parser import Parser
from xdsl.rewriter import Rewriter
from xdsl.utils.mlir_lexer import MLIRTokenKind

from .diagnostics import ConversionError

FORMATS = {
    "f16": (2, 0x7C00, 0x03FF),
    "bf16": (2, 0x7F80, 0x007F),
    "f32": (4, 0x7F800000, 0x007FFFFF),
    "f64": (8, 0x7FF0000000000000, 0x000FFFFFFFFFFFFF),
}


def nan_count(data, dtype):
    if dtype.name not in FORMATS:
        return 0
    size, exponent, fraction = FORMATS[dtype.name]
    return sum(
        word & exponent == exponent and word & fraction != 0
        for word in (
            int.from_bytes(data[index : index + size], "little")
            for index in range(0, len(data), size)
        )
    )


@dataclass(frozen=True)
class LiteralWords:
    attribute: builtin.FloatAttr | builtin.DenseIntOrFPElementsAttr
    data: bytes
    dtype: builtin.AnyFloat

    @property
    def spelling(self):
        if isinstance(self.attribute, builtin.FloatAttr):
            return "0x" + self.data[::-1].hex().upper() + " : " + str(self.dtype)
        return (
            'dense<"0x' + self.data.hex().upper() + '"> : ' + str(self.attribute.type)
        )

    def record(self):
        return {
            "dtype": self.dtype.name,
            "type": str(self.attribute.get_type()),
            "word_bits": len(self.data) * 8
            if isinstance(self.attribute, builtin.FloatAttr)
            else self.dtype.bitwidth,
            "encoding": "little-endian-IEEE-words",
            "words": self.data.hex(),
            "nan_count": nan_count(self.data, self.dtype),
            "shape": None
            if isinstance(self.attribute, builtin.FloatAttr)
            else list(self.attribute.type.get_shape()),
        }


class LiteralParser(Parser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._floating_literal_words = {}
        self._dense_literal_words = []

    def _retain(self, attribute, data, dtype):
        if nan_count(data, dtype):
            self._floating_literal_words[id(attribute)] = LiteralWords(
                attribute, data, dtype
            )
        return attribute

    def parse_optional_builtin_int_or_float_attr(self):
        # Adapt the upstream typed token conversion, never printed IR lines.
        boolean = self.try_parse_builtin_boolean_attr()
        if boolean is not None:
            return boolean
        hexadecimal = self._current_token.text[:2] in ("0x", "0X")
        value = self.parse_optional_number()
        if value is None:
            return None
        if self._current_token.kind != MLIRTokenKind.COLON:
            return (
                builtin.FloatAttr(value, builtin.f64)
                if isinstance(value, float)
                else builtin.IntegerAttr(value, builtin.i64)
            )
        dtype = self._parse_attribute_type()
        if isinstance(dtype, builtin.AnyFloat):
            raw = (
                value.to_bytes(dtype.compile_time_size, "little")
                if hexadecimal
                else dtype.pack([float(value)])
            )
            approximate = next(dtype.iter_unpack(raw))
            attribute = builtin.FloatAttr(approximate, dtype)
            return self._retain(attribute, raw, dtype)
        if isinstance(dtype, (builtin.IntegerType, builtin.IndexType)):
            if isinstance(value, float):
                self.raise_error("Floating point value is not valid for integer type.")
            return builtin.IntegerAttr(value, dtype)
        self.raise_error("Invalid type given for integer or float attribute.")

    def parse_dense_int_or_fp_elements_attr(self, type):
        words = []
        self._dense_literal_words.append(words)
        try:
            attribute = super().parse_dense_int_or_fp_elements_attr(type)
        finally:
            self._dense_literal_words.pop()
        dtype = attribute.type.element_type
        if dtype.name not in FORMATS:
            return attribute
        raw = b"".join(words) if words else attribute.data.data
        count = len(attribute.data.data) // dtype.compile_time_size
        if words and len(words) == 1:
            raw *= count
        if len(raw) != len(attribute.data.data):
            self.raise_error(
                "floating literal token inventory differs from its typed dense shape"
            )
        exact = builtin.DenseIntOrFPElementsAttr(attribute.type, builtin.BytesAttr(raw))
        return self._retain(exact, raw, dtype)

    class _TensorLiteralElement(Parser._TensorLiteralElement):
        def to_type(self, parser, type_):
            if isinstance(
                type_, builtin.AnyFloat
            ) and self.span.text.lower().startswith("0x"):
                raw = int(self.span.text, 16).to_bytes(
                    type_.compile_time_size, "little"
                )
                number = next(type_.iter_unpack(raw))
            else:
                number = super().to_type(parser, type_)
                raw = (
                    type_.pack([number])
                    if isinstance(type_, builtin.AnyFloat)
                    else None
                )
            if raw is not None and parser._dense_literal_words:
                parser._dense_literal_words[-1].append(raw)
            return number

    def literal_words(self, module):
        result, consumed = {}, set()
        for op in module.walk():
            value = op.properties.get("value", op.attributes.get("value"))
            literal = self._floating_literal_words.get(id(value))
            if literal is None:
                continue
            if (
                op.name != "arith.constant"
                or len(op.results) != 1
                or value is not literal.attribute
                or value.get_type() != op.results[0].type
            ):
                raise ConversionError(
                    "exact floating literal words require their original typed arith.constant",
                    opcode=op.name,
                )
            result[op] = literal
            consumed.add(id(value))
        if consumed != set(self._floating_literal_words):
            raise ConversionError(
                "nonconstant NaN attributes need their own exact-word representation"
            )
        return result


def properties(op, literals=None):
    result = {key: str(value) for key, value in op.properties.items()}
    literal = (literals or {}).get(op)
    if literal is not None:
        if op.properties.get("value") is not literal.attribute:
            raise ConversionError("source floating literal changed before inventory")
        result["value"] = literal.spelling
    return result


def materialize(module, literals, *, source_ids=None):
    """Make every captured word executable, preserving its source location."""
    reports, plans = [], []
    attached = set(module.walk())
    for original, literal in literals.items():
        if (
            original not in attached
            or original.name != "arith.constant"
            or len(original.results) != 1
            or original.properties.get("value") is not literal.attribute
            or original.results[0].type != literal.attribute.get_type()
            or literal.dtype.name not in FORMATS
            or (
                literal.attribute.type
                if isinstance(literal.attribute, builtin.FloatAttr)
                else literal.attribute.type.element_type
            )
            != literal.dtype
            or not literal.data
            or len(literal.data) % FORMATS[literal.dtype.name][0]
            or not nan_count(literal.data, literal.dtype)
        ):
            raise ConversionError(
                "floating literal changed before its exact words were materialized"
            )
        shape = (
            ()
            if isinstance(literal.attribute, builtin.FloatAttr)
            else literal.attribute.type.get_shape()
        )
        if len(literal.data) // FORMATS[literal.dtype.name][0] not in (
            1,
            math.prod(shape),
        ):
            raise ConversionError(
                "floating literal storage differs from its original typed shape"
            )
        dtype = builtin.IntegerType(literal.dtype.bitwidth)
        if isinstance(literal.attribute, builtin.FloatAttr):
            value = builtin.IntegerAttr(
                int.from_bytes(literal.data, "little", signed=True), dtype
            )
        else:
            type_ = literal.attribute.type
            if isinstance(type_, builtin.TensorType):
                integer_type = builtin.TensorType(
                    dtype, type_.get_shape(), type_.encoding
                )
            elif isinstance(type_, builtin.VectorType):
                integer_type = builtin.VectorType(
                    dtype, type_.get_shape(), type_.scalable_dims
                )
            else:
                raise ConversionError(
                    "floating literal words require scalar, tensor or vector storage"
                )
            value = builtin.DenseIntOrFPElementsAttr(
                integer_type, builtin.BytesAttr(literal.data)
            )
        integer = arith.ConstantOp(value)
        restored = arith.BitcastOp(integer.result, original.results[0].type)
        for child in (integer, restored):
            child.location = original.location
            child.attributes.update(original.attributes)
        reports.append(
            {
                "kind": "source-floating-literal-words",
                "location": str(original.location),
                "source_ids": source_ids(original.location)
                if source_ids is not None
                else [],
                **literal.record(),
            }
        )
        plans.append((original, integer, restored))
    for original, integer, restored in plans:
        Rewriter.replace_op(original, (integer, restored), restored.results)
    return reports


def validate_reports(records, reports):
    """Validate exact source word declarations without granting equivalence."""
    from .ir import context

    source = {
        record["id"]: record for record in records if "floating_literal_words" in record
    }
    if not isinstance(reports, list):
        raise ValueError(  # noqa: TRY004 - invalid external evidence is a value error.
            "floating literal reports require their complete source inventory"
        )
    seen = set()
    fields = {"dtype", "type", "word_bits", "encoding", "words", "nan_count", "shape"}
    for report in reports:
        if (
            not isinstance(report, dict)
            or set(report) != fields | {"kind", "location", "source_ids"}
            or report.get("kind") != "source-floating-literal-words"
            or not isinstance(report.get("source_ids"), list)
            or any(not isinstance(value, str) for value in report["source_ids"])
        ):
            raise ValueError(
                "floating literal reports differ from their exact word schema"
            )
        info = {key: report[key] for key in fields}
        dtype = info.get("dtype")
        if (
            not isinstance(dtype, str)
            or dtype not in FORMATS
            or type(info.get("word_bits")) is not int
            or info["word_bits"] != FORMATS[dtype][0] * 8
            or info.get("encoding") != "little-endian-IEEE-words"
            or not isinstance(info.get("type"), str)
            or not isinstance(info.get("words"), str)
            or not info["words"]
            or len(info["words"]) % (FORMATS[dtype][0] * 2)
            or any(char not in "0123456789abcdef" for char in info["words"])
        ):
            raise ValueError(
                "floating literal reports have invalid typed storage words"
            )
        try:
            type_ = Parser(context(), info["type"]).parse_type()
        except Exception as error:
            raise ValueError(
                "floating literal reports have an invalid source type"
            ) from error
        shaped = isinstance(type_, (builtin.TensorType, builtin.VectorType))
        element = type_.element_type if shaped else type_
        shape = list(type_.get_shape()) if shaped else None
        data = bytes.fromhex(info["words"])
        count = len(data) // FORMATS[dtype][0]
        if (
            not isinstance(element, builtin.AnyFloat)
            or element.name != dtype
            or str(type_) != info["type"]
            or info.get("shape") != shape
            or shape is not None
            and any(type(value) is not int or value < 0 for value in info["shape"])
            or count not in (1, math.prod(shape) if shaped else 1)
            or type(info.get("nan_count")) is not int
            or info["nan_count"] != nan_count(data, element)
            or info["nan_count"] < 1
        ):
            raise ValueError(
                "floating literal reports differ from their original typed shape or NaN words"
            )
        candidates = set(report["source_ids"]) & set(source)
        if len(candidates) != 1 or seen & candidates:
            raise ValueError(
                "floating literal reports must account for each source constant once"
            )
        record = source[next(iter(candidates))]
        spelling = (
            "0x" + data[::-1].hex().upper() + " : " + dtype
            if not shaped
            else 'dense<"0x' + data.hex().upper() + '"> : ' + info["type"]
        )
        if (
            record.get("opcode") != "arith.constant"
            or record.get("output_types") != [info["type"]]
            or record.get("floating_literal_words") != info
            or record.get("properties", {}).get("value") != spelling
        ):
            raise ValueError(
                "floating literal reports differ from the complete source constant"
            )
        seen.update(candidates)
    if seen != set(source):
        raise ValueError(
            "floating literal reports do not cover every source word constant"
        )


def validate_trace(trace):
    reports = [
        report
        for report in trace.get("transformations", [])
        if report.get("kind") == "source-floating-literal-words"
    ]
    validate_reports(trace.get("source_operations", []), reports)
