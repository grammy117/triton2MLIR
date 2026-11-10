"""Portable exact expansions for shared's rounding/equal-width cast gaps.

Meta 17f0c234 FpToFpConverter rejects RTZ and emits an invalid extension
for equal-width formats. Keep its ordinary RTNE conversions; expand only
the missing forms through standard scalar casts inside elementwise generics.
The f64-to-16-bit routes jam an inexact f32 intermediate to odd before RTNE,
avoiding double rounding and unsupported direct narrow runtime helpers.
"""

from xdsl.dialects import arith, builtin, math, tensor
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError
from .standard_floats import FP8_TYPE_NAMES

WIDTHS = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}


def expand_float_casts(module, source_ids):
    from .shared_tensor import opcode

    plans = []
    for op in module.walk():
        name = opcode(op)
        if name not in ("tt.fp_to_fp", "arith.truncf"):
            continue
        ids = source_ids(op.location)

        def refuse(message):
            raise ConversionError(message, opcode=name, source_id=next(iter(ids), None))

        if len(op.operands) != 1 or len(op.results) != 1:
            refuse("floating conversion requires one operand and result")
        source, target = op.operands[0].type, op.results[0].type
        if name == "arith.truncf":
            src_element = (
                source.element_type
                if isinstance(source, builtin.TensorType)
                else source
            )
            dst_element = (
                target.element_type
                if isinstance(target, builtin.TensorType)
                else target
            )
            if str(src_element) != "f64" or str(dst_element) not in ("f16", "bf16"):
                continue
        shaped = isinstance(source, builtin.TensorType)
        if shaped != isinstance(target, builtin.TensorType):
            refuse("floating conversion must preserve its scalar/tensor shape")
        if shaped and (
            source.get_shape() != target.get_shape()
            or any(n < 0 for n in source.get_shape())
        ):
            refuse("floating source tiles require matching static shapes")
        sf, tf = (type_.element_type if shaped else type_ for type_ in (source, target))
        src, dst = str(sf), str(tf)
        if name == "arith.truncf":
            if src != "f64" or dst not in ("f16", "bf16"):
                continue
            if any(
                key not in ("fastmath",) and not key.startswith("prov.")
                for key in (*op.attributes, *op.properties)
            ):
                refuse("f64/16-bit conversion has an unqualified arithmetic property")
            flags = op.properties.get("fastmath", op.attributes.get("fastmath"))
            if flags is not None and (
                not isinstance(flags, arith.FastMathFlagsAttr) or flags.data
            ):
                refuse("f64/16-bit conversion requires neutral floating flags")
            mode = "rtne"
        else:
            if any(
                key not in ("op_name", "op_name__", "rounding")
                and not key.startswith("prov.")
                for key in (*op.attributes, *op.properties)
            ):
                refuse("TTIR floating conversion has an unqualified property")
            rounding = op.properties.get("rounding", op.attributes.get("rounding"))
            if rounding is None:
                mode = "rtne"
            elif (
                isinstance(rounding, builtin.IntegerAttr)
                and rounding.type == builtin.i32
                and rounding.value.data in (0, 1)
            ):
                mode = "rtz" if rounding.value.data == 0 else "rtne"
            else:
                refuse(
                    "floating conversion requires the pinned RTNE/RTZ rounding attribute"
                )
            if src in FP8_TYPE_NAMES and dst == src:
                pass  # Same-format casts preserve all original words.
            elif src in FP8_TYPE_NAMES and dst in WIDTHS:
                if mode == "rtne":
                    continue  # Qualified exact widening reuses upstream's converter.
            elif src not in WIDTHS or dst not in WIDTHS:
                if mode == "rtz":
                    refuse(
                        "FP8 narrowing/equal-width rounding needs its explicit numeric profile"
                    )
                continue
            else:
                if WIDTHS[src] > WIDTHS[dst] and rounding is None:
                    refuse("TTIR narrowing requires an explicit rounding mode")
                if (
                    mode == "rtne"
                    and WIDTHS[src] != WIDTHS[dst]
                    and not (src == "f64" and dst in ("f16", "bf16"))
                ):
                    continue  # Existing standard RTNE conversions remain upstream-owned.
        plans.append((op, sf, tf, mode, ids))

    reports = []
    for original, source, target, mode, ids in plans:
        generated = []

        def emit(op):
            op.location = original.location
            generated.append(op)
            return op.results[0]

        def toward_zero(value, src, dst):
            nearest = emit(arith.TruncFOp(value, dst))
            back = emit(arith.ExtFOp(nearest, src))
            actual, rounded = emit(math.AbsFOp(value)), emit(math.AbsFOp(back))
            overshot = emit(arith.CmpfOp(rounded, actual, "ogt"))
            word_type = builtin.IntegerType(WIDTHS[str(dst)])
            bits = emit(arith.BitcastOp(nearest, word_type))
            one = emit(arith.ConstantOp.from_int_and_width(1, word_type))
            lower = emit(arith.SubiOp(bits, one))
            selected = emit(arith.SelectOp(overshot, lower, bits))
            return emit(arith.BitcastOp(selected, dst))

        def narrow(value, src, dst, rounding):
            if str(src) == "f64" and str(dst) in ("f16", "bf16"):
                intermediate = toward_zero(value, src, builtin.f32)
                if rounding == "rtz":
                    return toward_zero(intermediate, builtin.f32, dst)
                back = emit(arith.ExtFOp(intermediate, src))
                inexact = emit(arith.CmpfOp(back, value, "une"))
                bits = emit(arith.BitcastOp(intermediate, builtin.i32))
                one = emit(arith.ConstantOp.from_int_and_width(1, builtin.i32))
                odd = emit(arith.OrIOp(bits, one))
                jammed = emit(arith.SelectOp(inexact, odd, bits))
                intermediate = emit(arith.BitcastOp(jammed, builtin.f32))
                return emit(arith.TruncFOp(intermediate, dst))
            return (
                toward_zero(value, src, dst)
                if rounding == "rtz"
                else emit(arith.TruncFOp(value, dst))
            )

        shaped = isinstance(original.operands[0].type, builtin.TensorType)
        block = Block(arg_types=[source, target]) if shaped else None
        value = block.args[0] if shaped else original.operands[0]
        if source == target:
            result = emit(arith.BitcastOp(value, target))
        elif str(source) in FP8_TYPE_NAMES or WIDTHS[str(source)] < WIDTHS[str(target)]:
            result = emit(arith.ExtFOp(value, target))
        elif WIDTHS[str(source)] == WIDTHS[str(target)]:
            value = emit(arith.ExtFOp(value, builtin.f32))
            result = narrow(value, builtin.f32, target, mode)
        else:
            result = narrow(value, source, target, mode)
        if shaped:
            terminal = YieldOp(result)
            terminal.location = original.location
            block.add_ops([*generated, terminal])
            output = original.results[0].type
            shape = output.get_shape()
            empty = tensor.EmptyOp([], output)
            expanded = GenericOp(
                [original.operands[0]],
                [empty.tensor],
                Region(block),
                [builtin.AffineMapAttr(AffineMap.identity(len(shape)))] * 2,
                [IteratorTypeAttr.parallel()] * len(shape),
                [output],
            )
            generated = [empty, expanded]
            result = expanded.results[0]
        else:
            expanded = result.owner
        expanded.attributes.update(
            {
                key: value
                for key, value in (
                    *original.attributes.items(),
                    *original.properties.items(),
                )
                if key.startswith("prov.")
            }
        )
        expanded.attributes.update(
            {
                "prov.composite_family": builtin.StringAttr("floating_conversion"),
                "prov.cast_source": builtin.StringAttr(str(source)),
                "prov.cast_target": builtin.StringAttr(str(target)),
                "prov.cast_rounding": builtin.StringAttr(mode),
            }
        )
        for op in generated:
            op.location = original.location
        Rewriter.replace_op(original, generated, [result])
        expansion = (
            "same-format-word-identity"
            if source == target
            else "standard-exact-extension"
            if str(source) in FP8_TYPE_NAMES
            or WIDTHS[str(source)] < WIDTHS[str(target)]
            else "exact-f32-format-bridge"
            if WIDTHS[str(source)] == WIDTHS[str(target)]
            else "f32-round-to-odd-bridge"
            if str(source) == "f64"
            and str(target) in ("f16", "bf16")
            and mode == "rtne"
            else "nearest-cast-and-toward-zero-word-correction"
        )
        reports.append(
            {
                "kind": "source-floating-conversion",
                "source_ids": ids,
                "source_type": str(source),
                "target_type": str(target),
                "rounding": mode,
                "expansion": expansion,
                "physical_mapping": "undecided",
                "computed_nan_payload": "original-word-preserved"
                if source == target
                else "unspecified",
            }
        )
    return reports
