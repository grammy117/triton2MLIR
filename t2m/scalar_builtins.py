# Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Compose shared's scalar arithmetic mappings before lifting scalar regions.

Adapted from FpToFpConverter, ClampConverter, PreciseSqrtConverter,
PreciseDivConverter and the external math table in Meta/shared
17f0c234ce474bee334af36e351c62eca7852836 ConversionPatterns.hpp. The archived
Microsoft converter uses the same patterns. Tensor arithmetic stays with the
shared pipeline; exact rounding/high-product expansions run before this pass.
No operation is hoisted across source control or granted purity by its name.
"""

from xdsl.dialects import arith, builtin, math
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError
from .external_math import contract as external_contract


def expand_scalar_builtins(module, source_ids):
    from .shared_tensor import opcode

    supported = {
        "tt.precise_divf": (2, ()),
        "tt.precise_sqrt": (1, ()),
        "tt.clampf": (3, ("propagateNan",)),
        "tt.fp_to_fp": (1, ("rounding",)),
        "tt.bitcast": (1, ()),
        "tt.extern_elementwise": (None, ("pure", "symbol", "libname", "libpath")),
    }
    math_ops = {operation.name: operation for operation in math.Math.operations}
    plans = []
    for original in module.walk():
        name = opcode(original)
        if name not in supported:
            continue
        types = [value.type for value in (*original.operands, *original.results)]
        if any(isinstance(type_, builtin.TensorType) for type_ in types):
            continue
        # Pointer bitcasts belong to the existing address/effect analysis.
        if name == "tt.bitcast" and not all(
            isinstance(type_, (builtin.IntegerType, builtin.AnyFloat))
            for type_ in types
        ):
            continue
        ids = source_ids(original.location)

        def refuse(message, *, name=name, ids=ids):
            raise ConversionError(message, opcode=name, source_id=next(iter(ids), None))

        arity, properties = supported[name]
        if (
            len(original.results) != 1
            or arity is not None
            and len(original.operands) != arity
            or original.regions
            or original.successors
        ):
            refuse("scalar builtin needs its complete scalar prototype without regions")
        allowed = {"op_name", "op_name__", *properties}
        if any(
            key not in allowed and not key.startswith("prov.")
            for key in (*original.properties, *original.attributes)
        ):
            refuse("scalar builtin has an unqualified source property")
        attrs = {**original.attributes, **original.properties}
        if name == "tt.extern_elementwise":
            contract = external_contract(original, next(iter(ids), "scalar-external"))
            detail = contract
        elif name == "tt.bitcast":
            if types[0].bitwidth != types[-1].bitwidth:
                refuse("scalar bitcast must retain the original numeric word width")
            detail = {}
        else:
            if not all(isinstance(type_, builtin.AnyFloat) for type_ in types):
                refuse("scalar floating builtin requires floating operands and result")
            if name == "tt.fp_to_fp":
                rounding = attrs.get("rounding")
                if rounding is not None and (
                    not isinstance(rounding, builtin.IntegerAttr)
                    or rounding.type != builtin.i32
                    or rounding.value.data != 1
                ):
                    refuse(
                        "scalar cast needs its exact rounding expansion before region lifting"
                    )
                if types[0].bitwidth == types[-1].bitwidth:
                    refuse("equal-width scalar casts need the exact format expansion")
                if types[0].bitwidth > types[-1].bitwidth and rounding is None:
                    refuse("scalar narrowing needs an explicit source rounding mode")
                detail = {"rounding": "rtne"}
            else:
                if any(type_ != types[-1] for type_ in types):
                    refuse("scalar builtin operand and result formats must match")
                detail = {}
                if name == "tt.clampf":
                    policy = attrs.get("propagateNan")
                    if (
                        not isinstance(policy, builtin.IntegerAttr)
                        or policy.type != builtin.i32
                        or policy.value.data not in (0, 65535)
                    ):
                        refuse("scalar clamp requires the pinned NONE/ALL NaN policy")
                    detail = {"propagate_nan": "all" if policy.value.data else "none"}
        plans.append((original, name, ids, detail))

    reports = []
    # Validate the complete candidate set before replacing even a dead scalar.
    for original, name, ids, detail in plans:
        operands, target = list(original.operands), original.results[0].type
        if name == "tt.precise_divf":
            emitted = [arith.DivfOp(*operands)]
        elif name == "tt.precise_sqrt":
            emitted = [math.SqrtOp(*operands)]
        elif name == "tt.clampf":
            all_nan = detail["propagate_nan"] == "all"
            lower = (arith.MaximumfOp if all_nan else arith.MaxnumfOp)(*operands[:2])
            upper = (arith.MinimumfOp if all_nan else arith.MinnumfOp)(
                lower, operands[2]
            )
            emitted = [lower, upper]
        elif name == "tt.fp_to_fp":
            cast = (
                arith.ExtFOp
                if operands[0].type.bitwidth < target.bitwidth
                else arith.TruncFOp
            )
            emitted = [cast(operands[0], target)]
        elif name == "tt.bitcast":
            emitted = [arith.BitcastOp(operands[0], target)]
        else:
            emitted = [math_ops[detail["target"]](*operands)]
        for child in emitted:
            child.location = original.location
            child.attributes.update(
                {
                    key: value
                    for key, value in (
                        *original.attributes.items(),
                        *original.properties.items(),
                    )
                    if key.startswith("prov.")
                }
            )
        Rewriter.replace_op(original, emitted)
        reports.append(
            {
                "kind": "source-scalar-builtin",
                "source_ids": ids,
                "source_opcode": name,
                "expansion": [child.name for child in emitted],
                "control_position": "unchanged",
                "physical_mapping": "undecided",
                **detail,
            }
        )
    return reports
