"""Source-sized dense reshapes with executable, index-width-independent guards.

LLVM a47bddccec30255619bb8c37fa59700e661d4e66 TensorOps.cpp and tensor/ops.mlir
define dynamic collapse/expand and explicit output extents. ArithOps.td defines
index mului_extended. No shape predictor, sampled input or physical alias enters.
"""

import math

from xdsl.dialects import arith, builtin, cf, tensor


def admissible(source, target, *, input_view):
    if (
        not isinstance(source, builtin.TensorType)
        or not isinstance(target, builtin.TensorType)
        or not isinstance(source.encoding, builtin.NoneAttr)
        or not isinstance(target.encoding, builtin.NoneAttr)
        or source.element_type != target.element_type
    ):
        return False
    if input_view:
        return True
    shape = target.get_shape()
    return sum(n < 0 for n in shape) <= 1 and not (
        any(n < 0 for n in shape) and 0 in shape
    )


class RuntimeViews:
    def __init__(self, append):
        self.append = append

    def constant(self, number):
        return self.append(
            arith.ConstantOp.from_int_and_width(number, builtin.IndexType()),
            "runtime-view-constant",
        )

    def check(self, value, message):
        self.append(
            cf.AssertOp(value, "Triton executed call: runtime view " + message),
            "runtime-view-contract",
        )

    def product(self, dimensions):
        zero, one = self.constant(0), self.constant(1)
        empty = self.append(arith.CmpiOp(zero, one, "eq"), "runtime-view-empty")
        for value in dimensions:
            self.check(
                self.append(
                    arith.CmpiOp(value, zero, "sge"), "runtime-view-nonnegative"
                ),
                "negative extent",
            )
            is_zero = self.append(arith.CmpiOp(value, zero, "eq"), "runtime-view-zero")
            empty = self.append(arith.OrIOp(empty, is_zero), "runtime-view-empty")
        count = one
        for value in dimensions:
            factor = self.append(
                arith.SelectOp(empty, one, value), "runtime-view-factor"
            )
            multiply = arith.MulUIExtendedOp(count, factor)
            self.append(multiply, "runtime-view-product")
            count, high = multiply.results
            exact = self.append(
                arith.CmpiOp(high, zero, "eq"), "runtime-view-no-overflow"
            )
            positive = self.append(
                arith.CmpiOp(count, zero, "sge"), "runtime-view-product-nonnegative"
            )
            self.check(
                self.append(
                    arith.AndIOp(exact, positive), "runtime-view-product-valid"
                ),
                "element product overflow",
            )
        return self.append(
            arith.SelectOp(empty, zero, count), "runtime-view-element-count"
        )

    def extents(self, value):
        return [
            self.constant(size)
            if size >= 0
            else self.append(
                tensor.DimOp(value, self.constant(axis)),
                "runtime-view-actual-dimension",
            )
            for axis, size in enumerate(value.type.get_shape())
        ]

    def reshape(self, value, target, kernel_shape, dimensions, *, input_view):
        if not admissible(value.type, target, input_view=input_view):
            raise ValueError(
                "runtime dense view needs exact storage and unambiguous source dimensions"
            )
        shape = target.get_shape()
        if input_view and (
            len(shape) != len(kernel_shape)
            or any(isinstance(n, str) and n not in dimensions for n in kernel_shape)
        ):
            raise ValueError(
                "runtime dense view lacks its original kernel dimension operands"
            )
        count = self.product(self.extents(value))
        if input_view:
            sizes = [
                dimensions[n] if isinstance(n, str) else self.constant(n)
                for n in kernel_shape
            ]
        else:
            factor = self.product([self.constant(n) for n in shape if n >= 0])
            zero, one = self.constant(0), self.constant(1)
            safe_factor = self.append(
                arith.SelectOp(
                    self.append(
                        arith.CmpiOp(factor, zero, "eq"), "runtime-view-divisor-zero"
                    ),
                    one,
                    factor,
                ),
                "runtime-view-safe-divisor",
            )
            unknown = self.append(
                arith.DivUIOp(count, safe_factor), "runtime-view-inferred-dimension"
            )
            sizes = [self.constant(n) if n >= 0 else unknown for n in shape]
        total = self.product(sizes)
        self.check(
            self.append(arith.CmpiOp(count, total, "eq"), "runtime-view-equal-counts"),
            "element count mismatch",
        )
        element = target.element_type

        def flat_type(dimensions):
            return builtin.TensorType(
                element,
                [
                    builtin.DYNAMIC_INDEX
                    if any(n < 0 for n in dimensions)
                    else math.prod(dimensions)
                ],
            )

        def group(rank):
            return builtin.ArrayAttr(
                [
                    builtin.ArrayAttr(
                        builtin.IntegerAttr(i, builtin.i64) for i in range(rank)
                    )
                ]
                if rank
                else []
            )

        rank = len(value.type.get_shape())
        if rank > 1:
            value = self.append(
                tensor.CollapseShapeOp.build(
                    operands=[value],
                    result_types=[flat_type(value.type.get_shape())],
                    properties={"reassociation": group(rank)},
                ),
                "runtime-view-collapse",
            )
        elif rank == 0:
            scalar = self.append(
                tensor.ExtractOp.build(operands=[value, []], result_types=[element]),
                "runtime-view-scalar-read",
            )
            value = self.append(
                tensor.FromElementsOp(scalar), "runtime-view-scalar-pack"
            )
        flat = flat_type(shape)
        if value.type != flat:
            value = self.append(tensor.CastOp(value, flat), "runtime-view-flat-cast")
        if len(shape) > 1:
            value = self.append(
                tensor.ExpandShapeOp(
                    value,
                    [
                        size
                        for size, declared in zip(sizes, shape, strict=True)
                        if declared < 0
                    ],
                    group(len(shape)),
                    shape,
                    target,
                ),
                "runtime-view-expand",
            )
        elif not shape:
            value = self.append(
                tensor.CollapseShapeOp.build(
                    operands=[value],
                    result_types=[target],
                    properties={"reassociation": group(0)},
                ),
                "runtime-view-scalar-collapse",
            )
        return value
