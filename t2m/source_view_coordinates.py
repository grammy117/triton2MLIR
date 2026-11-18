"""Executable logical coordinates from original tensor transpose/slice SSA.

Reuse the runtime view product guards and LLVM a47bddcc TensorOps.cpp's
mixed offset/size/stride and shape-reification contracts. Records describe the
source path; only actual dimensions, operands and executable checks supply facts.
"""

from xdsl.dialects import arith, builtin, cf, scf, tensor
from xdsl.ir import Operation

from .executed_runtime_views import RuntimeViews

POLICY = "runtime-logical-coordinates-v1"
PROFILE = "t2m.runtime-source-tensor-views.v1"
BINDING_PROFILE = "t2m.execution-checked-call-views.v3"
SCHEMA = "t2m.runtime-source-coordinates.v1"


def shape(value):
    typ = value.type
    if (not isinstance(typ, builtin.TensorType) or not isinstance(typ.encoding, builtin.NoneAttr)
            or len(typ.get_shape()) > 8
            or any(size < 0 and size != builtin.DYNAMIC_INDEX for size in typ.get_shape())):
        raise ValueError("runtime source views require bounded unencoded tensor ranks")
    return typ.get_shape()


def static(op, name):
    value = {**op.attributes, **op.properties}.get(name)
    if not isinstance(value, builtin.DenseArrayBase) or value.elt_type != builtin.i64:
        raise ValueError("runtime source view lacks complete standard coordinate fields")
    return list(value.iter_values())


def slice_body(op):
    if isinstance(op, tensor.ExtractSliceOp):
        return op
    if not isinstance(op, scf.IfOp) or len(op.results) != 1:
        return None
    regions = (op.true_region, op.false_region)
    if any(len(region.blocks) != 1 or region.block.args for region in regions):
        raise ValueError("runtime source slice conditional needs two closed branches")
    positive, empty = [tuple(region.block.ops) for region in regions]
    if (len(positive) != 2 or len(empty) != 2
            or not isinstance(positive[0], tensor.ExtractSliceOp) or not isinstance(empty[0], tensor.EmptyOp)
            or not all(isinstance(branch[1], scf.YieldOp) for branch in (positive, empty))
            or tuple(positive[1].operands) != tuple(positive[0].results)
            or tuple(empty[1].operands) != tuple(empty[0].results)
            or positive[0].result.type != empty[0].tensor.type or positive[0].result.type != op.results[0].type
            or tuple(positive[0].sizes) != tuple(empty[0].operands)
            or any(op.is_ancestor(value.owner) for value in positive[0].operands)):
        raise ValueError("runtime source slice requires an exact extraction/empty refinement")
    return positive[0]


def retained_axes(extraction):
    """Match TensorOps.cpp getDroppedDims; only static unit axes may vanish."""
    sizes, target = static(extraction, "static_sizes"), shape(extraction.result)
    position, kept = len(target) - 1, []
    for axis in reversed(range(len(sizes))):
        size = sizes[axis]
        if size == 1 and (position < 0 or target[position] != 1):
            continue
        if position < 0 or size >= 0 and target[position] >= 0 and size != target[position]:
            raise ValueError("runtime source slice drops a nonunit axis or changes a retained size")
        kept.append(axis)
        position -= 1
    if position != -1:
        raise ValueError("runtime source slice does not supply every result axis")
    return list(reversed(kept))


def path(view, root, depth=0):
    target = shape(view)
    shape(root)
    if view.type.element_type != root.type.element_type or depth >= 32:
        raise ValueError("runtime source view changes storage or exceeds its source path budget")
    if view is root:
        return []
    op = view.owner
    if not isinstance(op, Operation) or len(op.results) != 1 or not op.operands:
        raise ValueError("runtime source view lacks an executable path to its allocation root")
    extraction = slice_body(op)
    parent = extraction.source if extraction is not None else op.operands[0]
    source = shape(parent)
    if extraction is None and len(source) != len(target):
        raise ValueError("runtime source view currently requires rank-preserving coordinates")
    if op.name == "linalg.transpose":
        permutation = static(op, "permutation")
        if (len(op.operands) != 2 or sorted(permutation) != list(range(len(source)))
                or len(op.regions) != 1 or len(op.regions[0].blocks) != 1):
            raise ValueError("runtime source view needs a complete word-preserving transpose")
        block = op.regions[0].block
        if (len(block.args) not in (1, 2) or len(tuple(block.ops)) != 1
                or block.last_op.name != "linalg.yield" or tuple(block.last_op.operands) != (block.args[0],)):
            raise ValueError("runtime source transpose changes element words")
    elif extraction is not None:
        for field in ("offsets", "sizes", "strides"):
            values = static(extraction, "static_" + field)
            if (len(values) != len(source) or sum(v == builtin.DYNAMIC_INDEX for v in values) != len(getattr(extraction, field))
                    or any(v < 0 and v != builtin.DYNAMIC_INDEX for v in values)):
                raise ValueError("runtime source slice has incomplete mixed coordinates")
        retained_axes(extraction)
    elif isinstance(op, tensor.CastOp):
        if len(op.operands) != 1 or any(a >= 0 and b >= 0 and a != b for a, b in zip(source, target, strict=True)):
            raise ValueError("runtime source cast changes logical dimensions")
    else:
        raise ValueError("runtime source view requires a standard transpose/slice path")
    return [*path(parent, root, depth + 1), op]


class Coordinates(RuntimeViews):
    def check(self, predicate, message):
        self.append(cf.AssertOp(predicate, "Triton executed call: source view " + message), "source-view-contract")

    def equal(self, left, right, message):
        self.check(self.append(arith.CmpiOp(left, right, "eq"), "source-view-equality"), message)

    def add(self, left, right):
        result = self.append(arith.AddiOp(left, right), "source-view-offset-add")
        self.check(self.append(arith.CmpiOp(result, self.constant(0), "sge"), "source-view-add-fits"),
                   "coordinate addition overflow")
        return result

    def mixed(self, op, field):
        operands = iter(getattr(op, field))
        return [next(operands) if n == builtin.DYNAMIC_INDEX else self.constant(n)
                for n in static(op, "static_" + field)]

    def geometry(self, view, root, chain):
        dimensions = self.extents(root)
        self.product(dimensions)
        strides = [self.product(dimensions[i + 1:]) for i in range(len(dimensions))]
        offset, steps = self.constant(0), []
        for op in chain:
            actual = self.extents(op.results[0])
            if op.name == "linalg.transpose":
                permutation = static(op, "permutation")
                expected = [dimensions[axis] for axis in permutation]
                strides = [strides[axis] for axis in permutation]
                steps.append({"op": op.name, "permutation": permutation, "result_type": str(op.results[0].type)})
            elif (extraction := slice_body(op)) is not None:
                origins, sizes, increments = [self.mixed(extraction, name) for name in ("offsets", "sizes", "strides")]
                self.product(origins)
                self.product(sizes)
                self.product(increments)
                if isinstance(op, scf.IfOp):
                    nonempty = self.append(arith.ConstantOp.from_int_and_width(1, builtin.i1), "source-view-nonempty")
                    for size in sizes:
                        positive = self.append(arith.CmpiOp(size, self.constant(0), "sgt"), "source-view-positive-size")
                        nonempty = self.append(arith.AndIOp(nonempty, positive), "source-view-nonempty")
                    self.equal(op.cond, nonempty, "empty-slice branch differs from actual dimensions")
                for origin, size, increment, bound in zip(origins, sizes, increments, dimensions, strict=True):
                    zero, one = self.constant(0), self.constant(1)
                    self.check(self.append(arith.CmpiOp(increment, zero, "sgt"), "source-view-positive-stride"),
                               "slice stride is not positive")
                    self.check(self.append(arith.CmpiOp(origin, bound, "sle"), "source-view-origin-bound"),
                               "slice origin exceeds its source dimension")
                    nonempty = self.append(arith.CmpiOp(size, zero, "sgt"), "source-view-nonempty-axis")
                    prior = self.append(arith.SubiOp(size, one), "source-view-last-element")
                    count = self.append(arith.SelectOp(nonempty, prior, zero), "source-view-safe-last-element")
                    last = self.add(origin, self.product([count, increment]))
                    valid = self.append(arith.CmpiOp(last, bound, "slt"), "source-view-last-in-bounds")
                    empty = self.append(arith.CmpiOp(size, zero, "eq"), "source-view-empty-axis")
                    self.check(self.append(arith.OrIOp(empty, valid), "source-view-slice-in-bounds"),
                               "slice exceeds its source dimension")
                for origin, stride in zip(origins, strides, strict=True):
                    offset = self.add(offset, self.product([origin, stride]))
                strides = [self.product([stride, increment]) for stride, increment in zip(strides, increments, strict=True)]
                kept = retained_axes(extraction)
                expected = [sizes[axis] for axis in kept]
                strides = [strides[axis] for axis in kept]
                steps.append({"op": op.name, "result_type": str(op.results[0].type),
                    **{name: static(extraction, "static_" + name) for name in ("offsets", "sizes", "strides")},
                    **({"retained_axes": kept} if len(kept) != len(sizes) else {})})
            else:
                expected = dimensions
                steps.append({"op": op.name, "result_type": str(op.results[0].type)})
            for observed, wanted in zip(actual, expected, strict=True):
                self.equal(observed, wanted, "operation result dimension differs from source coordinates")
            dimensions = actual
        return dimensions, strides, offset, {"schema": SCHEMA, "root_order": "logical-dense-row-major", "steps": steps}

    def scalar(self, value):
        from .executed_model_arguments import cast_index
        if isinstance(value.type, builtin.IndexType):
            return value
        if not isinstance(value.type, builtin.IntegerType) or value.type.bitwidth not in (8, 16, 32, 64):
            raise ValueError("runtime source coordinates require actual signed integer/index operands")
        return cast_index(self.append, value, builtin.IndexType())

    def bind(self, configurations, mapping, by_name, scalars, provenance):
        if not isinstance(configurations, dict) or not configurations or len(configurations) > 64:
            raise ValueError("runtime source views need complete bounded bindings")
        pointers = provenance.get("source_argument_binding", {}).get("bindings", [])
        records = {}
        for name in sorted(configurations):
            config = configurations[name]
            if (not isinstance(name, str) or name not in by_name or name in mapping.values()
                    or not isinstance(config, dict) or set(config) != {"root", "source_arg", "shape", "strides", "offset"}
                    or not isinstance(config["root"], str) or config["root"] not in mapping
                    or type(config["source_arg"]) is not int or not 0 <= config["source_arg"] < len(pointers)
                    or not isinstance(config["shape"], list) or not isinstance(config["strides"], list)
                    or len(config["shape"]) != len(config["strides"])):
                raise ValueError("runtime source view binding is incomplete or ambiguous")
            fields = [*config["shape"], *config["strides"], config["offset"]]
            if any(not isinstance(field, str) or field not in scalars for field in fields):
                raise ValueError("runtime source view fields need actual kernel scalar operands")
            if pointers[config["source_arg"]] != {"kind": "pointer", "allocation": config["root"], "offset": config["offset"]}:
                raise ValueError("runtime source view changes its executable allocation pointer binding")
            root_name = mapping[config["root"]]
            root, view = by_name[root_name], by_name[name]
            chain = path(view, root)
            if len(shape(view)) != len(config["shape"]):
                raise ValueError("runtime source view loses a logical coordinate axis")
            dimensions, strides, offset, description = self.geometry(view, root, chain)
            for actual, field in zip([*dimensions, *strides, offset], fields, strict=True):
                self.equal(actual, self.scalar(scalars[field]), "kernel argument disagrees with original view coordinates: " + field)
            records[name] = {"binding": config, "root_schema_operand": root_name, "root_type": str(root.type),
                             "view_type": str(view.type), "layout": description}
        return records
