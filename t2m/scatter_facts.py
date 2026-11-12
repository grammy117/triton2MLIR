"""Prove dense tile addresses behind upstream's unstructured scatter boundary.

Meta 17f0c234, PtrAnalysis::mulState and MaskState::parseAnd/parseCmp are
the reference: recover pointer axes independently of arbitrary mask payloads,
and retain only necessary rectangular bounds from conjunctions. At this stage
upstream has converted the index/mask tensors to linalg, so projected maps and
scalar SSA are inspected structurally. This analysis emits no computation and
does not evaluate lanes/programs or treat example indices as facts.
"""

from dataclasses import dataclass, field
from math import prod

from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerAttr, TensorType
from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr

from .diagnostics import ConversionError
from .shared_indexing import Linear
from .symbolic_layout import Polynomial as P, dense_strides, expression, recover_axes


def lane(axis, size):
    return P.make([((("d", axis, 0, size - 1),), 1)])


@dataclass
class Node:
    value: object
    coordinates: tuple = ()
    bindings: dict = field(default_factory=dict)
    indices: tuple = ()


class TileFacts:
    def __init__(self, facts, grid):
        self.facts, self.grid = facts, grid
        self.steps, self.cache = 0, {}
        self.environments = []

    def tick(self):
        self.steps += 1
        if self.steps > 1024:
            raise ConversionError("scatter tile proof exceeds its bounded SSA analysis budget")

    def affine(self, value, dimensions):
        if isinstance(value, AffineDimExpr):
            return dimensions[value.position]
        if isinstance(value, AffineConstantExpr):
            return P.constant(value.value)
        if isinstance(value, AffineBinaryOpExpr) and value.kind in (AffineBinaryOpKind.Add, AffineBinaryOpKind.Mul):
            a, b = self.affine(value.lhs, dimensions), self.affine(value.rhs, dimensions)
            return a + b if value.kind == AffineBinaryOpKind.Add else a * b
        raise ConversionError("scatter index needs projected affine linalg maps")

    def normalize(self, node):
        while True:
            self.tick()
            value, op = node.value, node.value.owner
            if value in node.bindings:
                node = node.bindings[value]
                continue
            if not isinstance(value.type, TensorType):
                return node
            name = getattr(op, "name", "")
            if name == "tensor.from_elements" and value.type.get_shape() == (1,) and len(op.operands) == 1:
                node = Node(op.operands[0], bindings=node.bindings, indices=node.indices)
                continue
            if name == "linalg.fill":
                node = Node(op.inputs[0], bindings=node.bindings, indices=node.indices)
                continue
            if name == "tensor.expand_shape":
                source = op.operands[0]
                before, after = source.type.get_shape(), value.type.get_shape()
                groups = tuple(tuple(n.value.data for n in group) for group in op.reassociation)
                if (len(groups) != len(before) or len(node.coordinates) != len(after)
                        or tuple(i for group in groups for i in group) != tuple(range(len(after)))
                        or any(n <= 0 for n in (*before, *after))
                        or any(not group or prod(after[i] for i in group) != n for group, n in zip(groups, before, strict=True))):
                    raise ConversionError("scatter expand_shape needs an exact static reassociation")
                coordinates = []
                for group in groups:
                    coordinate = P.constant(0)
                    for position, axis in enumerate(group):
                        coordinate = coordinate + node.coordinates[axis].scale(prod(after[i] for i in group[position + 1:]))
                    coordinates.append(coordinate)
                node = Node(source, tuple(coordinates), node.bindings, node.indices)
                continue
            if name == "linalg.transpose":
                source = op.operands[0]
                permutation = tuple(op.permutation.get_values())
                if (sorted(permutation) != list(range(len(node.coordinates)))
                        or tuple(source.type.get_shape()[i] for i in permutation) != value.type.get_shape()):
                    raise ConversionError("scatter transpose needs a typed axis permutation")
                coordinates = [None] * len(permutation)
                for axis, position in enumerate(permutation):
                    coordinates[position] = node.coordinates[axis]
                node = Node(source, tuple(coordinates), node.bindings, node.indices)
                continue
            if name != "linalg.generic":
                return node
            maps = tuple(a.data for a in op.indexing_maps)
            if (len(op.results) != 1 or value is not op.results[0]
                    or any(a.data.value != "parallel" for a in op.iterator_types)
                    or len(maps) != len(op.operands)):
                raise ConversionError("scatter indices require a pure pointwise linalg value region")
            output = maps[-1]
            order = [d.position if isinstance(d, AffineDimExpr) else -1 for d in output.results]
            if output.num_symbols or sorted(order) != list(range(output.num_dims)) or len(order) != len(node.coordinates):
                raise ConversionError("scatter index output map is not a projected permutation")
            dimensions = [None] * len(order)
            for axis, position in enumerate(order):
                dimensions[position] = node.coordinates[axis]
            environment = dict(node.bindings)
            self.environments.append(environment)
            for arg, operand, map_ in zip(op.body.block.args, op.operands, maps, strict=True):
                if map_.num_symbols:
                    raise ConversionError("scatter index maps cannot contain unresolved symbols")
                coordinates = tuple(self.affine(d, dimensions) for d in map_.results)
                environment[arg] = Node(operand, coordinates, node.bindings, node.indices)
            terminator = op.body.block.last_op
            if terminator.name != "linalg.yield" or len(terminator.operands) != 1:
                raise ConversionError("scatter index region needs one explicit yielded value")
            node = Node(terminator.operands[0], bindings=environment, indices=tuple(dimensions))

    def number(self, node):
        node = self.normalize(node)
        value, op = node.value, node.value.owner
        key = value, node.coordinates, node.indices, id(node.bindings) if node.bindings else 0
        if key in self.cache:
            if self.cache[key] is None:
                raise ConversionError("scatter index has cyclic/unproved state")
            return self.cache[key]
        self.cache[key] = None
        name = getattr(op, "name", "")
        def operand(index):
            return self.number(Node(op.operands[index], bindings=node.bindings, indices=node.indices))
        if value in self.facts:
            result = P.affine(self.facts[value])
        elif name == "arith.constant":
            data = op.value
            if isinstance(data, IntegerAttr):
                result = P.constant(data.value.data)
            elif isinstance(data, DenseIntOrFPElementsAttr) and len(data) <= 4096 and data.is_splat():
                number = data.get_values()[0]
                if type(number) is not int:
                    raise ConversionError("scatter integer indices cannot come from floating constants")
                result = P.constant(number)
            else:
                raise ConversionError("scatter indices need scalar/splat integer constants")
        elif name == "linalg.index":
            result = node.indices[op.dim.value.data]
        elif name in ("arith.index_cast", "arith.extsi", "arith.extui", "arith.trunci"):
            result = operand(0)
            if name == "arith.extui" and result.bounds(self.grid)[0] < 0:
                raise ConversionError("scatter unsigned index extension needs a nonnegative source")
        elif name in ("arith.addi", "arith.subi", "arith.muli"):
            a, b = operand(0), operand(1)
            result = a * b if name == "arith.muli" else a + b.scale(-1 if name == "arith.subi" else 1)
        elif isinstance(value.type, TensorType):
            raise ConversionError("scatter indices need affine tile values; indirect conflicts remain unproved")
        else:
            result = expression(value, self.facts, self.grid)
        type_ = value.type.element_type if isinstance(value.type, TensorType) else value.type
        width = getattr(getattr(type_, "width", None), "data", 64)
        low, high = result.bounds(self.grid)
        if not -(1 << (width - 1)) <= low <= high < 1 << (width - 1):
            raise ConversionError("scatter index arithmetic may overflow before logical addressing")
        self.cache[key] = result
        return result

    def necessary_bounds(self, node):
        """Return proved expressions < 0 whenever this lane's mask is true."""
        try:
            node = self.normalize(node)
            op = node.value.owner
            name = getattr(op, "name", "")
            def operand(i):
                return Node(op.operands[i], bindings=node.bindings, indices=node.indices)
            if name == "arith.andi":
                return tuple(dict.fromkeys((*self.necessary_bounds(operand(0)), *self.necessary_bounds(operand(1)))))
            if name == "arith.ori":
                left, right = self.necessary_bounds(operand(0)), self.necessary_bounds(operand(1))
                return tuple(v for v in left if v in right)
            if name == "arith.cmpi":
                a, b = self.number(operand(0)), self.number(operand(1))
                predicate = op.predicate.value.data
                if predicate >= 6:
                    if min(a.bounds(self.grid)[0], b.bounds(self.grid)[0]) < 0:
                        return ()
                    predicate -= 4
                if predicate == 0:
                    return a + b.scale(-1) + P.constant(-1), b + a.scale(-1) + P.constant(-1)
                if predicate in (4, 5):
                    a, b, predicate = b, a, predicate - 2
                if predicate in (2, 3):
                    return (a + b.scale(-1) + P.constant(-int(predicate == 3)),)
        except ConversionError:
            # An arbitrary payload predicate adds no rectangular facts. Its
            # complete executable SSA is still retained by the upstream path.
            pass
        return ()


def scatter_tile(buffer, abi, indices, mask, facts):
    if not isinstance(indices.type, TensorType):
        raise ConversionError("scalar scatter needs a scalar-pointer/effect adapter")
    sizes = indices.type.get_shape()
    if len(sizes) != len(buffer.shape) or any(n <= 0 for n in sizes):
        raise ConversionError("scatter needs matching static tile and logical tensor ranks")
    coordinates = tuple(lane(i, n) for i, n in enumerate(sizes))
    analysis = TileFacts(facts, abi.grid_bounds)
    value = analysis.number(Node(indices, coordinates))
    base, strides = [], [[] for _ in sizes]
    for atoms, coefficient in value.terms:
        local = [a for a in atoms if a[0] == "d"]
        if not local:
            base.append((atoms, coefficient))
        elif len(local) == 1:
            strides[local[0][1]].append((tuple(a for a in atoms if a[0] != "d"), coefficient))
        else:
            raise ConversionError("scatter lane mapping is nonlinear; injectivity is not proved")
    strides = tuple(P.make(terms) for terms in strides)
    _, axes = recover_axes(abi, buffer, sizes, strides, [P.constant(0)] * len(sizes), facts)
    remaining, forms = P.make(base), []
    for stride in dense_strides(abi, buffer):
        coordinate = Linear()
        for term in remaining.terms:
            try:
                part = P.make([term]).divide_stride(stride)
            except ConversionError:
                continue
            coordinate = coordinate + part
        remaining = remaining + (P.affine(coordinate) * stride).scale(-1)
        forms.append(coordinate)
    if remaining.terms:
        raise ConversionError("scatter base offset does not factor into dense logical coordinates")
    conditions = analysis.necessary_bounds(Node(mask, coordinates)) if mask is not None else ()
    for source_axis, destination_axis in enumerate(axes):
        form, size = forms[destination_axis], sizes[source_axis]
        if form.bounds(abi.grid_bounds)[0] < 0:
            raise ConversionError("scatter tile starts below its declared logical tensor")
        dimension = abi.dimension(buffer.shape[destination_axis])
        dimension = P.affine(dimension) if isinstance(dimension, Linear) else P.constant(dimension)
        full = P.affine(form) + P.constant(size) + dimension.scale(-1)
        if full.bounds(abi.grid_bounds)[1] <= 0:
            continue
        last = P.affine(form) + coordinates[source_axis] + dimension.scale(-1)
        if not any((last + condition.scale(-1)).bounds(abi.grid_bounds)[1] <= 0 for condition in conditions):
            raise ConversionError("scatter mask does not prove active lanes inside the declared logical axes")
    return tuple(forms), sizes, axes
