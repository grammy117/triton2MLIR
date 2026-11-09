"""Functionalize triton-shared's tensor stage, without its memory/runtime ABI.

Upstream owns TTIR arithmetic and region decomposition. This adapter owns the
declared logical tensor ABI, ordered memory versions, bounds, and provenance.
The profile supports affine logical grids, bounded source loops and structured
tensor descriptors. Unsupported memory/control-flow forms fail closed.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math as host_math
from pathlib import Path

from xdsl.dialects import arith, builtin, cf, func, math, scf, tensor
from xdsl.dialects.builtin import (ArrayAttr, DenseArrayBase, DYNAMIC_INDEX, IndexType,
                                  IntegerAttr, ModuleOp, NameLoc, StringAttr, TensorType)
from xdsl.dialects.linalg.ops import GenericOp, TransposeOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.parser import Parser

from .diagnostics import ConversionError
from .dynamic_abi import GridExtent, shape_contract, validate_dimensions
from .control_flow import value_only_region
from .frontend import DESCRIPTOR_OPS, expand_tensor_pointers
from .ir import context, text
from .initialization import DefiniteWrites
from .loop_domains import bounded_loop, require_signed_loop
from .scans import expand_scans
from .scalar_abi import RuntimeScalar, guard_scalar
from .shape_lowering import bind_dimensions, emit as shape_emit, launch_sizes, product as shape_product, program_id
from .shape_lowering import affine_index
from .symbolic_layout import DenseAxes, recover_axes
from .shared_gather import gather, structured_gather
from .scatter_facts import scatter_tile
from .shared_indexing import (Linear, check_source_addresses, independent_tile, linear,
                              logical_coordinates, seed_recurrences)
from .standard_ops import ForallOp, InParallelOp, ParallelInsertSliceOp
from .tail_masks import TailExtent, describe, maximum, recover_extent
from .upstream import inspect_optimizer, lower_with_shared
from .views import axis_permutation, inverse_permutation

DTYPES = {name: getattr(builtin, name) for name in
          ("f16", "bf16", "f32", "f64", "i1", "i8", "i16", "i32", "i64")}


class SharedParser(Parser):
    # xDSL 0.68 parses dense<0xFF800000> : tensor<...xf32> as the
    # numeric integer 4286578688. MLIR uses IEEE *bits* here. Adapt the
    # typed token conversion, not printed IR lines or operation semantics.
    class _TensorLiteralElement(Parser._TensorLiteralElement):
        def to_type(self, parser, type_):
            if isinstance(type_, builtin.AnyFloat) and self.span.text.lower().startswith("0x"):
                import math as host_math
                raw = int(self.span.text, 16).to_bytes(type_.compile_time_size, "little")
                number = next(type_.iter_unpack(raw))
                if host_math.isnan(number):
                    parser.raise_error("NaN constant payload preservation is not qualified", at_position=self.span)
                return number
            return super().to_type(parser, type_)


def opcode(op):
    return op.op_name.data if op.name == "builtin.unregistered" else op.name


def parse_shared(source):
    ctx = context()
    ctx.allow_unregistered = True  # Only intake; output is checked separately.
    ctx.load_dialect(math.Math)
    ctx.load_dialect(cf.Cf)
    from .standard_memref import memref_dialect
    from .standard_bufferization import bufferization_dialect
    ctx.load_dialect(memref_dialect())
    ctx.load_dialect(bufferization_dialect())
    return SharedParser(ctx, source).parse_module()


@dataclass(frozen=True)
class TensorBuffer:
    name: str
    shape: tuple[int | str, ...]
    effect: str = "read"
    dtype: str = "f32"

    def __post_init__(self):
        if not self.name.isidentifier() or not self.shape or any(
                not (type(n) is int and n > 0 or isinstance(n, str) and n.isidentifier()) for n in self.shape):
            raise ValueError("tensor buffers require a name and positive static or named runtime dimensions")
        if self.dtype not in DTYPES or self.effect not in ("read", "write", "readwrite"):
            raise ValueError("tensor buffers require a supported dtype and read/write/readwrite effect")

    @property
    def type(self):
        return TensorType(DTYPES[self.dtype], [DYNAMIC_INDEX if isinstance(n, str) else n for n in self.shape])

    @property
    def strides(self):
        if any(isinstance(n, str) for n in self.shape):
            raise ConversionError("runtime dense layouts require symbolic stride analysis")
        strides, product = [], 1
        for extent in reversed(self.shape):
            strides.insert(0, product)
            product *= extent
        return tuple(strides)


@dataclass(frozen=True)
class TensorABI:
    entry: str
    buffers: tuple[TensorBuffer, ...]
    scalar_constants: tuple[int, ...] = ()
    grid: tuple[int | GridExtent, ...] = (1,)
    intent: str = ""
    runtime_scalars: tuple[RuntimeScalar, ...] = ()

    def __post_init__(self):
        if not self.entry.isidentifier() or not self.buffers:
            raise ValueError("tensor ABI requires an entry and buffers")
        names = [b.name for b in (*self.buffers, *self.runtime_scalars)]
        if len(set(names)) != len(names) or not self.outputs:
            raise ValueError("tensor ABI requires unique names and at least one writable result")
        validate_dimensions(self)
        if not 1 <= len(self.grid) <= 3 or any(type(n) is not int or not 0 < n < 2**31 for n in self.grid_bounds) or host_math.prod(self.grid_bounds) >= 2**63:
            raise ValueError("logical grid requires 1..3 positive i32 dimensions and an i64-size domain")
        if any(type(n) is not int for n in self.scalar_constants):
            raise ValueError("specialized scalar arguments must be integers")
        positions = [s.source_arg for s in self.runtime_scalars]
        count = len(self.buffers) + len(self.scalar_constants) + len(positions)
        if len(set(positions)) != len(positions) or any(p >= count for p in positions):
            raise ValueError("runtime scalar source positions must be unique and inside the complete source signature")

    @property
    def dynamic(self):
        return any(isinstance(g, GridExtent) for g in self.grid) or any(isinstance(n, str) for b in self.buffers for n in b.shape)

    @property
    def grid_bounds(self):
        scalars = {s.name: s for s in self.runtime_scalars}
        return tuple((scalars[g.dimension].bounds[1] + g.divisor - 1) // g.divisor
                     if isinstance(g, GridExtent) else g for g in self.grid)

    def dimension(self, value):
        if not isinstance(value, str):
            return value
        scalar = next(s for s in self.runtime_scalars if s.name == value)
        return Linear(parameters=((value, *scalar.bounds, 1),))

    def runtime_shapes(self):
        return shape_contract(self)

    def as_dict(self):
        schema = "t2m.shared-tensor-abi.v2" if any(b.effect == "write" for b in self.buffers) else "t2m.shared-tensor-abi.v1"
        fields = asdict(self)
        if self.runtime_scalars:
            schema = "t2m.shared-tensor-abi.v4" if any(s.bounds is not None for s in self.runtime_scalars) else "t2m.shared-tensor-abi.v3"
            fields["runtime_scalars"] = tuple(s.as_dict() for s in self.runtime_scalars)
        else:
            fields.pop("runtime_scalars")
        if self.dynamic:
            schema = "t2m.shared-tensor-abi.v5"
            fields["grid"] = tuple(g.as_dict() if isinstance(g, GridExtent) else g for g in self.grid)
        return {"schema": schema, **fields}

    @property
    def inputs(self):
        return (*[b for b in self.buffers if b.effect != "write"], *self.runtime_scalars)

    @property
    def source_bindings(self):
        """Pointers in declaration order, legacy constants in remaining slots."""
        count = len(self.buffers) + len(self.scalar_constants) + len(self.runtime_scalars)
        runtime = {s.source_arg: ("runtime", s) for s in self.runtime_scalars}
        remaining = iter([*(('buffer', b) for b in self.buffers), *(('constant', n) for n in self.scalar_constants)])
        return tuple(runtime[i] if i in runtime else next(remaining) for i in range(count))

    @property
    def outputs(self):
        return tuple(b for b in self.buffers if b.effect != "read")

    @classmethod
    def from_dict(cls, value):
        fields = dict(value)
        schema = fields.pop("schema", "t2m.shared-tensor-abi.v1")
        if schema not in tuple("t2m.shared-tensor-abi.v" + str(i) for i in range(1, 6)):
            raise ValueError("unknown shared tensor ABI schema")
        if schema not in ("t2m.shared-tensor-abi.v3", "t2m.shared-tensor-abi.v4", "t2m.shared-tensor-abi.v5") and fields.get("runtime_scalars"):
            raise ValueError("runtime scalar arguments require ABI v3")
        if schema in ("t2m.shared-tensor-abi.v3", "t2m.shared-tensor-abi.v4", "t2m.shared-tensor-abi.v5") and not fields.get("runtime_scalars"):
            raise ValueError("ABI v3 requires declared runtime scalar arguments")
        fields["runtime_scalars"] = tuple(RuntimeScalar(**{**s, **({"bounds": tuple(s["bounds"])} if "bounds" in s else {})}) for s in fields.get("runtime_scalars", ()))
        if schema != "t2m.shared-tensor-abi.v5" and (schema == "t2m.shared-tensor-abi.v4") != any(s.bounds is not None for s in fields["runtime_scalars"]):
            raise ValueError("guarded runtime scalar bounds require explicit ABI v4")
        fields["buffers"] = tuple(TensorBuffer(**{**b, "shape": tuple(b["shape"])}) for b in fields["buffers"])
        for name in ("grid", "scalar_constants"):
            if name in fields:
                fields[name] = tuple(fields[name])
        fields["grid"] = tuple(GridExtent(**g) if isinstance(g, dict) else g for g in fields.get("grid", (1,)))
        result = cls(**fields)
        if result.dynamic != (schema == "t2m.shared-tensor-abi.v5"):
            raise ValueError("runtime tensor dimensions and grids require explicit ABI v5")
        return result


def _array(op, name):
    value = op.properties.get(name, op.attributes.get(name))
    if not isinstance(value, DenseArrayBase):
        raise ConversionError(f"expected structured {name} array", opcode=opcode(op))
    return tuple(value.get_values())


def _pointer_type(type_):
    if isinstance(type_, TensorType):
        type_ = type_.get_element_type()
    return isinstance(type_, builtin.UnregisteredAttr) and type_.attr_name.data == "tt.ptr"


def _unused_pointer_initialization(use):
    op = use.operation
    return (op.name == "scf.for" and use.index >= 3
            and _pointer_type(op.body.block.args[use.index - 2].type)
            and not op.body.block.args[use.index - 2].uses)


def _integer(value):
    op = value.owner
    if not hasattr(op, "name"):
        raise ConversionError("memory descriptor depends on an unspecialized block argument")
    if op.name == "arith.constant" and isinstance(op.value, IntegerAttr):
        return op.value.value.data
    if op.name in ("arith.index_cast", "arith.extsi", "arith.extui", "arith.trunci"):
        number = _integer(op.operands[0])
        target = value.type
        if hasattr(target, "width") and not -(1 << (target.width.data - 1)) <= number < (1 << (target.width.data - 1)):
            raise ConversionError("specialized index cast may wrap")
        return number
    if op.name in ("arith.addi", "arith.subi", "arith.muli", "arith.minsi", "arith.maxsi"):
        a, b = map(_integer, op.operands)
        number = {"arith.addi": lambda: a + b, "arith.subi": lambda: a - b,
                  "arith.muli": lambda: a * b, "arith.minsi": lambda: min(a, b),
                  "arith.maxsi": lambda: max(a, b)}[op.name]()
        if hasattr(value.type, "width") and not -(1 << (value.type.width.data - 1)) <= number < (1 << (value.type.width.data - 1)):
            raise ConversionError("specialized address arithmetic may overflow")
        return number
    raise ConversionError("structured memory indices are not statically resolved", opcode=op.name)


def _source_ids(location):
    if isinstance(location, NameLoc):
        name = location.desc.data
        return [name] if name.startswith(("ttir.op.", "normalized.ttir.op.")) else _source_ids(location.location)
    if isinstance(location, builtin.FusedLoc):
        return list(dict.fromkeys(s for loc in location.locations for s in _source_ids(loc)))
    if isinstance(location, builtin.CallSiteLoc):
        return list(dict.fromkeys(_source_ids(location.callee) + _source_ids(location.caller)))
    return []


def _preflight(module, abi, *, id_prefix="ttir.op."):
    functions = [op for op in module.ops if opcode(op) == "tt.func"]
    entries = [op for op in functions if op.properties["sym_name"].data == abi.entry]
    if len(entries) != 1 or len(list(module.ops)) != len(functions):
        raise ConversionError("expected the declared TTIR entry and optional private helpers")
    function = entries[0]
    if function.properties["function_type"].outputs:
        raise ConversionError("kernel entry return values need an explicit ABI binding")
    if any(op is not function and op.properties.get("sym_visibility") != StringAttr("private") for op in functions):
        raise ConversionError("additional public entry functions are unsupported")
    if len(function.regions[0].blocks) != 1:
        raise ConversionError("multiple entry blocks need an explicit CFG adapter")
    args = function.regions[0].block.args
    if len(args) != len(abi.source_bindings):
        raise ConversionError("source arguments differ from the declared pointer/runtime/specialized scalar bindings")
    for arg, (kind, value) in zip(args, abi.source_bindings, strict=True):
        if kind == "buffer" and str(arg.type) != f"!tt.ptr<{value.dtype}>":
            raise ConversionError(f"pointer type differs from tensor ABI for {value.name}")
        if kind == "runtime" and arg.type != value.type:
            raise ConversionError(f"runtime scalar type differs from source for {value.name}")
    records, values = [], {}
    for op in module.walk():
        for region in op.regions:
            for block in region.blocks:
                for value in block.args:
                    values[value] = f"ttir.value.{len(values):04d}"
        for value in op.results:
            values[value] = f"ttir.value.{len(values):04d}"
    for op in module.walk():
        if op is module:
            continue
        source_id, name = f"{id_prefix}{len(records):04d}", opcode(op)
        records.append({"id": source_id, "opcode": name, "location": str(op.location),
                        "inputs": [values[v] for v in op.operands], "outputs": [values[v] for v in op.results],
                        "input_types": [str(v.type) for v in op.operands],
                        "output_types": [str(v.type) for v in op.results],
                        "properties": {k: str(v) for k, v in op.properties.items()},
                        "attributes": {k: str(v) for k, v in op.attributes.items() if k != "op_name"}})
        op.location = NameLoc(StringAttr(source_id), op.location)
        if name in ("tt.atomic_rmw", "tt.atomic_cas", "tt.inline_asm", "tt.extern_elementwise",
                    "tt.print", "tt.assert") or name.startswith(("ttg.", "ttng.", "gpu.", "nvvm.", "rocdl.")):
            raise ConversionError("operation needs a separately qualified effect or numeric profile",
                                  source_id=source_id, opcode=name)
        if name == "tt.load":
            volatile = op.properties.get("isVolatile", op.attributes.get("isVolatile"))
            if volatile is not None and (not isinstance(volatile, IntegerAttr) or volatile.value.data):
                raise ConversionError("volatile observations cannot become stable tensor reads", source_id=source_id, opcode=name)
        if name == "arith.constant":
            value = op.properties.get("value")
            if isinstance(value, builtin.FloatAttr) and host_math.isnan(value.value.data):
                raise ConversionError("NaN constant payload preservation is not qualified", source_id=source_id, opcode=name)
        if name == "tt.dot":
            precision = op.properties.get("inputPrecision", op.attributes.get("inputPrecision"))
            if not isinstance(precision, IntegerAttr) or precision.value.data != 2:
                raise ConversionError("dot precision must explicitly be IEEE; upstream drops this policy", source_id=source_id, opcode=name)
            accumulator = op.operands[2].owner
            value = getattr(accumulator, "properties", {}).get("value")
            positive_zero = (isinstance(value, builtin.DenseIntOrFPElementsAttr) and value.is_splat()
                and all(v == 0 and (not isinstance(v, float) or host_math.copysign(1, v) > 0) for v in value.get_values()))
            records[-1]["restore_matmul_accumulator"] = not positive_zero
    return records


def _restore_accumulators(module, records):
    """Repair the surveyed upstream zero-matmul-then-add expansion structurally.

    The source accumulator belongs in the contraction's destination. Moving it
    after the entire contraction loses the original recurrence/numeric intent.
    Refuse changed/ambiguous patterns instead of guessing from operation order.
    """
    required = {r["id"] for r in records if r.get("restore_matmul_accumulator")}
    repaired, observed = [], set()
    for matmul in list(module.walk()):
        ids = set(_source_ids(matmul.location)) & required
        if opcode(matmul) not in ("linalg.matmul", "linalg.batch_matmul") or not ids:
            continue
        users = {use.operation for use in matmul.results[0].uses}
        if len(users) != 1:
            raise ConversionError("upstream accumulator expansion has ambiguous contraction uses")
        add = next(iter(users))
        if not isinstance(add, GenericOp) or len(add.inputs) != 2 or len(add.outputs) != 1 or len(add.results) != 1:
            raise ConversionError("upstream nonzero accumulator expansion is not the qualified pointwise add")
        rank = len(matmul.results[0].type.get_shape())
        if any(m.data != AffineMap.identity(rank) for m in add.indexing_maps):
            raise ConversionError("upstream accumulator add changed its indexing maps")
        scalar = list(add.body.block.ops)
        if (len(scalar) != 2 or scalar[0].name not in ("arith.addf", "arith.addi")
            or scalar[1].name != "linalg.yield" or tuple(scalar[0].operands) not in
                (tuple(add.body.block.args[:2]), tuple(reversed(add.body.block.args[:2])))
            or tuple(scalar[1].operands) != tuple(scalar[0].results)):
            raise ConversionError("upstream accumulator combiner changed")
        accumulator = next((v for v in add.inputs if v is not matmul.results[0]), None)
        if accumulator is None or accumulator.type != matmul.results[0].type:
            raise ConversionError("upstream source accumulator has an incompatible tensor type")
        matmul.operands = (*matmul.operands[:2], accumulator)
        add.results[0].replace_all_uses_with(matmul.results[0])
        add.parent.erase_op(add)
        observed.update(ids)
        repaired.append({"kind": "restore-source-contraction-accumulator", "source_ids": sorted(ids),
                         "checked_pattern": "unique-use identity-map pointwise add of contraction and source C"})
    if observed != required:
        raise ConversionError("upstream did not expose every required source accumulator for restoration")
    return repaired


def _functionalize(module, abi):
    function = next(op for op in module.ops if opcode(op) == "tt.func")
    outer = Block(arg_types=[b.type for b in abi.inputs])
    writable = list(abi.outputs)
    grid = abi.grid_bounds
    parallel = host_math.prod(grid) > 1 or any(isinstance(g, GridExtent) for g in abi.grid)
    body = Block(arg_types=[IndexType(), *[b.type for b in writable]]) if parallel else outer
    mapping, buffers, descriptors, facts = {}, {}, {}, {}
    gather_descriptors = {}
    states = {b.name: arg for b, arg in zip(abi.inputs, outer.args, strict=True)}
    for scalar in abi.runtime_scalars:
        guard_scalar(scalar, states[scalar.name], outer, function.location)
    dimensions = bind_dimensions(abi, states, outer, function.location)
    launch = launch_sizes(abi, dimensions, outer, function.location) if abi.dynamic else None
    for buffer in abi.buffers:
        if buffer.effect == "write":
            empty = tensor.EmptyOp([dimensions[n] for n in buffer.shape if isinstance(n, str)], buffer.type)
            outer.add_op(empty)
            states[buffer.name] = empty.tensor
    initial = [states[b.name] for b in writable]
    initialization = DefiniteWrites(abi)
    shared = {b.name: arg for b, arg in zip(writable, body.args[1:], strict=True)} if parallel else {}
    states.update(shared)
    writes, reads, source_loops, source_conditionals, source_gathers, source_scatters = {}, [], [], [], [], []
    launch_iv = body.args[0] if parallel else None
    loop_counter = 0
    loop_values = {}
    for arg, (kind, value) in zip(function.regions[0].block.args, abi.source_bindings, strict=True):
        if kind == "buffer":
            buffers[arg] = value
        elif kind == "runtime":
            mapping[arg] = states[value.name]
            if value.bounds is not None:
                facts[mapping[arg]] = Linear(parameters=((value.name, *value.bounds, 1),))
        else:
            if not isinstance(arg.type, builtin.IntegerType) or not -(1 << (arg.type.width.data - 1)) <= value < 1 << (arg.type.width.data - 1):
                raise ConversionError("scalar specialization must fit its source integer type")
            constant = arith.ConstantOp(IntegerAttr(value, arg.type))
            body.add_op(constant)
            mapping[arg] = constant.result

    def append(op, source):
        op.location = source.location
        op.attributes.update({"prov.family": StringAttr("triton"),
                              "prov.op": StringAttr(opcode(source)),
                              "prov.source_ids": ArrayAttr(StringAttr(s) for s in _source_ids(source.location))})
        body.add_op(op)
        return op

    @dataclass(frozen=True)
    class Coordinate:
        form: Linear
        value: object = None

    def resolve(op, field, operands, coordinates=False, raw=False):
        dynamic = iter(operands)
        result = []
        for number in _array(op, field):
            if number == DYNAMIC_INDEX:
                value = next(dynamic, None)
                if value is None:
                    raise ConversionError("structured descriptor is missing a dynamic operand", opcode=opcode(op))
                value = mapping.get(value, value)
                if raw:
                    result.append(value)
                    continue
                if coordinates:
                    result.append(Coordinate(linear(value, facts, grid), value))
                    continue
                number = _integer(value)
            if coordinates:
                number = Coordinate(Linear(number))
            result.append(number)
        if next(dynamic, None) is not None:
            raise ConversionError("structured descriptor segment mismatch", opcode=opcode(op))
        return tuple(result)

    def descriptor(op):
        base = op.operands[0]
        if base not in buffers:
            raise ConversionError("structured base pointer needs a qualified scalar-pointer/loop adapter", opcode=opcode(op))
        segments = _array(op, "operandSegmentSizes")
        if len(segments) != 4 or segments[0] != 1 or sum(segments) != len(op.operands):
            raise ConversionError("invalid structured pointer operand segments")
        slices, start = [], 1
        for length in segments[1:]:
            slices.append(op.operands[start:start + length])
            start += length
        shape = resolve(op, "static_shape", slices[2])
        if any(shape) or _array(op, "order"):
            raise ConversionError("wraparound/block-pointer bounds need a qualified tensor view adapter", opcode=opcode(op))
        buffer, sizes = buffers[base], _array(op, "sizes")
        if any(isinstance(n, str) for n in buffer.shape):
            strides = resolve(op, "static_strides", slices[0], raw=True)
            offsets = resolve(op, "static_offsets", slices[1], raw=True)
            forms, axes = recover_axes(abi, buffer, sizes, strides, offsets, facts)
            coordinates = tuple(Coordinate(form, affine_index(form, launch_iv, grid, launch, states,
                loop_values, lambda value: append(value, op)) if form.dynamic else None) for form in forms)
            return buffer, sizes, coordinates, DenseAxes(axes)
        strides = resolve(op, "static_strides", slices[0])
        offsets = resolve(op, "static_offsets", slices[1], coordinates=True)
        if (len(sizes) != len(buffer.shape) or len(strides) != len(sizes) or
                any(stride < 0 or stride == 0 and size != 1 for stride, size in zip(strides, sizes, strict=True))):
            raise ConversionError("descriptor requires matching logical rank and positive structured strides", opcode=opcode(op))
        total = Linear()
        for off in offsets:
            total = total + off.form
        try:
            forms = logical_coordinates(total, buffer.strides, grid)
        except ConversionError:
            # Upstream sums per-axis *element* offsets (PtrState::
            # accumulateTargetOffset). A surplus masked column can cross the
            # row stride, so splitting that sum by quotient/remainder loses
            # the original axes. Recover aligned contributions individually.
            # Actual mask-dependent bounds are still checked at each access.
            low, high = total.bounds(grid)
            if not -(1 << 63) <= low <= high < 1 << 63:
                raise ConversionError("logical address arithmetic may overflow")
            axes = axis_permutation(buffer, strides, [Linear()] * len(sizes),
                                    [1] * len(sizes), grid, physical_offsets=True,
                                    dimensions=tuple(abi.dimension(n) for n in buffer.shape))
            coordinates = [None] * len(sizes)
            for off, stride, axis in zip(offsets, strides, axes, strict=True):
                stride = buffer.strides[axis]  # A size-one source axis may have stride zero.
                form = off.form.divide(stride)
                value = off.value
                if form.dynamic and stride != 1:
                    divisor = append(arith.ConstantOp.from_int_and_width(stride, IndexType()), op).result
                    value = append(arith.DivSIOp(value, divisor), op).result
                coordinates[axis] = Coordinate(form, value)
            return buffer, sizes, tuple(coordinates), strides
        value = None
        if any(form.dynamic for form in forms):
            for off in offsets:
                term = off.value
                if not off.form.dynamic:
                    term = append(arith.ConstantOp.from_int_and_width(off.form.constant, IndexType()), op).result
                value = term if value is None else append(arith.AddiOp(value, term), op).result
        coordinates = []
        for form, stride in zip(forms, buffer.strides, strict=True):
            coordinate = value
            if value is not None and stride != 1:
                divisor = append(arith.ConstantOp.from_int_and_width(stride, IndexType()), op).result
                coordinate = append(arith.DivSIOp(value, divisor), op).result
                value = append(arith.RemSIOp(value, divisor), op).result
            coordinates.append(Coordinate(form, coordinate))
        return buffer, sizes, tuple(coordinates), strides

    def access(op, desc, store=False):
        buffer, sizes, offsets, strides = desc
        legacy = opcode(op) in ("tt.load", "tt.store")
        unit = opcode(op) in ("tts.gather", "tts.scatter")
        if legacy and len(op.operands) != (2 if store else 1):
            raise ConversionError("post-loop pointer memory requires an unmasked recovered descriptor", opcode=opcode(op))
        mask = () if legacy or unit else _array(op, "static_mask_dims")
        dynamic = () if unit else op.operands[2:] if store else op.operands[1:1 + _array(op, "operandSegmentSizes")[1]]
        if mask:
            values = iter(dynamic)
            if len(mask) != len(sizes):
                raise ConversionError("structured mask rank differs from its tile", opcode=opcode(op))
            active = []
            for number, size in zip(mask, sizes, strict=True):
                if number == DYNAMIC_INDEX:
                    v = next(values, None)
                    if v is None:
                        raise ConversionError("structured mask is missing a dynamic operand", opcode=opcode(op))
                    number = recover_extent(mapping.get(v, v), size, [off.form for off in offsets], facts, grid)
                active.append(number)
            active = tuple(active)
            if next(values, None) is not None:
                raise ConversionError("structured mask operand segment mismatch", opcode=opcode(op))
        else:
            active = sizes
        if len(active) != len(sizes) or any(maximum(n) < 0 or maximum(n) > size for n, size in zip(active, sizes, strict=True)):
            raise ConversionError("mask must describe a bounded rectangular prefix", opcode=opcode(op))
        axes = axis_permutation(buffer, [None] * len(sizes) if isinstance(strides, DenseAxes) else strides,
                                [off.form for off in offsets], active, grid, physical_offsets=True,
                                dimensions=tuple(abi.dimension(n) for n in buffer.shape),
                                proven_axes=strides.axes if isinstance(strides, DenseAxes) else None)
        # Empty accesses make no memory observation; keep their standard slice
        # coordinates within the logical tensor as well.
        if any(n == 0 for n in active):
            offsets = tuple(Coordinate(Linear(0)) for _ in offsets)
        else:
            # A fully inactive tile may start past the buffer end. Its zero
            # length is proved above; standard slices still require an in-bounds
            # offset. Keep the logical form for ownership, clamp only its SSA.
            bounded = []
            for off, dimension in zip(offsets, buffer.shape, strict=True):
                value = off.value
                bound = abi.dimension(dimension)
                if (off.form + (bound.scale(-1) if isinstance(bound, Linear) else Linear(-bound))).bounds(grid)[1] > 0:
                    limit = dimensions[dimension] if isinstance(dimension, str) else append(arith.ConstantOp.from_int_and_width(dimension, IndexType()), op).result
                    if value is None:
                        value = append(arith.ConstantOp.from_int_and_width(off.form.constant, IndexType()), op).result
                    value = append(arith.MinSIOp(value, limit), op).result
                bounded.append(Coordinate(off.form, value))
            offsets = tuple(bounded)
        return buffer, sizes, offsets, active, axes

    def unit_descriptor(op, store=False):
        # Upstream represents all-unit tensor pointers as gather/scatter.
        # A single zero index has no irregular indexing or write conflicts.
        if len(op.operands) not in ((3, 4) if store else (2,)) or op.operands[0] not in buffers:
            raise ConversionError("gather/scatter needs a qualified tensor index/effect adapter", opcode=opcode(op))
        buffer = buffers[op.operands[0]]
        indices = mapping[op.operands[1]]
        if (not isinstance(indices.type, TensorType) or not all(n == 1 for n in indices.type.get_shape())
                or not all(n == 1 for n in buffer.shape)):
            raise ConversionError("only all-unit, zero-index gather/scatter is qualified", opcode=opcode(op))
        owner = indices.owner
        if opcode(owner) == "linalg.fill":
            index = _integer(owner.inputs[0])
        elif isinstance(owner, arith.ConstantOp) and isinstance(owner.value, builtin.DenseIntOrFPElementsAttr):
            index = next(iter(owner.value.get_values()))
        else:
            raise ConversionError("unit gather/scatter index is not established constant zero", opcode=opcode(op))
        type_ = op.operands[2].type if store else op.results[0].type
        if index != 0 or type_ != buffer.type:
            raise ConversionError("unit gather/scatter index or value shape differs from the logical tensor", opcode=opcode(op))
        return buffer, buffer.shape, tuple(Coordinate(Linear()) for _ in buffer.shape), buffer.strides

    def transpose(value, axes, source):
        if axes == tuple(range(len(axes))):
            return value
        shape = value.type.get_shape()
        target = TensorType(value.type.get_element_type(), [shape[i] for i in axes])
        dynamic = []
        for i in axes:
            if shape[i] == DYNAMIC_INDEX:
                index = append(arith.ConstantOp.from_int_and_width(i, IndexType()), source).result
                dynamic.append(append(tensor.DimOp(value, index), source).result)
        empty = append(tensor.EmptyOp(dynamic, target), source).tensor
        return append(TransposeOp(value, empty, DenseArrayBase.from_list(builtin.i64, axes)), source).results[0]

    def slice_op(source, offsets, sizes, *, destination=None, parallel_insert=False):
        dynamic = [off.value for off in offsets if off.value is not None]
        static = [DYNAMIC_INDEX if off.value is not None else off.form.constant for off in offsets]
        shape = [DYNAMIC_INDEX if isinstance(n, TailExtent) else n for n in sizes]
        dynamic_sizes = [n.value for n in sizes if isinstance(n, TailExtent)]
        properties = {"static_offsets": DenseArrayBase.from_list(builtin.i64, static),
                      "static_sizes": DenseArrayBase.from_list(builtin.i64, shape),
                      "static_strides": DenseArrayBase.from_list(builtin.i64, [1] * len(sizes))}
        if destination is None:
            return tensor.ExtractSliceOp.build(operands=[source, dynamic, dynamic_sizes, []], properties=properties,
                       result_types=[TensorType(source.type.get_element_type(), shape)])
        cls = ParallelInsertSliceOp if parallel_insert else tensor.InsertSliceOp
        return cls.build(operands=[source, destination, dynamic, dynamic_sizes, []], properties=properties,
                         result_types=[] if parallel_insert else [destination.type])

    def pad_tile(loaded, sizes, active, source, fill):
        padding = Block(arg_types=[IndexType()] * len(sizes))
        padding.add_op(tensor.YieldOp(fill))
        high, dynamic_high = [], []
        for size, n in zip(sizes, active, strict=True):
            if isinstance(n, TailExtent):
                cap = append(arith.ConstantOp.from_int_and_width(size, IndexType()), source).result
                dynamic_high.append(append(arith.SubiOp(cap, n.value), source).result)
                high.append(DYNAMIC_INDEX)
            else:
                high.append(size - n)
        shape = [DYNAMIC_INDEX if h == DYNAMIC_INDEX else size for size, h in zip(sizes, high, strict=True)]
        loaded = append(tensor.PadOp(loaded, [], dynamic_high, Region(padding), [0] * len(sizes),
                        high, TensorType(loaded.type.element_type, shape)), source).result
        if dynamic_high:
            loaded = append(tensor.CastOp.build(operands=[loaded],
                result_types=[TensorType(loaded.type.element_type, sizes)]), source).dest
        return loaded

    def scatter_access(op):
        if len(op.operands) not in (3, 4) or op.operands[0] not in buffers:
            raise ConversionError("scatter needs a declared logical tensor and explicit index/value/mask schema", opcode=opcode(op))
        buffer = buffers[op.operands[0]]
        index, value = mapping[op.operands[1]], mapping[op.operands[2]]
        mask = mapping[op.operands[3]] if len(op.operands) == 4 else None
        if (not isinstance(value.type, TensorType) or not isinstance(index.type, TensorType)
                or index.type.get_shape() != value.type.get_shape() or value.type.element_type != buffer.type.element_type
                or mask is not None and mask.type != TensorType(builtin.i1, value.type.get_shape())):
            raise ConversionError("scatter index/value/mask types differ from the logical tensor ABI", opcode=opcode(op))
        try:
            if all(n == 1 for n in buffer.shape):
                return (*access(op, unit_descriptor(op, store=True), store=True), mask)
            forms, sizes, axes = scatter_tile(buffer, abi, index, mask, facts)
        except ConversionError as exc:
            if exc.source_id is not None:
                raise
            raise ConversionError(str(exc), source_id=next(iter(_source_ids(op.location)), None),
                                  opcode=exc.opcode or opcode(op)) from exc
        offsets, physical = [], []
        for destination_axis, (form, dimension) in enumerate(zip(forms, buffer.shape, strict=True)):
            size = sizes[axes.index(destination_axis)]
            raw = affine_index(form, launch_iv, grid, launch, states, loop_values, lambda x: append(x, op))
            bound = abi.dimension(dimension)
            bound_form = bound if isinstance(bound, Linear) else Linear(bound)
            offset = raw
            if (form + bound_form.scale(-1)).bounds(grid)[1] > 0:
                limit = dimensions[dimension] if isinstance(dimension, str) else append(arith.ConstantOp.from_int_and_width(dimension, IndexType()), op).result
                offset = append(arith.MinSIOp(raw, limit), op).result
            extent = size
            if (form + Linear(size) + bound_form.scale(-1)).bounds(grid)[1] > 0:
                limit = dimensions[dimension] if isinstance(dimension, str) else append(arith.ConstantOp.from_int_and_width(dimension, IndexType()), op).result
                remaining = append(arith.SubiOp(limit, raw), op).result
                cap = append(arith.ConstantOp.from_int_and_width(size, IndexType()), op).result
                zero = append(arith.ConstantOp.from_int_and_width(0, IndexType()), op).result
                clipped = append(arith.MaxSIOp(append(arith.MinSIOp(cap, remaining), op).result, zero), op).result
                extent = TailExtent(size, bound, form, clipped)
            offsets.append(Coordinate(form, offset))
            physical.append(extent)
        active = tuple(physical[axis] for axis in axes)
        return buffer, sizes, tuple(offsets), active, axes, mask

    def translate(source_block):
        nonlocal body, loop_counter
        for op in source_block.ops:
            name = opcode(op)
            if name == "ub.poison":
                if len(op.results) != 1 or not _pointer_type(op.results[0].type) or not all(
                        _unused_pointer_initialization(use) for use in op.results[0].uses):
                    raise ConversionError("poison is observable outside unused pointer carrier initialization", opcode=name)
            elif name == "scf.if":
                source_conditionals.append({"source_ids": _source_ids(op.location),
                                            "memory": "branch-results-carry-tensor-versions",
                                            "initialization": "intersection-of-both-branch-facts"})
                if any(isinstance(v.type, builtin.UnregisteredAttr) or
                       isinstance(v.type, TensorType) and isinstance(v.type.get_element_type(), builtin.UnregisteredAttr)
                       for v in op.results):
                    raise ConversionError("branch-returned pointers need a descriptor merge adapter", opcode=name)
                incoming, before = dict(states), initialization.snapshot()
                regions, initialized = [], []
                previous = body
                for source_region in (op.true_region, op.false_region):
                    states.update(incoming)
                    initialization.restore(before)
                    body = Block()
                    if source_region.blocks:
                        if len(source_region.blocks) != 1 or source_region.block.args:
                            raise ConversionError("conditional regions need one argument-free block", opcode=name)
                        translate(source_region.block)
                    else:
                        body.add_op(scf.YieldOp(*[states[b.name] for b in writable]))
                    regions.append(Region(body))
                    initialized.append(initialization.snapshot())
                body = previous
                initialization.merge_branches(*initialized)
                branch = append(scf.IfOp(mapping[op.cond], [*[v.type for v in op.results],
                                                          *[b.type for b in writable]], *regions), op)
                mapping.update(zip(op.results, branch.results[:len(op.results)], strict=True))
                states.update((b.name, value) for b, value in zip(writable, branch.results[len(op.results):], strict=True))
            elif name == "scf.for":
                if value_only_region(op):
                    source_loops.append({"source_ids": _source_ids(op.location),
                                         "bounds": "source-SSA; runtime-or-static",
                                         "memory_effects": "no-buffer-accesses",
                                         "recurrence": "original-standard-value-region"})
                    append(op.clone(value_mapper=mapping), op)
                    continue
                arguments = op.body.block.args
                pointer_slots = [i for i, v in enumerate(arguments[1:]) if _pointer_type(v.type)]
                if any(arguments[i + 1].uses for i in pointer_slots):
                    raise ConversionError("live pointer loop arguments need descriptor-state normalization", opcode=name)
                if any((isinstance(v.type, builtin.UnregisteredAttr) or
                        isinstance(v.type, TensorType) and isinstance(v.type.get_element_type(), builtin.UnregisteredAttr))
                       and not _pointer_type(v.type) for v in arguments):
                    raise ConversionError("unqualified loop-carried source type", opcode=name)
                if any(not isinstance(arguments[i + 1].type, TensorType) for i in pointer_slots):
                    raise ConversionError("scalar pointer carriers need a rank-aware descriptor adapter", opcode=name)
                kept = [i for i in range(len(arguments) - 1) if i not in pointer_slots]
                require_signed_loop(op)
                limits = [mapping[v] for v in op.operands[:3]]
                domain = bounded_loop([linear(v, facts, grid).bounds(grid) for v in limits],
                                      width=getattr(getattr(arguments[0].type, "width", None), "data", 64))
                lb, ub, step = domain.lower, domain.upper[1], domain.step
                if not domain.exact and pointer_slots:
                    raise ConversionError("runtime pointer carriers need final-state normalization", opcode=name)
                loop_evidence = {"source_ids": _source_ids(op.location), "lower": lb, "step": step,
                                 "memory_effects": "functional-loop-carried-tensor-versions"}
                if domain.exact:
                    loop_evidence.update(upper=ub, iterations=domain.maximum_iterations)
                else:
                    loop_evidence.update(upper_interval=domain.upper, minimum_iterations=domain.minimum_iterations,
                        maximum_iterations=domain.maximum_iterations, bounds="original-runtime-SSA",
                        memory_effects="read-only-source-accesses; no-write-credit-from-iterations")
                source_loops.append(loop_evidence)
                memory_inputs = [states[b.name] for b in writable]
                pointer_ranks = [len(arguments[i + 1].type.get_shape()) for i in pointer_slots]
                kept_arguments = [arguments[0], *[arguments[i + 1] for i in kept]]
                nested = Block(arg_types=[*[v.type for v in kept_arguments], *[b.type for b in writable],
                                         *([IndexType()] * sum(pointer_ranks))])
                mapping.update(zip(kept_arguments, nested.args[:len(kept_arguments)], strict=True))
                states.update((b.name, v) for b, v in zip(writable, nested.args[len(kept_arguments):len(kept_arguments) + len(writable)], strict=True))
                iterations = (ub - lb + step - 1) // step
                facts[nested.args[0]] = Linear(lb, loops=((loop_counter, 0, iterations - 1, step),))
                final_facts, recurrences = seed_recurrences(op, [mapping.get(v) for v in arguments], mapping,
                                                           facts, grid, lb, ub, step, _integer, exact_iterations=domain.exact)
                source_loops[-1]["index_recurrences"] = recurrences
                identity = loop_counter
                loop_counter += 1
                previous, body = body, nested
                before_loop = initialization.snapshot()
                loop_values[identity] = (nested.args[0], lb, step)
                translate(op.body.block)
                del loop_values[identity]
                if not domain.exact:
                    changed = any(states[b.name] is not nested.args[len(kept_arguments) + i]
                                  for i, b in enumerate(writable))
                    if changed:
                        loop_evidence["memory_effects"] = "conditional-loop-carried-tensor-versions"
                    if domain.minimum_iterations == 0:
                        initialization.restore(before_loop)
                        loop_evidence["initialization"] = "entry-facts-only; zero-trip-reachable"
                    else:
                        loop_evidence["initialization"] = "post-body-facts; positive-trip-count-proved"
                body = previous
                initial = [mapping[op.operands[i + 3]] for i in kept]
                zeros = [append(arith.ConstantOp.from_int_and_width(0, IndexType()), op).result for _ in range(sum(pointer_ranks))]
                loop = append(scf.ForOp(*limits, [*initial, *memory_inputs, *zeros], Region(nested)), op)
                for position, i in enumerate(kept):
                    mapping[op.results[i]] = loop.results[position]
                    if i in final_facts:
                        facts[loop.results[position]] = final_facts[i]
                states.update((b.name, v) for b, v in zip(writable, loop.results[len(kept):len(kept) + len(writable)], strict=True))
                cursor = len(kept) + len(writable)
                for i, rank in zip(pointer_slots, pointer_ranks, strict=True):
                    yielded = op.body.block.last_op.operands[i]
                    if yielded not in descriptors:
                        raise ConversionError("returned pointer has no recovered structured descriptor", opcode=name)
                    buffer, sizes, offsets, strides = descriptors[yielded]
                    if len(offsets) != rank:
                        raise ConversionError("returned pointer descriptor rank changed", opcode=name)
                    final = []
                    for off, value in zip(offsets, loop.results[cursor:cursor + rank], strict=True):
                        constant = off.form.constant + sum(high * c for loop_id, _low, high, c in off.form.loops if loop_id == identity)
                        form = Linear(constant, off.form.programs, tuple(term for term in off.form.loops if term[0] != identity), off.form.parameters)
                        facts[value] = form
                        final.append(Coordinate(form, value))
                    descriptors[op.results[i]] = buffer, sizes, tuple(final), strides
                    cursor += rank
            elif name == "scf.yield":
                values, coordinates = [], []
                for value in op.operands:
                    if _pointer_type(value.type):
                        if value not in descriptors:
                            raise ConversionError("yielded pointer lacks a structured descriptor", opcode=name)
                        for off in descriptors[value][2]:
                            coordinates.append(off.value if off.form.dynamic else
                                append(arith.ConstantOp.from_int_and_width(off.form.constant, IndexType()), op).result)
                    else:
                        values.append(mapping[value])
                append(scf.YieldOp(*values, *[states[b.name] for b in writable], *coordinates), op)
            elif name == "bufferization.alloc_tensor":
                if op.operands or len(op.results) != 1 or not isinstance(op.results[0].type, TensorType) or DYNAMIC_INDEX in op.results[0].type.get_shape():
                    raise ConversionError("only static, copy-free allocation tensors can become tensor.empty", opcode=name)
                mapping[op.results[0]] = append(tensor.EmptyOp([], op.results[0].type), op).tensor
            elif name in ("tt.get_program_id", "tt.get_num_programs"):
                axis = op.properties["axis"].value.data
                size = grid[axis] if axis < len(grid) else 1
                if launch is not None and name == "tt.get_num_programs" and axis < len(launch):
                    value = append(arith.IndexCastOp(launch[axis], op.results[0].type), op).result
                    if isinstance(abi.grid[axis], GridExtent):
                        key = "$grid:" + str(axis)
                        states[key] = value
                        scalar = next(s for s in abi.runtime_scalars if s.name == abi.grid[axis].dimension)
                        low = (scalar.bounds[0] + abi.grid[axis].divisor - 1) // abi.grid[axis].divisor
                        facts[value] = Linear(parameters=((key, low, size, 1),))
                    else:
                        facts[value] = Linear(size)
                elif name == "tt.get_num_programs" or size == 1:
                    value = append(arith.ConstantOp(IntegerAttr(size if name == "tt.get_num_programs" else 0, op.results[0].type)), op).result
                elif launch is not None:
                    value = program_id(axis, launch_iv, launch, lambda x: append(x, op))
                    value = append(arith.IndexCastOp(value, op.results[0].type), op).result
                    facts[value] = Linear(0, tuple(int(i == axis) for i in range(3)))
                else:
                    value = launch_iv
                    divisor = host_math.prod(grid[axis + 1:])
                    if divisor != 1:
                        c = append(arith.ConstantOp.from_int_and_width(divisor, IndexType()), op).result
                        value = append(arith.DivUIOp(value, c), op).result
                    c = append(arith.ConstantOp.from_int_and_width(size, IndexType()), op).result
                    value = append(arith.RemUIOp(value, c), op).result
                    value = append(arith.IndexCastOp(value, op.results[0].type), op).result
                    facts[value] = Linear(0, tuple(int(i == axis) for i in range(3)))
                mapping[op.results[0]] = value
            elif name == "tts.make_tptr":
                descriptors[op.results[0]] = descriptor(op)
            elif name == "tts.make_gather_scatter_tptr":
                gather_descriptors[op.results[0]] = op
            elif name == "tts.load" and op.operands[0] in gather_descriptors:
                pointer = gather_descriptors[op.operands[0]]
                if pointer.operands[0] not in buffers:
                    raise ConversionError("structured gather base must be a declared logical allocation", opcode=name)
                buffer = buffers[pointer.operands[0]]
                mapping[op.results[0]] = structured_gather(op, pointer, buffer, states[buffer.name], mapping, append,
                    maximum_shape=tuple(abi.dimension(n).bounds(grid)[1] if isinstance(n, str) else n for n in buffer.shape))
                source_gathers.append({"source_ids": _source_ids(op.location), "buffer": buffer.name,
                                       "bounds": "runtime-assert-before-active-extract",
                                       "mask": "rectangular-prefix-branch-no-inactive-extraction",
                                       "representation": "upstream-structured-gather-descriptor"})
            elif name == "tts.gather" and op.operands[0] in buffers and buffers[op.operands[0]].effect == "read":
                buffer = buffers[op.operands[0]]
                mapping[op.results[0]] = gather(op, buffer, states[buffer.name], mapping, append,
                    maximum_shape=tuple(abi.dimension(n).bounds(grid)[1] if isinstance(n, str) else n for n in buffer.shape))
                source_gathers.append({"source_ids": _source_ids(op.location), "buffer": buffer.name,
                                       "bounds": "runtime-assert-before-active-extract",
                                       "mask": "per-lane-branch-no-inactive-extraction",
                                       "repeated_indices": "immutable-reads; no-ownership-conflict"})
            elif name in ("tts.load", "tt.load", "tts.gather"):
                if name != "tts.gather" and op.operands[0] not in descriptors:
                    raise ConversionError("load pointer is not a qualified structured descriptor", opcode=name)
                desc = unit_descriptor(op) if name == "tts.gather" else descriptors[op.operands[0]]
                buffer, sizes, offsets, active, axes = access(op, desc)
                physical = tuple(active[i] for i in inverse_permutation(axes))
                initialization.load(buffer, tuple(off.form for off in offsets), physical,
                                    source_id=next(iter(_source_ids(op.location)), None), opcode=name)
                if parallel and buffer.effect == "readwrite":
                    reads.append((buffer.name, tuple(off.form for off in offsets), physical))
                loaded = append(slice_op(states[buffer.name], offsets, physical), op).result
                loaded = transpose(loaded, axes, op)
                if active != sizes:
                    segments = _array(op, "operandSegmentSizes")
                    if segments[2] != 1:
                        raise ConversionError("masked load without other needs defined-lane propagation", opcode=name)
                    fill = mapping.get(op.operands[-1])
                    if fill is None:
                        raise ConversionError("masked load fill is not a translated scalar", opcode=name)
                    loaded = pad_tile(loaded, sizes, active, op, fill)
                mapping[op.results[0]] = loaded
            elif name in ("tts.store", "tt.store", "tts.scatter"):
                if name != "tts.scatter" and op.operands[0] not in descriptors:
                    raise ConversionError("store pointer is not a qualified structured descriptor", opcode=name)
                lane_mask = None
                if name == "tts.scatter":
                    buffer, sizes, offsets, active, axes, lane_mask = scatter_access(op)
                else:
                    buffer, sizes, offsets, active, axes = access(op, descriptors[op.operands[0]], store=True)
                physical = tuple(active[i] for i in inverse_permutation(axes))
                if buffer.effect == "read":
                    raise ConversionError("source stores into a declared read-only buffer", opcode=name)
                if parallel:
                    footprint = (tuple(off.form for off in offsets), physical)
                    independent_tile(footprint[0], tuple(maximum(n) for n in physical), grid)
                    if buffer.name in writes and writes[buffer.name][0] != footprint:
                        raise ConversionError("parallel local stores need one established owned tile per output")
                    writes[buffer.name] = (footprint, offsets, op)
                value = mapping[op.operands[2 if name == "tts.scatter" else 1]]
                if lane_mask is not None:
                    initialization.load(buffer, tuple(off.form for off in offsets), physical,
                        source_id=next(iter(_source_ids(op.location)), None), opcode=name)
                    if parallel and buffer.effect == "readwrite":
                        reads.append((buffer.name, tuple(off.form for off in offsets), physical))
                    old = transpose(append(slice_op(states[buffer.name], offsets, physical), op).result, axes, op)
                    if active != sizes:
                        element = buffer.type.element_type
                        literal = builtin.FloatAttr(0.0, element) if isinstance(element, builtin.AnyFloat) else IntegerAttr(0, element)
                        fill = append(arith.ConstantOp(literal), op).result
                        old = pad_tile(old, sizes, active, op, fill)
                    from xdsl.dialects.linalg.attrs import IteratorTypeAttr
                    from xdsl.dialects.linalg.ops import YieldOp
                    block = Block(arg_types=[value.type.element_type, builtin.i1, value.type.element_type])
                    selected = arith.SelectOp(block.args[1], block.args[0], block.args[2])
                    block.add_ops([selected, YieldOp(selected.result)])
                    maps = [builtin.AffineMapAttr(AffineMap.identity(len(sizes)))] * 3
                    merged = append(GenericOp([value, lane_mask], [old], Region(block), maps,
                        [IteratorTypeAttr.parallel()] * len(sizes), [value.type]), op)
                    merged.attributes.update({"prov.family": StringAttr("memory"), "prov.op": StringAttr("masked_store"),
                        "prov.mask_semantics": StringAttr("preserve-inactive-initialized-values")})
                    value = merged.results[0]
                if name == "tts.scatter":
                    source_scatters.append({"source_ids": _source_ids(op.location), "buffer": buffer.name,
                        "address": "proved-affine-dense-tile", "mask": "exact-per-lane-select" if lane_mask is not None else "unmasked",
                        "proof": "projected-linalg-index-and-necessary-mask-bounds.v1",
                        "inactive_values": "preserved-initialized-state", "write_independence": "separately-proved-owned-tile"})
                if active != sizes:
                    value = append(slice_op(value, [Coordinate(Linear()) for _ in sizes], active), op).result
                value = transpose(value, inverse_permutation(axes), op)
                states[buffer.name] = append(slice_op(value, offsets, physical, destination=states[buffer.name]), op).result
                if lane_mask is None:
                    initialization.store(buffer, tuple(off.form for off in offsets), physical)
            elif name == "tt.return":
                if not parallel:
                    append(func.ReturnOp(*(states[b.name] for b in writable)), op)
            else:
                for child in op.walk():
                    child_name = opcode(child)
                    if child.name == "builtin.unregistered" or child_name.split(".")[0] not in ("arith", "math", "tensor", "linalg", "scf", "cf") or child_name in ("cf.br", "cf.cond_br"):
                        raise ConversionError("upstream left an operation requiring a tensor/effect adapter",
                                              source_id=next(iter(_source_ids(child.location)), None), opcode=child_name)
                    if child_name == "builtin.unregistered" or any(v not in mapping and v in buffers for v in child.operands):
                        raise ConversionError("pointer escaped structured memory conversion", opcode=child_name)
                cloned = op.clone(value_mapper=mapping)
                append(cloned, op)
    translate(function.regions[0].block)
    if parallel:
        def index_constant(number, source):
            return append(arith.ConstantOp.from_int_and_width(number, IndexType()), source).result

        def owned_index(form, source):
            # Rebuild owned coordinates outside source branches/loops. Their
            # SSA temporaries need not dominate the final parallel publication.
            if form.loops:
                raise ConversionError("parallel publication needs a final-state proof for source-loop-dependent extents")
            value = index_constant(form.constant, source)
            for axis, coefficient in enumerate(form.programs[:len(grid)]):
                if not coefficient or grid[axis] == 1:
                    continue
                if launch is not None:
                    pid = program_id(axis, launch_iv, launch, lambda x: append(x, source))
                    term = append(arith.MuliOp(pid, index_constant(coefficient, source)), source).result
                    value = append(arith.AddiOp(value, term), source).result
                    continue
                divisor = host_math.prod(grid[axis + 1:])
                pid = launch_iv
                if divisor != 1:
                    pid = append(arith.DivUIOp(pid, index_constant(divisor, source)), source).result
                pid = append(arith.RemUIOp(pid, index_constant(grid[axis], source)), source).result
                term = append(arith.MuliOp(pid, index_constant(coefficient, source)), source).result
                value = append(arith.AddiOp(value, term), source).result
            for name, _, _, coefficient in form.parameters:
                parameter = states[name]
                if parameter.type != IndexType():
                    parameter = append(arith.IndexCastOp(parameter, IndexType()), source).result
                term = append(arith.MuliOp(parameter, index_constant(coefficient, source)), source).result
                value = append(arith.AddiOp(value, term), source).result
            return value

        for name, offsets, active in reads:
            if name not in writes or writes[name][0] != (offsets, active):
                raise ConversionError("readwrite load crosses a program's established owned tile")
        updates = Block()
        for buffer in writable:
            if buffer.name not in writes:
                raise ConversionError("parallel writable result has no established store tile")
            (forms, active), _offsets, source = writes[buffer.name]
            offsets, published = [], []
            for form, size, dimension in zip(forms, active, buffer.shape, strict=True):
                raw = owned_index(form, source)
                value = raw
                bound = abi.dimension(dimension)
                if (form + (bound.scale(-1) if isinstance(bound, Linear) else Linear(-bound))).bounds(grid)[1] > 0:
                    limit = dimensions[dimension] if isinstance(dimension, str) else index_constant(dimension, source)
                    value = append(arith.MinSIOp(value, limit), source).result
                offsets.append(Coordinate(form, value))
                if isinstance(size, TailExtent):
                    limit = owned_index(size.limit, source) if isinstance(size.limit, Linear) else index_constant(size.limit, source)
                    origin = owned_index(size.offset, source)
                    remaining = append(arith.SubiOp(limit, origin), source).result
                    extent = append(arith.MinSIOp(index_constant(size.cap, source), remaining), source).result
                    extent = append(arith.MaxSIOp(index_constant(0, source), extent), source).result
                    size = TailExtent(size.cap, size.limit, size.offset, extent)
                published.append(size)
            active = tuple(published)
            tile = append(slice_op(states[buffer.name], offsets, active), source).result
            updates.add_op(slice_op(tile, offsets, active, destination=shared[buffer.name], parallel_insert=True))
        body.add_op(InParallelOp.build(regions=[Region(updates)]))
        dynamic_launch = any(isinstance(g, GridExtent) for g in abi.grid)
        upper = shape_product(launch, lambda x: shape_emit(outer, x, function.location, "logical-grid-product")) if dynamic_launch else None
        forall = ForallOp.build(operands=[[], [upper] if dynamic_launch else [], [], initial], result_types=[[v.type for v in initial]],
                  properties={"staticLowerBound": DenseArrayBase.from_list(builtin.i64, [0]),
                              "staticUpperBound": DenseArrayBase.from_list(builtin.i64, [DYNAMIC_INDEX if dynamic_launch else host_math.prod(grid)]),
                              "staticStep": DenseArrayBase.from_list(builtin.i64, [1])}, regions=[Region(body)])
        forall.attributes.update({"prov.family": StringAttr("grid"), "prov.op": StringAttr("logical_launch"),
                                  "prov.logical_grid_bounds" if abi.dynamic else "prov.logical_grid": DenseArrayBase.from_list(builtin.i64, grid)})
        outer.add_ops([forall, func.ReturnOp(*forall.results)])
    evidence = initialization.finish(source_id=next(iter(_source_ids(function.location)), None))
    output = func.FuncOp(abi.entry, ([b.type for b in abi.inputs], [b.type for b in writable]), Region(outer))
    output.attributes.update({"prov.family": StringAttr("triton"), "prov.op": StringAttr("kernel"),
                              "prov.intent": StringAttr(abi.intent),
                              "prov.intent_origin": StringAttr("caller-description"),
                              "prov.numeric_profile": StringAttr("upstream-standard-expansion")})
    output.attributes["prov.logical_grid_bounds" if abi.dynamic else "prov.logical_grid"] = DenseArrayBase.from_list(builtin.i64, grid)
    output.attributes["prov.source_loops"] = StringAttr(json.dumps(source_loops, sort_keys=True))
    output.attributes["prov.source_conditionals"] = StringAttr(json.dumps(source_conditionals, sort_keys=True))
    output.attributes["prov.source_gathers"] = StringAttr(json.dumps(source_gathers, sort_keys=True))
    output.attributes["prov.source_scatters"] = StringAttr(json.dumps(source_scatters, sort_keys=True))
    if abi.runtime_scalars:
        output.attributes["prov.runtime_scalars"] = StringAttr(json.dumps([s.as_dict() for s in abi.runtime_scalars], sort_keys=True))
    output.attributes["prov.output_initialization"] = StringAttr(json.dumps(evidence, sort_keys=True))
    if parallel:
        output.attributes["prov.output_ownership"] = StringAttr(json.dumps({name: {
            "offsets": [asdict(off) for off in info[0][0]], "tile_sizes": [describe(n) for n in info[0][1]],
            "write_independence": "empty-access-domain" if any(maximum(n) == 0 for n in info[0][1]) else "distinct-axis-affine-tile-strides",
            "readwrite_reads": "owned-tile-only"} for name, info in writes.items()}, sort_keys=True))
    result = ModuleOp([output])
    result.attributes.update({"prov.level": StringAttr("linalg-on-tensors"),
                              "prov.frontend": StringAttr("triton2mlir.shared")})
    for op in result.walk():
        if op.name == "builtin.unregistered" or opcode(op).split(".")[0] not in ("builtin", "func", "arith", "math", "tensor", "linalg", "scf", "cf"):
            raise ConversionError("nonstandard operation survived tensor functionalization", opcode=opcode(op))
    result.verify()
    return result


@dataclass
class SharedTensorResult:
    module: object
    spec: TensorABI
    trace: dict
    source_ttir: str
    auxiliary_artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def mlir_text(self):
        return text(self.module)


def convert_shared(source, abi, work, *, optimizer=None, memory_profile="tensor"):
    """Reuse actual upstream passes, then functionalize qualified tensor memory.

    ``work`` is a new diagnostic directory, retained on both success and refusal.
    The returned standard module is experimental until numerically qualified;
    native verification and consumer execution are separately measured statuses.
    """
    if memory_profile not in ("tensor", "buffer", "guarded-buffer"):
        raise ValueError("memory profile must be explicitly tensor, buffer or guarded-buffer")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    optimizer = inspect_optimizer(optimizer)
    source_text = Path(source).read_bytes().decode("utf-8")
    pointer_report, original_inventory = None, None
    try:
        intake = lower_with_shared(source, work / "source", optimizer=optimizer, stage="source")
        source_ir = work / "source/lowered.mlir"
    except ConversionError as original_error:
        # Some shared distributions removed legacy block-pointer operations.
        # The pinned Triton frontend still verifies and expands them officially.
        try:
            pointer_report = expand_tensor_pointers(source, work / "tensor-pointers")
        except ConversionError:
            raise original_error
        original_inventory = json.loads((work / "tensor-pointers/inventory.json").read_text())
        if pointer_report["source_sha256"] != hashlib.sha256(source_text.encode()).hexdigest():
            raise ConversionError("original source changed during tensor-pointer normalization")
        intake = lower_with_shared(work / "tensor-pointers/lowered.mlir", work / "expanded-source", optimizer=optimizer, stage="source")
        source_ir = work / "expanded-source/lowered.mlir"
    parsed = parse_shared(source_ir.read_text())
    records = _preflight(parsed, abi, id_prefix="normalized.ttir.op." if original_inventory else "ttir.op.")
    annotated = work / "annotated.ttir"
    # Name locations survive the upstream rewrite infrastructure; attributes on
    # replaced source ops generally do not. Preserve the exact original too.
    from io import StringIO
    from xdsl.printer import Printer
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True, print_debuginfo=True).print_op(parsed)
    annotated.write_text(stream.getvalue() + "\n")
    lower_with_shared(annotated, work / "inline", optimizer=optimizer, stage="inline")
    middle_source = work / "inline/lowered.mlir"
    if not pointer_report and any(r["opcode"] in DESCRIPTOR_OPS for r in records):
        pointer_report = expand_tensor_pointers(middle_source, work / "tensor-pointers")
        # Native generic/local-scope printing resolves debug-location aliases
        # for xDSL, which cannot read Triton's forward location aliases safely.
        lower_with_shared(work / "tensor-pointers/lowered.mlir", work / "expanded-source", optimizer=optimizer, stage="source")
        middle_source = work / "expanded-source/lowered.mlir"
    middle_module = parse_shared(middle_source.read_text())
    check_source_addresses(middle_module, abi)
    scan_reports = expand_scans(middle_module, _source_ids)
    if scan_reports:
        stream = StringIO()
        Printer(stream=stream, print_generic_format=True, print_debuginfo=True).print_op(middle_module)
        middle_source = work / "expanded-scans.ttir"
        middle_source.write_text(stream.getvalue() + "\n")
    stage = lower_with_shared(middle_source, work / "tensor-ir", optimizer=optimizer, stage="tensor-ir")
    lowered = parse_shared((work / "tensor-ir/lowered.mlir").read_text())
    if len(list(lowered.ops)) != 1 or opcode(next(iter(lowered.ops))) != "tt.func":
        raise ConversionError("upstream left helper functions; complete inlining is required for tensorization")
    transformations = _restore_accumulators(lowered, records)
    transformations.extend(scan_reports)
    if pointer_report:
        transformations.append({**pointer_report, "source_ids": [r["id"] for r in
                               (original_inventory["operations"] if original_inventory else records)
                               if r["opcode"] in DESCRIPTOR_OPS],
                               "source_descriptor_intent": "retained-in-exact-original-source-and-typed-inventory"})
    admission = None
    if memory_profile == "guarded-buffer":
        from .guarded_memory import instrument
        admission = instrument(lowered, abi, _source_ids)
        result = witness = None
    else:
        result = witness = _functionalize(lowered, abi)
    auxiliary = {}
    memory_stage = None
    if memory_profile in ("buffer", "guarded-buffer"):
        from .buffer_memory import functionalize_memory, parse_memory
        witness_text = text(witness) if witness is not None else None
        stream = StringIO()
        Printer(stream=stream, print_generic_format=True, print_debuginfo=True).print_op(lowered)
        memory_input = stream.getvalue() + "\n"
        (work / "memory-input.mlir").write_text(memory_input)
        memory_stage = lower_with_shared(work / "memory-input.mlir", work / "memory-ir", optimizer=optimizer,
            stage="memory-effects" if admission is not None else "memory-ir")
        memory_source = (work / "memory-ir/lowered.mlir").read_text()
        memory_module = parse_memory(memory_source)
        if admission is not None:
            from .guarded_reads import materialize
            admission["read_adaptations"] = materialize(memory_module, abi, _source_ids)
            if admission["read_adaptations"]:
                transformations.append({"kind": "source-gather-observations-to-memory-loads",
                    "adaptations": admission["read_adaptations"],
                    "boundary": "before-memory-CSE; original-source-read-effects-retained"})
        result = functionalize_memory(memory_module, abi, guards=admission)
        auxiliary = {"memory-input.mlir": memory_input,
                     "memory-stage.mlir": memory_source,
                     "memory-stage.json": json.dumps(memory_stage, indent=2, sort_keys=True) + "\n"}
        if admission is not None:
            admission.update(guarded_input_sha256=hashlib.sha256(memory_input.encode()).hexdigest(),
                             continuation_output_sha256=hashlib.sha256(memory_source.encode()).hexdigest())
            auxiliary["memory-admission.json"] = json.dumps(admission, indent=2, sort_keys=True) + "\n"
        else:
            auxiliary["memory-witness.mlir"] = witness_text
        transformations.append({"kind": "functional-standard-memory-boundary",
                               "semantic_gate": "executable-guarded-memory-admission" if admission is not None else "existing-source-tensor-initialization-bounds-and-ownership-analysis",
                               "launcher": "internal-counts-xyz-then-ids-xyz; public-tensor-scalar-ABI"})
    emitted = []
    source_names = {r["id"]: r["opcode"] for r in records}
    for op in result.walk():
        if memory_profile in ("buffer", "guarded-buffer"):
            op.attributes.setdefault("prov.family", StringAttr("triton"))
            op.attributes.setdefault("prov.op", StringAttr(op.name.split(".")[-1]))
        ids = _source_ids(op.location)
        if ids:
            op.attributes["prov.source_ids"] = ArrayAttr(StringAttr(s) for s in ids)
            op.attributes["prov.original_ops"] = ArrayAttr(StringAttr(source_names[s]) for s in ids)
        op.attributes["prov.region_id"] = StringAttr(f"kernel:{abi.entry}")
        op.attributes["prov.emitted_id"] = StringAttr(f"mlir.op.{len(emitted):04d}")
        emitted.append({"id": f"mlir.op.{len(emitted):04d}", "opcode": opcode(op), "source_ids": ids})
    if original_inventory:
        # Expanded operations are a distinct graph, never the original TTIR.
        for op in result.walk():
            if "prov.source_ids" in op.attributes:
                op.attributes["prov.normalized_ids"] = op.attributes.pop("prov.source_ids")
            if "prov.original_ops" in op.attributes:
                op.attributes["prov.normalized_ops"] = op.attributes.pop("prov.original_ops")
        for record in emitted:
            record["normalized_ids"] = record.pop("source_ids")
        result.attributes["prov.original_source_sha256"] = StringAttr(hashlib.sha256(source_text.encode()).hexdigest())
    trace = {"schema": "t2m.shared-tensor-provenance.v1", "source_operations": records,
             "emitted_operations": emitted,
             "transformations": transformations,
             "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
             "mlir_sha256": hashlib.sha256(text(result).encode()).hexdigest(),
             "spec": abi.as_dict(), "optimizer": intake["optimizer"],
             "intent": {"description": abi.intent, "origin": "caller-description", "grants_semantics": False},
             "pipeline": stage["command"][1:-1],
             "correspondence": "upstream-preserved-source-name-locations; missing descendants are not independent dead-code proofs",
             "memory": {"bounds": "checked-affine-domain-clipped-rectangular-accesses", "ordering": "functional-buffer-versions",
                        "index_overflow": "checked-source-address-widths-before-index-promotion",
                        "aliasing": "independent-storage-ABI-precondition",
                        "grid": "unmapped-logical-forall-with-owned-tiles" if host_math.prod(abi.grid_bounds) > 1 or any(isinstance(g, GridExtent) for g in abi.grid) else "single-logical-program"},
             "numeric_equivalence": "requires-qualification-against-source",
             "native_verification": "not-performed", "consumer_execution": "not-performed",
             "abi_difference": "readwrite buffers are incoming tensors; write-only outputs omit inputs only with complete initialization checks"}
    if original_inventory:
        trace.update(schema="t2m.shared-tensor-provenance.v2", normalized_operations=records,
                     source_operations=original_inventory["operations"], original_blocks=original_inventory["blocks"],
                     original_attributes=original_inventory["attributes"],
                     correspondence="original-native-verified-typed-graph; emissions-and-memory-facts-reference-normalized-graph; original-source-locations-retained")
    entry = next(iter(witness.ops)) if witness is not None else None
    trace["memory"]["source_loops"] = json.loads(entry.attributes["prov.source_loops"].data) if entry else [r for r in admission["control_flow"] if r["opcode"] != "scf.if"]
    trace["memory"]["source_conditionals"] = json.loads(entry.attributes["prov.source_conditionals"].data) if entry else [r for r in admission["control_flow"] if r["opcode"] == "scf.if"]
    trace["memory"]["source_gathers"] = json.loads(entry.attributes["prov.source_gathers"].data) if entry else [r for r in admission["accesses"] if r["opcode"] == "tts.gather"]
    trace["memory"]["source_scatters"] = json.loads(entry.attributes["prov.source_scatters"].data) if entry else [r for r in admission["accesses"] if r["opcode"] == "tts.scatter"]
    if admission is not None:
        trace["memory"]["source_accesses"] = admission["accesses"]
    if trace["memory"]["source_gathers"]:
        trace["memory"]["bounds"] += "; runtime-active-gather-bounds"
    if abi.runtime_scalars:
        trace["abi_difference"] += "; runtime scalar inputs follow logical tensor inputs and retain plain by-value types/source positions"
    trace["memory"]["output_initialization"] = json.loads(entry.attributes["prov.output_initialization"].data) if entry else {
        b.name: {"initial_value_required": b.effect == "readwrite", "coverage": "runtime-complete-write-check" if b.effect == "write" else "incoming-state-preserved"} for b in abi.outputs}
    if entry is not None and "prov.output_ownership" in entry.attributes:
        trace["memory"]["output_ownership"] = json.loads(entry.attributes["prov.output_ownership"].data)
    if memory_stage is not None:
        trace["profile"] = "shared-guarded-buffer" if admission is not None else "shared-functional-buffer"
        trace["memory"].update(representation="standard-memory-inside-functional-tensor-ABI",
            continuation_input_sha256=hashlib.sha256(auxiliary["memory-input.mlir"].encode()).hexdigest(),
            continuation_output_sha256=hashlib.sha256(auxiliary["memory-stage.mlir"].encode()).hexdigest(),
            continuation_pipeline=memory_stage["command"][1:-1],
            semantic_gate="source tensor memory analysis; no additional opcode coverage implied")
        if admission is not None:
            trace["memory"].update(admission_sha256=hashlib.sha256(auxiliary["memory-admission.json"].encode()).hexdigest(),
                bounds="executable-active-flat-index-checks", ordering="source-program-order; private-output-state; reads-observe-completed-writes",
                grid="unmapped-logical-forall-with-runtime-write-ownership-and-lane-uniqueness",
                semantic_gate="independent-runtime-memory-admission; source numeric policy remains separate")
        else:
            trace["memory"]["witness_sha256"] = hashlib.sha256(auxiliary["memory-witness.mlir"].encode()).hexdigest()
        trace["abi_difference"] += "; mixed standard tensor/memref payload; writable state privately allocated and readwrite inputs copied"
    return SharedTensorResult(result, abi, trace, source_text, auxiliary)
