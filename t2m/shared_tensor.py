"""Functionalize triton-shared's tensor stage, without its memory/runtime ABI.

Upstream owns TTIR arithmetic and region decomposition. This adapter owns the
declared logical tensor ABI, ordered memory versions, bounds, and provenance.
The initial profile requires one logical program and statically resolved
structured descriptors. Unsupported memory/control-flow forms fail closed.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math as host_math
from pathlib import Path

from xdsl.dialects import arith, builtin, cf, func, math, scf, tensor
from xdsl.dialects.builtin import (ArrayAttr, DenseArrayBase, DYNAMIC_INDEX, IndexType,
                                  IntegerAttr, ModuleOp, NameLoc, StringAttr, TensorType)
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.parser import Parser

from .diagnostics import ConversionError
from .ir import context, text
from .upstream import inspect_optimizer, lower_with_shared

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
    return SharedParser(ctx, source).parse_module()


@dataclass(frozen=True)
class TensorBuffer:
    name: str
    shape: tuple[int, ...]
    effect: str = "read"
    dtype: str = "f32"

    def __post_init__(self):
        if not self.name.isidentifier() or not self.shape or any(type(n) is not int or n <= 0 for n in self.shape):
            raise ValueError("tensor buffers require a name and positive static dimensions")
        if self.dtype not in DTYPES or self.effect not in ("read", "readwrite"):
            raise ValueError("tensor buffers require a supported dtype and read/readwrite effect")

    @property
    def type(self):
        return TensorType(DTYPES[self.dtype], self.shape)

    @property
    def strides(self):
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
    grid: tuple[int, ...] = (1,)
    intent: str = ""

    def __post_init__(self):
        if not self.entry.isidentifier() or not self.buffers:
            raise ValueError("tensor ABI requires an entry and buffers")
        names = [b.name for b in self.buffers]
        if len(set(names)) != len(names) or not any(b.effect == "readwrite" for b in self.buffers):
            raise ValueError("tensor ABI requires unique names and at least one readwrite result")
        if not 1 <= len(self.grid) <= 3 or any(type(n) is not int or n != 1 for n in self.grid):
            raise ValueError("this tensor profile requires a single logical program; grids are not serialized")
        if any(type(n) is not int for n in self.scalar_constants):
            raise ValueError("specialized scalar arguments must be integers")

    def as_dict(self):
        return {"schema": "t2m.shared-tensor-abi.v1", **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        fields = dict(value)
        if fields.pop("schema", "t2m.shared-tensor-abi.v1") != "t2m.shared-tensor-abi.v1":
            raise ValueError("unknown shared tensor ABI schema")
        fields["buffers"] = tuple(TensorBuffer(**{**b, "shape": tuple(b["shape"])}) for b in fields["buffers"])
        for name in ("grid", "scalar_constants"):
            if name in fields:
                fields[name] = tuple(fields[name])
        return cls(**fields)


def _array(op, name):
    value = op.properties.get(name, op.attributes.get(name))
    if not isinstance(value, DenseArrayBase):
        raise ConversionError(f"expected structured {name} array", opcode=opcode(op))
    return tuple(value.get_values())


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
        return [name] if name.startswith("ttir.op.") else _source_ids(location.location)
    if isinstance(location, builtin.FusedLoc):
        return list(dict.fromkeys(s for loc in location.locations for s in _source_ids(loc)))
    if isinstance(location, builtin.CallSiteLoc):
        return list(dict.fromkeys(_source_ids(location.callee) + _source_ids(location.caller)))
    return []


def _preflight(module, abi):
    functions = [op for op in module.ops if opcode(op) == "tt.func"]
    entries = [op for op in functions if op.properties["sym_name"].data == abi.entry]
    if len(entries) != 1 or len(list(module.ops)) != len(functions):
        raise ConversionError("expected the declared TTIR entry and optional private helpers")
    function = entries[0]
    if any(op is not function and op.properties.get("sym_visibility") != StringAttr("private") for op in functions):
        raise ConversionError("additional public entry functions are unsupported")
    if len(function.regions[0].blocks) != 1:
        raise ConversionError("multiple entry blocks need an explicit CFG adapter")
    args = function.regions[0].block.args
    if len(args) != len(abi.buffers) + len(abi.scalar_constants):
        raise ConversionError("source arguments differ from buffers followed by specialized scalars")
    for arg, buffer in zip(args, abi.buffers):
        if str(arg.type) != f"!tt.ptr<{buffer.dtype}>":
            raise ConversionError(f"pointer type differs from tensor ABI for {buffer.name}")
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
        source_id, name = f"ttir.op.{len(records):04d}", opcode(op)
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
    body = Block(arg_types=[b.type for b in abi.buffers])
    mapping, buffers, descriptors = {}, {}, {}
    states = {b.name: arg for b, arg in zip(abi.buffers, body.args, strict=True)}
    for arg, buffer in zip(function.regions[0].block.args, abi.buffers):
        buffers[arg] = buffer
    scalar_args = function.regions[0].block.args[len(abi.buffers):]
    for arg, value in zip(scalar_args, abi.scalar_constants, strict=True):
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

    def resolve(op, field, operands):
        dynamic = iter(operands)
        result = []
        for number in _array(op, field):
            if number == DYNAMIC_INDEX:
                value = next(dynamic, None)
                if value is None:
                    raise ConversionError("structured descriptor is missing a dynamic operand", opcode=opcode(op))
                number = _integer(mapping.get(value, value))
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
        strides = resolve(op, "static_strides", slices[0])
        offsets = resolve(op, "static_offsets", slices[1])
        shape = resolve(op, "static_shape", slices[2])
        if any(shape) or _array(op, "order"):
            raise ConversionError("wraparound/block-pointer bounds need a qualified tensor view adapter", opcode=opcode(op))
        buffer, sizes = buffers[base], _array(op, "sizes")
        if len(sizes) != len(buffer.shape) or strides != buffer.strides:
            raise ConversionError("descriptor must match the declared logical rank and contiguous strides", opcode=opcode(op))
        if any(off % stride for off, stride in zip(offsets, strides, strict=True)):
            raise ConversionError("element offsets do not align with declared logical axes")
        return buffer, sizes, tuple(off // stride for off, stride in zip(offsets, strides, strict=True))

    def access(op, desc, store=False):
        buffer, sizes, offsets = desc
        mask = _array(op, "static_mask_dims")
        dynamic = op.operands[2:] if store else op.operands[1:1 + _array(op, "operandSegmentSizes")[1]]
        active = resolve(op, "static_mask_dims", dynamic) if mask else sizes
        if len(active) != len(sizes) or any(n < 0 or n > size for n, size in zip(active, sizes, strict=True)):
            raise ConversionError("mask must describe a bounded rectangular prefix", opcode=opcode(op))
        if any(off < 0 or (n and off + n > extent) for off, n, extent in zip(offsets, active, buffer.shape, strict=True)):
            raise ConversionError("active structured access exceeds the declared tensor bounds", opcode=opcode(op))
        # Empty accesses make no memory observation; keep their standard slice
        # coordinates within the logical tensor as well.
        if any(n == 0 for n in active):
            offsets = tuple(min(max(off, 0), extent) for off, extent in zip(offsets, buffer.shape, strict=True))
        return buffer, sizes, offsets, active

    for op in function.regions[0].block.ops:
        name = opcode(op)
        if name == "bufferization.alloc_tensor":
            if op.operands or len(op.results) != 1 or not isinstance(op.results[0].type, TensorType) or DYNAMIC_INDEX in op.results[0].type.get_shape():
                raise ConversionError("only static, copy-free allocation tensors can become tensor.empty", opcode=name)
            mapping[op.results[0]] = append(tensor.EmptyOp([], op.results[0].type), op).tensor
        elif name in ("tt.get_program_id", "tt.get_num_programs"):
            constant = append(arith.ConstantOp(IntegerAttr(0 if name.endswith("program_id") else 1, op.results[0].type)), op)
            mapping[op.results[0]] = constant.result
        elif name == "tts.make_tptr":
            descriptors[op.results[0]] = descriptor(op)
        elif name == "tts.load":
            if op.operands[0] not in descriptors:
                raise ConversionError("load pointer is not a qualified structured descriptor", opcode=name)
            buffer, sizes, offsets, active = access(op, descriptors[op.operands[0]])
            loaded = append(tensor.ExtractSliceOp.from_static_parameters(states[buffer.name], offsets, active), op).result
            if active != sizes:
                segments = _array(op, "operandSegmentSizes")
                if segments[2] != 1:
                    raise ConversionError("masked load without other needs defined-lane propagation", opcode=name)
                fill = mapping.get(op.operands[-1])
                if fill is None:
                    raise ConversionError("masked load fill is not a translated scalar", opcode=name)
                padding = Block(arg_types=[IndexType()] * len(sizes))
                padding.add_op(tensor.YieldOp(fill))
                loaded = append(tensor.PadOp(loaded, [], [], Region(padding), [0] * len(sizes),
                                [size - n for size, n in zip(sizes, active, strict=True)],
                                TensorType(buffer.type.get_element_type(), sizes)), op).result
            mapping[op.results[0]] = loaded
        elif name == "tts.store":
            if op.operands[0] not in descriptors:
                raise ConversionError("store pointer is not a qualified structured descriptor", opcode=name)
            buffer, sizes, offsets, active = access(op, descriptors[op.operands[0]], store=True)
            if buffer.effect != "readwrite":
                raise ConversionError("source stores into a declared read-only buffer", opcode=name)
            value = mapping[op.operands[1]]
            if active != sizes:
                value = append(tensor.ExtractSliceOp.from_static_parameters(value, [0] * len(sizes), active), op).result
            states[buffer.name] = append(tensor.InsertSliceOp.from_static_parameters(value, states[buffer.name], offsets, active), op).result
        elif name == "tt.return":
            append(func.ReturnOp(*(states[b.name] for b in abi.buffers if b.effect == "readwrite")), op)
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
    output = func.FuncOp(abi.entry, ([b.type for b in abi.buffers], [b.type for b in abi.buffers if b.effect == "readwrite"]), Region(body))
    output.attributes.update({"prov.family": StringAttr("triton"), "prov.op": StringAttr("kernel"),
                              "prov.intent": StringAttr(abi.intent),
                              "prov.numeric_profile": StringAttr("upstream-standard-expansion")})
    result = ModuleOp([output])
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

    @property
    def mlir_text(self):
        return text(self.module)


def convert_shared(source, abi, work, *, optimizer=None):
    """Reuse actual upstream passes, then functionalize qualified tensor memory.

    ``work`` is a new diagnostic directory, retained on both success and refusal.
    The returned standard module is experimental until numerically qualified;
    native verification and consumer execution are separately measured statuses.
    """
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    optimizer = inspect_optimizer(optimizer)
    source_text = Path(source).read_bytes().decode("utf-8")
    intake = lower_with_shared(source, work / "source", optimizer=optimizer, stage="source")
    parsed = parse_shared((work / "source/lowered.mlir").read_text())
    records = _preflight(parsed, abi)
    annotated = work / "annotated.ttir"
    # Name locations survive the upstream rewrite infrastructure; attributes on
    # replaced source ops generally do not. Preserve the exact original too.
    from io import StringIO
    from xdsl.printer import Printer
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True, print_debuginfo=True).print_op(parsed)
    annotated.write_text(stream.getvalue() + "\n")
    lower_with_shared(annotated, work / "inline", optimizer=optimizer, stage="inline")
    stage = lower_with_shared(work / "inline/lowered.mlir", work / "tensor-ir", optimizer=optimizer, stage="tensor-ir")
    lowered = parse_shared((work / "tensor-ir/lowered.mlir").read_text())
    if len(list(lowered.ops)) != 1 or opcode(next(iter(lowered.ops))) != "tt.func":
        raise ConversionError("upstream left helper functions; complete inlining is required for tensorization")
    transformations = _restore_accumulators(lowered, records)
    result = _functionalize(lowered, abi)
    emitted = []
    source_names = {r["id"]: r["opcode"] for r in records}
    for op in result.walk():
        ids = _source_ids(op.location)
        if ids:
            op.attributes["prov.source_ids"] = ArrayAttr(StringAttr(s) for s in ids)
            op.attributes["prov.original_ops"] = ArrayAttr(StringAttr(source_names[s]) for s in ids)
        op.attributes["prov.region_id"] = StringAttr(f"kernel:{abi.entry}")
        op.attributes["prov.emitted_id"] = StringAttr(f"mlir.op.{len(emitted):04d}")
        emitted.append({"id": f"mlir.op.{len(emitted):04d}", "opcode": opcode(op), "source_ids": ids})
    trace = {"schema": "t2m.shared-tensor-provenance.v1", "source_operations": records,
             "emitted_operations": emitted,
             "transformations": transformations,
             "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
             "mlir_sha256": hashlib.sha256(text(result).encode()).hexdigest(),
             "spec": abi.as_dict(), "optimizer": intake["optimizer"],
             "pipeline": stage["command"][1:-1],
             "correspondence": "upstream-preserved-source-name-locations; missing descendants are not independent dead-code proofs",
             "memory": {"bounds": "checked-static-rectangular-accesses", "ordering": "functional-buffer-versions",
                        "aliasing": "independent-storage-ABI-precondition", "grid": "single-logical-program"},
             "numeric_equivalence": "requires-qualification-against-source",
             "native_verification": "not-performed", "consumer_execution": "not-performed",
             "abi_difference": "all writable buffers are incoming tensors and returned values; no implicit uninitialized outputs"}
    return SharedTensorResult(result, abi, trace, source_text)
