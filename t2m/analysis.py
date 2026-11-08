"""Symbolic tensor access recovery. Proof cost depends on IR, not tensor size."""

from dataclasses import asdict, dataclass
import struct

from .diagnostics import ConversionError
from .indexing import Index, Mask, comparison

PAYLOAD = {"arith.addf": "add", "arith.subf": "sub", "arith.mulf": "mul"}
PURE = {"arith.constant", "tt.get_program_id", "tt.make_range", "tt.splat", "tt.addptr",
        "arith.addi", "arith.subi", "arith.muli", "arith.extsi", "arith.extui",
        "arith.cmpi", "arith.andi", "arith.ori", *PAYLOAD}
SUPPORTED = PURE | {"tt.load", "tt.store", "tt.return", "tt.func", "builtin.module"}


@dataclass(frozen=True)
class Pointer:
    buffer: str
    offset: Index = Index()


@dataclass(frozen=True)
class Float:
    bits: int
    width: int = 0

    @property
    def value(self):
        return struct.unpack("<f", struct.pack("<I", self.bits))[0]


@dataclass(frozen=True)
class Tile:
    source_id: str
    width: int
    defined: Mask = Mask()


@dataclass(frozen=True)
class Access:
    pointer: Pointer
    mask: Mask = Mask()
    fill: Float | None = None

    @property
    def width(self):
        return self.pointer.offset.width


@dataclass
class Analyzed:
    ttir: object
    spec: object
    records: list[dict]
    values: dict
    actions: list[tuple]
    output: tuple
    tile_width: int


def _width(type_):
    text = str(type_)
    if not text.startswith("tensor<"):
        return 0
    first, _, element = text[7:-1].partition("x")
    if not first.isdigit() or "x" in element:
        raise ValueError("currently requires one-dimensional static tiles")
    return int(first)


def _logical(pointer, buffer):
    """Invert the declared storage view without guessing contiguous physical memory."""
    offset, stride = pointer.offset, buffer.strides[0]
    coefficients = (offset.constant - buffer.storage_offset, offset.program, offset.lane)
    if any(coefficient % stride for coefficient in coefficients):
        raise ValueError("access is not aligned with the declared strided logical view")
    return Pointer(pointer.buffer, Index(*(coefficient // stride for coefficient in coefficients),
                                         offset.width))


def _valid_domain(offset, size):
    # 0 <= base(p) + lane < size.
    return Mask((Index(-offset.constant, -offset.program),),
                (Index(size - offset.constant, -offset.program),))


def analyze(ttir, spec):
    ops = ttir.operations
    records = [{"id": f"ttir.op.{i:04d}", "opcode": op.get_name(), "status": "pending"}
               for i, op in enumerate(ops)]
    stable_values = {arg.id(): f"arg.{i}" for i, arg in enumerate(ttir.entry_args)}
    stable_values.update({op.get_result(j).id(): f"{records[i]['id']}.result.{j}"
                          for i, op in enumerate(ops) for j in range(op.get_num_results())})
    for op, record in zip(ops, records, strict=True):
        record.update(operands=[stable_values[op.get_operand(j).id()] for j in range(op.get_num_operands())],
                      results=[stable_values[op.get_result(j).id()] for j in range(op.get_num_results())],
                      result_types=[str(op.get_result(j).get_type()) for j in range(op.get_num_results())])

    def fail(index, message):
        raise ConversionError(message, source_id=records[index]["id"], opcode=ops[index].get_name())

    for i, op in enumerate(ops):
        if op.get_name() not in SUPPORTED:
            fail(i, "operation is outside the implemented tensor-memory subset")
    if "fastmath" in ttir.text or "overflow<" in ttir.text:
        raise ConversionError("explicit arithmetic flags are outside the current numeric profile")
    definitions = {op.get_result(j).id(): i for i, op in enumerate(ops) for j in range(op.get_num_results())}
    live = {i for i, op in enumerate(ops) if op.get_name() not in PURE}
    pending = list(live)
    while pending:
        i = pending.pop()
        for j in range(ops[i].get_num_operands()):
            producer = definitions.get(ops[i].get_operand(j).id())
            if producer is not None and producer not in live:
                live.add(producer)
                pending.append(producer)
    env = {arg.id(): Pointer(buffer.name) for arg, buffer in zip(ttir.entry_args, spec.buffers, strict=True)}
    buffers = {b.name: b for b in spec.buffers}
    actions, stores, widths, reads = [], [], set(), []
    for i, op in enumerate(ops):
        name, record = op.get_name(), records[i]
        if i not in live:
            record.update(status="eliminated", reason="unused pure value (backward liveness fixpoint)")
            continue
        value = None
        record.update(status="analyzed", reason="recovered index/access semantics")
        try:
            args = [env[op.get_operand(j).id()] for j in range(op.get_num_operands())]
            if name in ("tt.func", "builtin.module"):
                record.update(status="structural", reason="function/module boundary re-expressed")
            elif name == "tt.return":
                if args:
                    fail(i, "TTIR return values are not supported")
                record.update(status="structural", reason="stores become tensor results")
            elif name == "arith.constant":
                decoded = ttir.constants.get(op.get_result(0).id())
                if decoded is None:
                    fail(i, "live constant is not a supported scalar/splat literal")
                kind, number = decoded
                width = _width(op.get_result(0).get_type())
                if kind == "f32" and number & 0x7f800000 == 0x7f800000 and number & 0x007fffff:
                    fail(i, "NaN constants require exact payload-preserving literal support")
                value = (Float(number, width) if kind == "f32" else
                         (Mask() if number else Mask.false()) if kind == "i1" else
                         Index(number, width=width))
            elif name == "tt.get_program_id":
                if op.get_int_attr("axis") != 0:
                    fail(i, "only logical program axis zero is supported")
                value = Index(program=1)
            elif name == "tt.make_range":
                start, end = op.get_int_attr("start"), op.get_int_attr("end")
                if start is None or end is None or end <= start:
                    fail(i, "range must have static increasing bounds")
                value = Index(start, lane=1, width=end - start)
            elif name == "tt.splat":
                width = _width(op.get_result(0).get_type())
                if isinstance(args[0], (Pointer, Mask)):
                    value = args[0]
                elif isinstance(args[0], Index) and not args[0].width:
                    a = args[0]
                    value = Index(a.constant, a.program, a.lane, width)
                elif isinstance(args[0], Float) and not args[0].width:
                    value = Float(args[0].bits, width)
                else:
                    fail(i, "splat operand is not a supported uniform value")
            elif name in ("arith.addi", "arith.subi", "arith.muli"):
                a, b = args
                if not isinstance(a, Index) or not isinstance(b, Index):
                    fail(i, "integer payload is outside the f32 frontend")
                if name == "arith.muli":
                    if b.program == b.lane == 0:
                        value = a.scale(b.constant)
                    elif a.program == a.lane == 0:
                        value = b.scale(a.constant)
                    else:
                        fail(i, "product of varying indices is not affine")
                else:
                    value = a.combine(b, -1 if name == "arith.subi" else 1)
            elif name in ("arith.extsi", "arith.extui"):
                value = args[0]
                if not isinstance(value, Index):
                    fail(i, "data casts are not implemented")
                if name == "arith.extui" and value.bounds(spec.grid[0])[0] < 0:
                    fail(i, "zero extension of a negative index is not a signed affine identity")
            elif name == "arith.cmpi":
                value = comparison(op.get_int_attr("predicate"), *args, spec.grid[0])
            elif name == "arith.andi" and all(isinstance(arg, Mask) for arg in args):
                value = args[0].intersect(args[1])
            elif name == "tt.addptr":
                pointer, offset = args
                if not isinstance(pointer, Pointer) or not isinstance(offset, Index):
                    fail(i, "unproved pointer arithmetic")
                value = Pointer(pointer.buffer, pointer.offset.combine(offset))
            elif name in ("tt.load", "tt.store"):
                if op.get_bool_attr("isVolatile"):
                    fail(i, "volatile memory cannot become pure tensor dataflow")
                if not isinstance(args[0], Pointer):
                    fail(i, "unresolved pointer")
                buffer = buffers[args[0].buffer]
                pointer = _logical(args[0], buffer)
                offset, programs = pointer.offset, spec.grid[0]
                if offset.lane != 1 or not offset.width:
                    fail(i, "only injective contiguous logical tile accesses are implemented")
                is_load = name == "tt.load"
                if (is_load and len(args) not in (1, 2, 3)) or (not is_load and len(args) not in (2, 3)):
                    fail(i, "unsupported memory operand form")
                mask_index = 1 if is_load else 2
                mask = args[mask_index] if len(args) > mask_index else Mask()
                if not isinstance(mask, Mask):
                    fail(i, "memory mask is not a proved rectangular predicate")
                valid = _valid_domain(offset, buffer.shape[0])
                if not mask.subset_of(valid, offset.width, programs):
                    low, high = offset.bounds(programs)
                    fail(i, f"active access may be outside {buffer.name}[0:{buffer.shape[0]}]; "
                         f"unmasked address range [{low}, {high}]")
                fill = args[2] if is_load and len(args) == 3 else None
                if fill is not None and not isinstance(fill, Float):
                    fail(i, "masked other must currently be a defined f32 scalar/splat")
                access = Access(pointer, mask, fill)
                widths.add(offset.width)
                record["access"] = {**asdict(access), "effect": buffer.effect,
                                    "bounds_proof": "mask-implies-logical-view-bounds",
                                    "physical_view": {"stride": buffer.strides[0],
                                                      "storage_offset": buffer.storage_offset}}
                record["advisory"] = {"cache": op.get_int_attr("cache"),
                                      "evict": op.get_int_attr("evict"),
                                      "disposition": "source policy retained as provenance; no physical cache selected"}
                if is_load:
                    if buffer.effect == "write":
                        fail(i, "load contradicts declared output write-only effect")
                    if stores and buffer.effect == "readwrite":
                        fail(i, "load after output store requires local memory-version sequencing")
                    value = Tile(record["id"], offset.width, mask if fill is None else Mask())
                    if fill is None and not mask.full(offset.width, programs):
                        record["undefined_lanes"] = "tracked through pointwise values; must be unobservable at every store"
                    actions.append((record["id"], name, access))
                    reads.append((i, access))
                else:
                    if buffer.effect == "read":
                        fail(i, "store contradicts declared read-only input effect")
                    if offset.constant != 0 or offset.program != offset.width:
                        fail(i, "output tiles must partition the output without overlap")
                    if buffer.effect == "write" and (
                            programs * offset.width < buffer.shape[0] or
                            not valid.subset_of(mask, offset.width, programs)):
                        fail(i, "write-only output mask/grid does not cover the output partition")
                    if not isinstance(args[1], Tile) or args[1].width != offset.width:
                        fail(i, "store value shape differs from its destination tile")
                    if not mask.subset_of(args[1].defined, offset.width, programs):
                        fail(i, "store may observe an undefined masked load lane; cannot invent a fill value")
                    stores.append((record["id"], access, args[1]))
                record.update(status="lowered", reason="proved bounded tensor tile access with exact predication")
            elif name in PAYLOAD:
                if len(args) != 2 or not all(isinstance(a, (Tile, Float)) for a in args):
                    fail(i, "elementwise operands must be f32 tiles/scalar splats")
                nonzero_widths = {a.width for a in args if a.width}
                if len(nonzero_widths) != 1:
                    fail(i, "elementwise operands must have a common tile width")
                width = nonzero_widths.pop()
                if str(op.get_result(0).get_type()) != f"tensor<{width}xf32>":
                    fail(i, "only f32 elementwise payload is implemented")
                defined = Mask()
                for operand in args:
                    if isinstance(operand, Tile):
                        defined = defined.intersect(operand.defined)
                value = Tile(record["id"], width, defined)
                actions.append((record["id"], name, tuple(args)))
                record.update(status="lowered", reason="elementwise scalar region in linalg.generic")
            else:
                fail(i, "live predicate/guard is unsupported; cannot drop an observed condition")
            if isinstance(value, Index):
                low, high = value.bounds(spec.grid[0])
                if low < -(1 << 31) or high >= (1 << 31):
                    fail(i, "index range does not prove absence of signed i32 wraparound")
            if value is not None:
                record["recovered_value"] = {"kind": type(value).__name__, **asdict(value)}
                env[op.get_result(0).id()] = value
        except (TypeError, ValueError, KeyError) as exc:
            if isinstance(exc, ConversionError):
                raise
            fail(i, f"unsupported operand/index form: {exc}")
    if len(stores) != 1 or len(widths) != 1:
        raise ConversionError("currently requires one store and a single common tile width")
    width = widths.pop()
    for i, access in reads:
        if buffers[access.pointer.buffer].effect == "readwrite":
            if access.pointer.offset != Index(0, width, 1, width):
                fail(i, "readwrite loads must stay within each program's independently owned output tile")
    return Analyzed(ttir, spec, records, env, actions, stores[0], width)
