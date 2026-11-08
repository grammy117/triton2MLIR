"""One tensor emission path with centrally stamped, serialization-safe provenance."""

from io import StringIO

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, scf, tensor
from xdsl.dialects.linalg import Linalg
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import FillOp, GenericOp, YieldOp
from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, DYNAMIC_INDEX, DenseArrayBase,
                                   FloatAttr, IndexType, ModuleOp, StringAttr, TensorType, f32, i64)
from xdsl.ir import Block, Dialect, Region
from xdsl.ir.affine import AffineMap
from xdsl.parser import Parser
from xdsl.printer import Printer

from .analysis import Float, PAYLOAD
from .indexing import Index
from .standard_ops import ForallOp, InParallelOp, ParallelInsertSliceOp, static_forall


def text(module):
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue() + "\n"


def context():
    ctx = Context()
    for dialect in (builtin.Builtin, arith.Arith, func.Func, Linalg):
        ctx.load_dialect(dialect)
    ctx.load_dialect(Dialect("scf", [*scf.Scf.operations, ForallOp, InParallelOp], list(scf.Scf.attributes)))
    ctx.load_dialect(Dialect("tensor", [*tensor.Tensor.operations, ParallelInsertSliceOp], list(tensor.Tensor.attributes)))
    return ctx


def parse(text_):
    module = Parser(context(), text_).parse_module()
    module.verify()
    return module


def emit(analyzed):
    spec = analyzed.spec
    inputs = [buffer for buffer in spec.buffers if buffer.effect != "write"]
    output = next(buffer for buffer in spec.buffers if buffer.effect != "read")
    whole = TensorType(f32, output.shape)
    block = Block(arg_types=[TensorType(f32, b.shape) for b in inputs])
    args = {b.name: arg for b, arg in zip(inputs, block.args, strict=True)}
    if output.effect == "readwrite":
        initial = args[output.name]
    else:
        empty = tensor.EmptyOp([], whole)
        block.add_op(empty)
        initial = empty.tensor
    grid_body = Block(arg_types=[IndexType(), whole])
    pid, shared = grid_body.args
    values, load_cache, index_cache, constant_tiles = {}, {}, {}, {}
    integer_constants, integer_values, integer_operations = {}, {}, {}
    current_sources, reused = [], []
    width = analyzed.tile_width
    tile_type = TensorType(f32, [width])

    def append(op, source_ids=None, *, family="layout", kind=None):
        op.attributes.update({"prov.family": StringAttr(family),
                              "prov.op": StringAttr(kind or op.name.split(".")[-1]),
                              "prov.source_ids": ArrayAttr(StringAttr(s) for s in
                                                          (current_sources if source_ids is None else source_ids))})
        grid_body.add_op(op)
        return op

    def const(number):
        if number not in integer_constants:
            value = append(arith.ConstantOp.from_int_and_width(number, IndexType())).result
            integer_constants[number] = value
            integer_values[value] = number
        return associate(integer_constants[number])

    def associate(value):
        owner = value.owner
        if hasattr(owner, "attributes"):
            prior = owner.attributes.get("prov.source_ids", ArrayAttr([]))
            ids = list(dict.fromkeys([*(item.data for item in prior), *current_sources]))
            owner.attributes["prov.source_ids"] = ArrayAttr(StringAttr(item) for item in ids)
        return value

    def binary(cls, a, b):
        first, second = integer_values.get(a), integer_values.get(b)
        functions = {arith.MuliOp: lambda x, y: x * y, arith.AddiOp: lambda x, y: x + y,
                     arith.SubiOp: lambda x, y: x - y, arith.MaxSIOp: max, arith.MinSIOp: min}
        if first is not None and second is not None:
            return const(functions[cls](first, second))
        if cls in (arith.MaxSIOp, arith.MinSIOp) and a is b:
            return associate(a)
        if cls == arith.MuliOp and (first == 0 or second == 0):
            return const(0)
        if (cls == arith.MuliOp and second == 1) or (cls in (arith.AddiOp, arith.SubiOp) and second == 0):
            return associate(a)
        if (cls == arith.MuliOp and first == 1) or (cls == arith.AddiOp and first == 0):
            return associate(b)
        key = cls, a, b
        if key not in integer_operations:
            integer_operations[key] = append(cls(a, b)).result
        return associate(integer_operations[key])

    def offset(index):
        base = Index(index.constant, index.program)
        if base not in index_cache:
            value = binary(arith.MuliOp, pid, const(base.program))
            if base.constant:
                value = binary(arith.AddiOp, value, const(base.constant))
            index_cache[base] = value
        return associate(index_cache[base])

    def slice_(source, off, size):
        dynamic = not isinstance(size, int)
        dimension = DYNAMIC_INDEX if dynamic else size
        return append(tensor.ExtractSliceOp.build(
            operands=[source, [off], [size] if dynamic else [], []],
            result_types=[TensorType(f32, [dimension])],
            properties={"static_offsets": DenseArrayBase.from_list(i64, [DYNAMIC_INDEX]),
                        "static_sizes": DenseArrayBase.from_list(i64, [dimension]),
                        "static_strides": DenseArrayBase.from_list(i64, [1])})).result

    def interval(access):
        """Recover safe source offset, active lane start and active length."""
        base = offset(access.pointer.offset)
        if access.mask.full(width, spec.grid[0]):
            return base, const(0), width
        low, high = const(0), const(width)
        for bound in access.mask.lower:
            low = binary(arith.MaxSIOp, low, offset(bound))
        low = binary(arith.MinSIOp, low, const(width))
        for bound in access.mask.upper:
            high = binary(arith.MinSIOp, high, offset(bound))
        high = binary(arith.MaxSIOp, low, high)
        size = binary(arith.SubiOp, high, low)
        address = binary(arith.AddiOp, base, low)
        # An empty active slice can have an invalid nominal base. No address
        # from a disabled source lane is extracted, even for all-inactive tiles.
        safe = binary(arith.MaxSIOp, const(0), address)
        shape = next(b.shape[0] for b in spec.buffers if b.name == access.pointer.buffer)
        safe = binary(arith.MinSIOp, const(shape), safe)
        return safe, low, size

    def materialize(value):
        if not isinstance(value, Float):
            return values[value.source_id]
        if value.bits not in constant_tiles:
            scalar = append(arith.ConstantOp(FloatAttr(value.value, f32)), family="fill")
            init = append(tensor.EmptyOp([], tile_type))
            fill = append(FillOp([scalar.result], [init.tensor], res=[tile_type]), family="fill")
            constant_tiles[value.bits] = fill.results[0]
        return constant_tiles[value.bits]

    for source_id, name, payload in analyzed.actions:
        current_sources[:] = [source_id]
        if name == "tt.load":
            if payload in load_cache:
                value, original = load_cache[payload]
                defining = value.owner
                prior = defining.attributes["prov.source_ids"]
                defining.attributes["prov.source_ids"] = ArrayAttr([*prior, StringAttr(source_id)])
                values[source_id] = value
                reused.append({"source_id": source_id, "reuses": original})
                continue
            off, low, size = interval(payload)
            value = slice_(args[payload.pointer.buffer], off, size)
            if not isinstance(size, int):
                high = binary(arith.SubiOp, const(width), binary(arith.AddiOp, low, size))
                pad_body = Block(arg_types=[IndexType()])
                # Missing source other permits an arbitrary value only after the
                # analysis proved these lanes cannot reach an observed store.
                fill = arith.ConstantOp(FloatAttr(payload.fill.value if payload.fill is not None else 0.0, f32))
                pad_body.add_ops([fill, tensor.YieldOp(fill.result)])
                # xDSL 0.68 requires a dynamic result for dynamic pad operands.
                # The proved low+size+high=BLOCK relation justifies the cast.
                padded = append(tensor.PadOp(value, [low], [high], Region(pad_body),
                                            [DYNAMIC_INDEX], [DYNAMIC_INDEX],
                                            TensorType(f32, [DYNAMIC_INDEX])))
                value = append(tensor.CastOp.build(operands=[padded.result], result_types=[tile_type])).results[0]
            values[source_id] = value
            load_cache[payload] = value, source_id
        else:
            input_values = [materialize(item) for item in payload]
            init = append(tensor.EmptyOp([], tile_type))
            scalar = Block(arg_types=[f32, f32, f32])
            cls = {"arith.addf": arith.AddfOp, "arith.subf": arith.SubfOp,
                   "arith.mulf": arith.MulfOp}[name]
            calculation = cls(scalar.args[0], scalar.args[1])
            scalar.add_ops([calculation, YieldOp(calculation.result)])
            op = GenericOp(inputs=input_values, outputs=[init.tensor], body=Region(scalar),
                           indexing_maps=[AffineMapAttr(AffineMap.identity(1))] * 3,
                           iterator_types=[IteratorTypeAttr.parallel()], result_types=[tile_type])
            values[source_id] = append(op, family="elementwise", kind=PAYLOAD[name]).results[0]
    store_id, access, tile = analyzed.output
    current_sources[:] = [store_id]
    off, low, size = interval(access)
    contribution = values[tile.source_id]
    if not isinstance(size, int):
        contribution = slice_(contribution, low, size)
    insertion = ParallelInsertSliceOp.build(
        operands=[contribution, shared, [off], [size] if not isinstance(size, int) else [], []],
        properties={"static_offsets": DenseArrayBase.from_list(i64, [DYNAMIC_INDEX]),
                    "static_sizes": DenseArrayBase.from_list(i64, [size if isinstance(size, int) else DYNAMIC_INDEX]),
                    "static_strides": DenseArrayBase.from_list(i64, [1])},
        attributes={"prov.source_ids": ArrayAttr([StringAttr(store_id)])})
    grid_body.add_op(InParallelOp.build(regions=[Region(Block([insertion]))]))
    grid = static_forall(spec.grid[0], initial, grid_body)
    grid.attributes.update({"prov.grid": DenseArrayBase.from_list(i64, spec.grid),
                            "prov.tile_shape": DenseArrayBase.from_list(i64, [width]),
                            "prov.independence": StringAttr("proved-disjoint-output-tiles")})
    block.add_ops([grid, func.ReturnOp(*grid.results)])
    function = func.FuncOp.from_region("forward", [arg.type for arg in block.args], [whole], Region(block))
    function.attributes.update({"prov.source_entry": StringAttr(spec.entry),
                                "prov.kernel": StringAttr(f"triton::{spec.entry}")})
    module = ModuleOp([function], attributes={"prov.level": StringAttr("linalg-on-tensors"),
                    "prov.frontend": StringAttr("triton"), "prov.intent": StringAttr(spec.intent),
                    "prov.intent_origin": StringAttr("caller-description"),
                    "prov.spec_sha256": StringAttr(spec.digest),
                    "prov.source_sha256": StringAttr(analyzed.ttir.digest)})

    def stamp(op, inherited_sources=(), inherited_family="layout", inherited_kind=None):
        ids = op.attributes.get("prov.source_ids", ArrayAttr(StringAttr(s) for s in inherited_sources))
        op.attributes.setdefault("prov.source_ids", ids)
        op.attributes.setdefault("prov.region_id", StringAttr(spec.entry))
        op.attributes.setdefault("prov.family", StringAttr(inherited_family))
        op.attributes.setdefault("prov.op", StringAttr(inherited_kind or op.name.split(".")[-1]))
        child_kind = op.attributes["prov.op"].data if isinstance(op, GenericOp) else inherited_kind
        for region in op.regions:
            for child_block in region.blocks:
                for child in child_block.ops:
                    stamp(child, [s.data for s in ids], op.attributes["prov.family"].data, child_kind)
    stamp(module)
    correspondence = []
    for i, op in enumerate(module.walk()):
        emission_id = f"mlir.op.{i:04d}"
        op.attributes["prov.emission_id"] = StringAttr(emission_id)
        correspondence.append({"id": emission_id, "opcode": op.name,
                               "source_ids": [a.data for a in op.attributes["prov.source_ids"]]})
    module.verify()
    parse(text(module))
    transformations = [{"name": "reuse-identical-tensor-reads", "reused_reads": reused,
                         "basis": "nonvolatile independent buffers, owned-tile readwrite reads precede sole store"}]
    return module, correspondence, transformations

