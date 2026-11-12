"""Optional source-language implementations; portable core has no ISA cases.

Providers implement the complete selected source snippet as a typed scalar DAG.
The adapter checks its value/effect/shape boundary and emits standard MLIR.
Source-language equivalence belongs to the named provider/specification and its
independent qualifications, not to an annotation or the source `pure` flag.
"""

import hashlib
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path

from xdsl.dialects import arith, builtin, tensor
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Operation, Region
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import Rewriter
from xdsl.traits import is_side_effect_free

from .diagnostics import ConversionError

GROUP = "triton2mlir.source_assembly.v1"
_LOADED = {}


class AssemblyRefusal(ValueError):
    """The selected source implementation cannot represent this entire snippet."""


@dataclass(frozen=True)
class AssemblyRequest:
    assembly: str
    constraints: str
    input_types: tuple
    output_types: tuple
    pack: int


@dataclass
class AssemblyExpansion:
    operations: list
    outputs: list
    contract: dict


def load_provider(name):
    if not isinstance(name, str) or not name:
        raise ValueError("source assembly requires an explicit installed provider name")
    matches = list(entry_points(group=GROUP, name=name))
    if len(matches) != 1:
        raise ValueError(f"source assembly provider {name!r} must have exactly one installed {GROUP} entry point")
    point, = matches
    distribution = point.dist
    if distribution is None or distribution.files is None:
        raise ValueError("source assembly provider needs installed distribution custody")
    files = {}
    for member in distribution.files:
        if member.suffix != ".py" and member.name not in ("entry_points.txt", "METADATA"):
            continue
        path = Path(distribution.locate_file(member))
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("source assembly provider custody requires bounded regular files")
        data = path.read_bytes()
        files[str(member)] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    if not any(path.endswith(".py") for path in files):
        raise ValueError("source assembly provider has no recorded Python implementation")
    identity = {"entry_point": name, "implementation": point.value,
        "distribution": distribution.metadata["Name"], "version": distribution.version, "files": files}
    if name in _LOADED:
        provider, previous = _LOADED[name]
        if identity != previous:
            raise ValueError("source assembly provider files changed after import; restart the compiler with a coherent installation")
        return provider, identity
    provider = point.load()()
    if getattr(provider, "api_version", None) != 1 or not callable(getattr(provider, "lower", None)):
        raise ValueError("source assembly provider does not implement API v1")
    _LOADED[name] = provider, identity
    # Reopen after import too; imported code cannot silently inherit identities
    # from a changed installation, even when Python retains a cached module.
    return load_provider(name)


def expand_source_assembly(module, source_ids, *, provider, tensor_carrier="linalg"):
    from .shared_tensor import opcode

    if tensor_carrier not in ("linalg", "arith"):
        raise ValueError("source assembly requires an explicit standard computation carrier")
    implementation, identity = load_provider(provider)
    reports = []
    for source in list(module.walk()):
        if opcode(source) != "tt.elementwise_inline_asm":
            continue
        ids = source_ids(source.location)

        def refuse(message, source=source, ids=ids):
            raise ConversionError(message, opcode=opcode(source), source_id=next(iter(ids), None))

        fields = source.properties
        assembly, constraints = fields.get("asm_string"), fields.get("constraints")
        pure, pack = fields.get("pure"), fields.get("packed_element")
        if (not isinstance(assembly, builtin.StringAttr) or not assembly.data or len(assembly.data.encode()) > 65536
                or not isinstance(constraints, builtin.StringAttr) or len(constraints.data.encode()) > 4096
                or not isinstance(pure, builtin.IntegerAttr) or pure.value.data == 0):
            refuse("source assembly requires a bounded snippet and an explicitly pure source contract; hardware/effect observations need another interface")
        if not isinstance(pack, builtin.IntegerAttr) or pack.value.data != 1:
            refuse("packed source assembly needs established register grouping and packing semantics")
        operands, outputs = list(source.operands), [v.type for v in source.results]
        if not outputs:
            refuse("pure source assembly must have declared value results")
        values = [v.type for v in operands] + outputs
        numeric = lambda t: isinstance(t, (builtin.IntegerType, builtin.AnyFloat))
        if not all(numeric(t.element_type if isinstance(t, builtin.TensorType) else t) for t in values):
            refuse("source assembly value interfaces must be numeric and cannot expose physical addresses")
        shapes = [t.get_shape() for t in values if isinstance(t, builtin.TensorType)]
        shape = shapes[0] if shapes else None
        if shapes and (any(n <= 0 for n in shape) or any(s != shape for s in shapes)
                       or not all(isinstance(t, builtin.TensorType) for t in outputs)):
            refuse("source assembly tensors require matching static tile shapes and tensor results")
        input_types = tuple(v.type.element_type if isinstance(v.type, builtin.TensorType) else v.type for v in operands)
        output_types = tuple(t.element_type if isinstance(t, builtin.TensorType) else t for t in outputs)
        tensor_inputs = [v for v in operands if isinstance(v.type, builtin.TensorType)]
        body = Block(arg_types=[v.type.element_type for v in tensor_inputs] + list(output_types)) if shapes else None
        arguments = []
        index = 0
        for value in operands:
            if isinstance(value.type, builtin.TensorType):
                arguments.append(body.args[index])
                index += 1
            else:
                arguments.append(value)
        request = AssemblyRequest(assembly.data, constraints.data, input_types, output_types, 1)
        try:
            expansion = implementation.lower(request, arguments)
        except AssemblyRefusal as exc:
            refuse(str(exc))
        if load_provider(provider)[1] != identity:
            refuse("source assembly provider changed during expansion")
        if not isinstance(expansion, AssemblyExpansion) or len(expansion.operations) > 16384:
            refuse("source assembly provider returned an invalid or oversized expansion")
        known = set(arguments)
        for op in expansion.operations:
            if (not isinstance(op, Operation) or op.parent is not None or op.regions or op.successors
                    or op.name.split(".")[0] not in ("arith", "math")
                    or not (is_side_effect_free(op) or type(op) is arith.ExtSIOp)
                    or any(v not in known for v in op.operands)
                    or not all(numeric(v.type) for v in [*op.operands, *op.results])):
                refuse("source assembly provider must return a closed effect-free standard scalar DAG")
            known.update(op.results)
            op.location = source.location
        if ([v.type for v in expansion.outputs] != list(output_types)
                or any(v not in known for v in expansion.outputs) or not isinstance(expansion.contract, dict)):
            refuse("source assembly provider changed result types, arity or value custody")
        emitted = expansion.operations
        if shapes and tensor_carrier == "arith":
            from .tensor_dag import lift
            try:
                emitted, results = lift(emitted, dict(zip(arguments, operands, strict=True)), expansion.outputs, outputs[0])
            except ValueError as error:
                refuse(str(error))
            if [value.type for value in results] != outputs:
                refuse("source assembly tensor arithmetic changed its declared result interface")
            for operation in emitted:
                operation.location = source.location
                operation.attributes["prov.composite_family"] = builtin.StringAttr("source_assembly")
            Rewriter.replace_op(source, emitted, results)
        elif shapes:
            body.add_ops(emitted)
            body.add_op(YieldOp(*expansion.outputs))
            empty = [tensor.EmptyOp([], t) for t in outputs]
            generic = GenericOp(tensor_inputs, [v.tensor for v in empty], Region(body),
                [builtin.AffineMapAttr(AffineMap.identity(len(shape)))] * (len(tensor_inputs) + len(outputs)),
                [IteratorTypeAttr.parallel()] * len(shape), outputs)
            generic.location = source.location
            generic.attributes["prov.composite_family"] = builtin.StringAttr("source_assembly")
            for value in empty:
                value.location = source.location
            Rewriter.replace_op(source, [*empty, generic])
        else:
            Rewriter.replace_op(source, emitted, expansion.outputs)
        reports.append({"kind": "source-assembly-implementation", "source_ids": ids,
            "provider": identity, "assembly_sha256": hashlib.sha256(assembly.data.encode()).hexdigest(),
            "constraints": constraints.data, "packed_elements": 1, "contract": expansion.contract,
            "input_types": [str(t) for t in input_types], "output_types": [str(t) for t in output_types],
            "expansion": ("standard-elementwise-arith" if tensor_carrier == "arith" else "standard-scalar-dag-in-parallel-linalg") if shapes else "standard-scalar-dag",
            "physical_mapping": "undecided", "source_equivalence": "provider-specification-and-qualified-cases; not-universal-proof"})
    return reports
