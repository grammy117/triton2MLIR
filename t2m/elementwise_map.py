"""Turn scalar TTIR maps into parallel tensor computation, retaining the body.

Triton 3.7.1 defines pack=1 as one scalar subregion invocation per element.
For larger packs, identical slot-local scalar DAGs are independent of the
physical register grouping. Certify that fact before using the same expansion.
The shared Meta/Microsoft arithmetic converters do not handle this region op.
Use standard linalg generalization: identity maps, parallel iterators and the
original inlined scalar region, including its structured control flow.
"""

import math
import struct

from xdsl.dialects import arith, builtin, tensor
from xdsl.dialects.builtin import AffineMapAttr, IntegerAttr, StringAttr, TensorType
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, ParametrizedAttribute, Region
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import Rewriter
from xdsl.traits import is_side_effect_free

from .diagnostics import ConversionError


def _attribute_key(value):
    # xDSL FloatData equality identifies +0 with -0 and all NaNs. Semantic
    # congruence must retain the actual represented words, including nested
    # attribute parameters; ordinary Python/Attribute equality is insufficient.
    if isinstance(value, builtin.FloatData):
        return type(value), struct.pack(">d", value.data)
    if isinstance(value, ParametrizedAttribute):
        return type(value), tuple(_attribute_key(p) for p in value.parameters)
    if isinstance(value, builtin.ArrayAttr):
        return type(value), tuple(_attribute_key(p) for p in value)
    if isinstance(value, builtin.DictionaryAttr):
        return type(value), tuple(sorted((k, _attribute_key(v)) for k, v in value.data.items()))
    return value


def _independent_packing(source, end, pack, refuse, source_ids):
    """Certify slot-local congruent DAGs without selecting a register layout.

    Intern typed SSA terms, separately retaining their actual slot dependency.
    Equal terms alone are insufficient: a0+a1 and a1+a0 must not become 2*a.
    Every executable attribute/property, operand order and result ordinal is
    part of a term. Location/provenance describe terms, not their semantics.
    No memory event, region or unknown effect is discarded by this proof.
    """
    from .shared_tensor import opcode

    terms, dependencies, interned, nodes = {}, {}, {}, {}

    def intern(key):
        return interned.setdefault(key, len(interned))

    for i, arg in enumerate(source.args):
        terms[arg] = intern(("argument", i // pack, arg.type))
        dependencies[arg] = i % pack

    def term(value):
        if value not in terms:
            # Scalar values captured from outside this invocation are uniform
            # across its slots. Identity prevents merging distinct captures.
            terms[value] = intern(("capture", value))
            dependencies[value] = None
        return terms[value]

    operations = list(source.ops)[:-1]
    for child in operations:
        # xDSL 0.68 omits the effect trait on ExtSIOp. Upstream MLIR's
        # Arith_IToICastOp inherits Pure through Arith_CastOp; admit that exact
        # registered class rather than treating unknown dialect ops as pure.
        pure = is_side_effect_free(child) or isinstance(child, arith.ExtSIOp)
        if child.regions or opcode(child).split(".")[0] not in ("arith", "math") or not pure:
            refuse("packed elementwise maps need an explicit logical grouping contract unless a pure scalar DAG proves slot independence", child)
        operands = tuple(term(v) for v in child.operands)
        slots = {dependencies[v] for v in child.operands} - {None}
        dependency = next(iter(slots)) if len(slots) == 1 else (None if not slots else -1)
        semantic_attributes = tuple(sorted((key, _attribute_key(value)) for key, value in child.attributes.items()
                                           if not key.startswith("prov.")))
        key = (opcode(child), operands, tuple(v.type for v in child.results),
               semantic_attributes, tuple(sorted((k, _attribute_key(v)) for k, v in child.properties.items())))
        # A dead mixed-slot node can have the same normalized term as a
        # slot-local node. It is not an equivalent source location.
        if dependency != -1:
            nodes.setdefault(key, []).append(child)
        for index, result in enumerate(child.results):
            terms[result] = intern(("operation", key, index))
            dependencies[result] = dependency

    selected = list(end.operands[::pack])
    for output in range(len(selected)):
        values = end.operands[output * pack:(output + 1) * pack]
        for slot, value in enumerate(values):
            term(value)
            if dependencies[value] not in (None, slot) or terms[value] != terms[values[0]]:
                refuse("packed elementwise maps need an explicit logical grouping contract: output slots are coupled or compute different scalar functions")

    # Retain only the representative outputs' dependency closure. Pure dead
    # arithmetic (including unused divisions) must not become speculative
    # executable work after different slots share one scalar formal.
    needed = set(selected)
    representatives = []
    for child in reversed(operations):
        if any(value in needed for value in child.results):
            needed.update(child.operands)
            representatives.append(child)
    representatives.reverse()
    live = set(end.operands)
    dead = []
    for child in reversed(operations):
        if any(value in live for value in child.results):
            live.update(child.operands)
        else:
            dead.extend(source_ids(child.location))
    equivalents = {}
    for group in nodes.values():
        locations = builtin.FusedLoc(builtin.ArrayAttr(child.location for child in group), builtin.NoneAttr())
        for child in group:
            equivalents[child] = locations
    proof = {"method": "typed-scalar-ssa-congruence-v1", "grouping": "independent-of-register-layout",
        "source_scalar_operations": len(operations), "representative_scalar_operations": len(representatives),
        "discarded_pure_dead_source_ids": sorted(set(dead)),
        "outputs": [{"tensor_result": index, "scalar_slots": pack,
                     "source_ids": sorted({identity for value in end.operands[index * pack:(index + 1) * pack]
                                           for identity in source_ids(getattr(value.owner, "location", end.location))})}
                    for index in range(len(selected))]}
    return representatives, selected, equivalents, proof


def expand_elementwise_maps(module, source_ids, *, memory_events=False):
    from .shared_tensor import _pointer_type, opcode
    reports = []
    for op in list(module.walk()):
        if opcode(op) != "tt.map_elementwise":
            continue
        ids = source_ids(op.location)

        def refuse(message, source=op):
            raise ConversionError(message, opcode=opcode(source), source_id=next(iter(source_ids(source.location)), None))

        pack = op.properties.get("pack")
        if not isinstance(pack, IntegerAttr) or pack.value.data < 1:
            refuse("elementwise maps require a positive integer pack")
        pack = pack.value.data
        inputs, outputs = [v.type for v in op.operands], [v.type for v in op.results]
        if not inputs or not outputs or not all(isinstance(t, TensorType) for t in inputs + outputs):
            refuse("elementwise maps require ranked numeric input/result tensors")
        shape = inputs[0].get_shape()
        if (any(n <= 0 for n in shape) or any(t.get_shape() != shape for t in inputs + outputs)
                or not all(isinstance(t.element_type, (builtin.IntegerType, builtin.AnyFloat)) for t in inputs + outputs)):
            refuse("elementwise map tensor shapes must match and elements must be numeric; source memory needs its own effect adapter")
        if math.prod(shape) % pack:
            refuse("elementwise map pack must divide its logical element count")
        if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
            refuse("elementwise maps need one inlined scalar region")
        source = op.regions[0].block
        has_memory = any(opcode(child) == "tt.load" for child in source.walk())
        end = source.last_op
        input_elements, output_elements = [t.element_type for t in inputs], [t.element_type for t in outputs]
        if ([v.type for v in source.args] != [t for t in input_elements for _ in range(pack)] or end is None
                or opcode(end) != "tt.map_elementwise.return"
                or [v.type for v in end.operands] != [t for t in output_elements for _ in range(pack)]):
            refuse("elementwise map scalar signature differs from its tensor elements")
        controls = {"scf.if", "scf.for", "scf.while", "scf.yield", "scf.condition"}
        for child in source.walk():
            name = opcode(child)
            if child is end:
                continue
            memory = memory_events and name in ("tt.load", "tt.addptr", "tt.bitcast")
            if name.split(".")[0] not in ("arith", "math") and name not in controls and not memory:
                refuse("elementwise map body requires inlined scalar arithmetic/math and structured value control", child)
            values = [*child.operands, *child.results,
                      *(arg for region in child.regions for block in region.blocks for arg in block.args)]
            if not all(isinstance(v.type, (builtin.IntegerType, builtin.IndexType, builtin.AnyFloat))
                       or memory_events and _pointer_type(v.type) and not isinstance(v.type, TensorType) for v in values):
                refuse("elementwise map scalar body cannot access tensors, addresses or effects", child)
        children, returned, locations, packing_proof = list(source.ops)[:-1], list(end.operands), {}, None
        if pack > 1:
            children, returned, locations, packing_proof = _independent_packing(source, end, pack, refuse, source_ids)
        body = Block(arg_types=input_elements + output_elements)
        mapping = {arg: body.args[i // pack] for i, arg in enumerate(source.args)}
        for child in children:
            cloned = child.clone(value_mapper=mapping)
            cloned.location = locations.get(child, child.location)
            body.add_op(cloned)
        terminal = YieldOp(*(mapping.get(v, v) for v in returned))
        terminal.location = end.location
        body.add_op(terminal)
        empty = [tensor.EmptyOp([], t) for t in outputs]
        for value in empty:
            value.location = op.location
        generic = GenericOp(list(op.operands), [v.tensor for v in empty], Region(body),
            [AffineMapAttr(AffineMap.identity(len(shape)))] * (len(inputs) + len(outputs)),
            [IteratorTypeAttr.parallel()] * len(shape), outputs)
        generic.location = op.location
        generic.attributes.update({"prov.composite_family": StringAttr("map_elementwise"),
            "prov.composite_id": StringAttr("map_elementwise:" + (ids[0] if ids else str(op.location))),
            "prov.map_pack": IntegerAttr(pack, builtin.i32)})
        if packing_proof:
            generic.attributes["prov.map_grouping"] = StringAttr("proved-slot-independent-v1")
        Rewriter.replace_op(op, [*empty, generic])
        reports.append({"kind": "ttir-elementwise-map", "source_ids": ids, "pack": pack,
            "shape": list(shape), "inputs": len(inputs), "results": len(outputs),
            "expansion": "parallel-linalg-generic-with-original-scalar-region", "physical_mapping": "undecided",
            **({"memory": "isolated-source-word-events-in-original-scalar-region"} if has_memory else {}),
            **({"packing_proof": packing_proof} if packing_proof else {})})
    return reports
