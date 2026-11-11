"""Turn scalar TTIR maps into parallel tensor computation, retaining the body.

Triton 3.7.1 defines pack=1 as one scalar subregion invocation per element.
For larger packs, identical slot-local scalar computations are independent of the
physical register grouping. Certify that fact before using the same expansion.
The shared Meta/Microsoft arithmetic converters do not handle this region op.
Use standard linalg generalization: identity maps, parallel iterators and the
original inlined scalar region, including its structured control flow.
"""

from collections import ChainMap
import math
import struct

from xdsl.dialects import arith, builtin, scf, tensor
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
    """Certify slot-local congruent computations without choosing a register layout.

    Intern typed SSA terms, separately retaining their actual slot dependency.
    Equal terms alone are insufficient: a0+a1 and a1+a0 must not become 2*a.
    Every executable attribute/property, operand order and result ordinal is
    part of a term. Location/provenance describe terms, not their semantics.
    Structured regions retain their binders, execution order and full bodies.
    No memory event, unproved termination or unknown effect is discarded.
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

    def operation_key(child, environment, path=()):
        # Region arguments use lexical binders, never capture identities. Equal
        # loops/branches in distinct slots therefore compare modulo SSA names,
        # while an outer induction variable cannot alias an inner one.
        control = isinstance(child, (scf.IfOp, scf.ForOp, scf.WhileOp))
        terminal = isinstance(child, (scf.YieldOp, scf.ConditionOp))
        # xDSL 0.68 omits purity on this exact upstream pure integer cast.
        # These exact SCF operations have only recursive memory effects; every
        # nested operation is checked below instead of repeatedly walking them.
        pure = control or is_side_effect_free(child) or isinstance(child, arith.ExtSIOp)
        if (not control and (child.regions or not pure
                or opcode(child).split(".")[0] not in ("arith", "math") and not terminal)):
            refuse("packed elementwise maps need an explicit logical grouping contract unless pure scalar computation proves slot independence", child)
        captures = set()
        operands = []
        for value in child.operands:
            if value in environment:
                operands.append(environment[value])
            else:
                operands.append(term(value))
                captures.add(value)
        regions = []
        for region_index, region in enumerate(child.regions):
            if len(region.blocks) != 1:
                refuse("packed scalar control requires single-block structured regions", child)
            block = region.block
            nested_path = (*path, region_index)
            local = environment.new_child()
            for index, arg in enumerate(block.args):
                local[arg] = intern(("region-argument", nested_path, index, arg.type))
            body = []
            for position, nested in enumerate(block.ops):
                key, free = operation_key(nested, local, (*nested_path, position))
                captures.update(free)
                body.append(intern(("region-operation", key)))
                for index, value in enumerate(nested.results):
                    local[value] = intern(("operation", key, index))
            regions.append((tuple(arg.type for arg in block.args), tuple(body)))
        semantic_attributes = tuple(sorted((key, _attribute_key(value)) for key, value in child.attributes.items()
                                           if not key.startswith("prov.")))
        key = (opcode(child), tuple(operands), tuple(v.type for v in child.results), semantic_attributes,
               tuple(sorted((k, _attribute_key(v)) for k, v in child.properties.items())), tuple(regions))
        return key, captures

    operations = list(source.ops)[:-1]
    uses = {}
    for child in operations:
        key, captures = operation_key(child, ChainMap())
        uses[child] = captures
        slots = {dependencies[v] for v in captures} - {None}
        dependency = next(iter(slots)) if len(slots) == 1 else (None if not slots else -1)
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
            needed.update(uses[child])
            representatives.append(child)
    representatives.reverse()
    live = set(end.operands)
    dead = []
    for child in reversed(operations):
        if any(value in live for value in child.results):
            live.update(uses[child])
        else:
            if child.regions:
                refuse("unused packed scalar control needs its execution/termination retained, not dead arithmetic elimination", child)
            dead.extend(source_ids(child.location))
    equivalents = {}
    for group in nodes.values():
        # Matching complete region syntax establishes correspondence at every
        # nested position, including conditions, yields and carried values.
        for corresponding in zip(*(tuple(child.walk()) for child in group), strict=True):
            locations = builtin.FusedLoc(builtin.ArrayAttr(child.location for child in corresponding), builtin.NoneAttr())
            for child in corresponding:
                equivalents[child] = locations
    structured = any(child.regions for child in operations)
    proof = {"method": "typed-structured-ssa-congruence-v1" if structured else "typed-scalar-ssa-congruence-v1",
        "grouping": "independent-of-register-layout",
        "source_scalar_operations": len(operations), "representative_scalar_operations": len(representatives),
        **({"source_nested_operations": sum(len(tuple(child.walk())) - 1 for child in operations)} if structured else {}),
        "discarded_pure_dead_source_ids": sorted(set(dead)),
        "outputs": [{"tensor_result": index, "scalar_slots": pack,
                     "source_ids": sorted({identity for value in end.operands[index * pack:(index + 1) * pack]
                                           for identity in source_ids(getattr(value.owner, "location", end.location))})}
                    for index in range(len(selected))]}
    return representatives, selected, equivalents, proof


def expand_elementwise_maps(module, source_ids, *, memory_events=False, layouts=None):
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
        if pack > 1 and layouts is not None:
            from .grouped_map import expand
            from .map_layout import KEY
            key = op.attributes.get(KEY)
            if not isinstance(key, StringAttr) or key.data not in layouts:
                refuse("packed map lacks its source-derived logical grouping")
            proof = expand(op, source, end, pack, layouts[key.data], refuse, memory_events=memory_events)
            reports.append({"kind": "ttir-elementwise-map", "source_ids": ids, "pack": pack,
                "shape": list(shape), "inputs": len(inputs), "results": len(outputs),
                "expansion": "parallel-logical-groups-with-original-scalar-region-and-coordinate-restoration",
                **({"memory": "isolated-source-word-events-in-original-scalar-region"} if has_memory else {}),
                "physical_mapping": "undecided", "packing_proof": proof})
            continue
        children, returned, locations, packing_proof = list(source.ops)[:-1], list(end.operands), {}, None
        if pack > 1:
            children, returned, locations, packing_proof = _independent_packing(source, end, pack, refuse, source_ids)
        body = Block(arg_types=input_elements + output_elements)
        mapping = {arg: body.args[i // pack] for i, arg in enumerate(source.args)}
        for child in children:
            cloned = child.clone(value_mapper=mapping)
            for original, copy in zip(child.walk(), cloned.walk(), strict=True):
                copy.location = locations.get(original, original.location)
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
